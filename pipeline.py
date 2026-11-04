"""Threadline Discovery pipeline: propose -> verify -> score -> visualize.

Built during the Meta Global AI Developer Hackathon.
Every model call goes through Meta Model API (Muse Spark for text and vision,
Muse Image for concept visuals).

Usage:
    python pipeline.py              # full run, with concept images
    python pipeline.py --no-images  # skip image generation (cheaper, faster)
"""
import base64
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE = Path(__file__).parent
SIGNALS_PATH = BASE / "data" / "signals_public.json"
RESULTS_PATH = BASE / "results.json"
CONCEPT_DIR = BASE / "static" / "concepts"

TEXT_MODEL = "muse-spark-1.3"
IMAGE_MODEL = "muse-image-1.0"
# Second verifier, tried in this order: Muse Spark Open (open weights) on a hosted partner,
# then Muse Wonder (a different hosted Meta model), then Muse Spark checking its own work.
OPEN_PROVIDERS = [
    {"name": "Baseten", "env": "BASETEN_API_KEY", "url": "https://inference.baseten.co/v1",
     "model": "meta-models/muse-spark-open-1026"},
    {"name": "Fireworks", "env": "FIREWORKS_API_KEY", "url": "https://api.fireworks.ai/inference/v1",
     "model": "accounts/fireworks/models/muse-spark-open"},
]
SECOND_MODEL = "muse-wonder-1.0"   # independent hosted Meta model, same Meta Model API key
OPEN_MAX_TOKENS = 6000    # Muse Spark Open always reasons, so leave room for the final answer
OPEN_REASONING = "medium"
CONDITION = "post-mastectomy"
MAX_OPPS = 6          # opportunities the proposer may return
MAX_IMAGES = 6        # concept visuals generated per run (top ranked)
IMAGE_ATTEMPTS = 2    # generate, check, regenerate once if features are missing

# Score weights (total 100). Computed in code, never asserted by the model.
W_EVIDENCE, W_SOURCES, W_VERIFIER = 40, 30, 30
EVIDENCE_FULL = 6     # distinct cited signals needed for full evidence points
SOURCES_FULL = 3      # distinct threads needed for full source points

GEN_SYSTEM = (
    "You are a market analyst for adaptive fashion brands. Using ONLY the numbered "
    "consumer signals provided, identify distinct product opportunities. Each must be a "
    "specific, buildable garment or accessory, not general advice. Opportunities must "
    "address different problems, so do not return near-duplicates. Cite the ids of the "
    "signals that support each one, preferring signals from different sources. Never "
    "invent facts or ids. Keep each concept to one plain sentence that describes the "
    "garment itself, built from what the cited signals ask for, without adding design "
    "details nobody mentioned and without naming body parts, scars, surgery, or medical "
    "terms. required_features lists 2 to 4 short features that can be SEEN in a product "
    "photo, such as closures, pockets, neckline, straps, or panel layout. Never list "
    "feel or performance claims such as softness, compression, or coverage, because a "
    "photo cannot show them. Each feature must be directly supported by a cited signal."
)
VERIFY_SYSTEM = (
    "You are a strict but fair fact-checker. You receive one opportunity and ONLY the "
    "signals it cites. Use no outside knowledge. Check three things separately: (1) the "
    "pain_point is stated in the signals, (2) the unmet_need is stated or clearly "
    "implied by the signals, and (3) every item in required_features is traceable to "
    "something the signals say or clearly imply. Ordinary wording differences are fine, "
    "and a feature counts as traceable if a signal describes the need it serves. The "
    "concept sentence is a designer's summary, so do not reject it for ordinary design "
    "detail, only check required_features. Mark supported=false only if a check clearly "
    "fails, for example a required feature that no signal supports, a claim that "
    "contradicts the signals, or evidence from a single voice for a broad claim. In "
    "reason, name the check that failed, or state the strongest supporting evidence. "
    "confidence is an integer from 1 (barely supported) to 10 (clearly and directly "
    "supported). Use the full range."
)
CHECK_SYSTEM = (
    "You inspect a product concept image. For each requested feature, say whether it is "
    "clearly visible in the image. Be strict: if you are unsure, answer false."
)

GEN_SCHEMA = {
    "type": "object",
    "properties": {"opportunities": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "pain_point": {"type": "string"},
            "unmet_need": {"type": "string"},
            "concept": {"type": "string"},
            "required_features": {"type": "array", "items": {"type": "string"}},
            "evidence_ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["title", "pain_point", "unmet_need", "concept",
                     "required_features", "evidence_ids"],
        "additionalProperties": False,
    }}},
    "required": ["opportunities"],
    "additionalProperties": False,
}
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "confidence": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["supported", "confidence", "reason"],
    "additionalProperties": False,
}
CHECK_SCHEMA = {
    "type": "object",
    "properties": {"features": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "visible": {"type": "boolean"}},
        "required": ["name", "visible"],
        "additionalProperties": False,
    }}},
    "required": ["features"],
    "additionalProperties": False,
}

_client = None


def client():
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://api.meta.ai/v1",
                         api_key=os.environ["MODEL_API_KEY"])
    return _client


def _extract_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model reply")
    return json.loads(text[start:end + 1])


def ask(system, user, name, schema, retries=2, image_data_url=None, model=None):
    """Call Muse Spark and return parsed JSON. First try is schema-constrained,
    later tries fall back to a plain JSON-only instruction."""
    content = user
    if image_data_url:
        content = [{"type": "text", "text": user},
                   {"type": "image_url", "image_url": {"url": image_data_url}}]
    last = None
    for attempt in range(retries + 1):
        try:
            if attempt == 0:
                msgs = [{"role": "system", "content": system},
                        {"role": "user", "content": content}]
                r = client().chat.completions.create(
                    model=model or TEXT_MODEL, messages=msgs,
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": name, "schema": schema}})
            else:
                hint = (system + "\nReply with ONLY a JSON object matching this schema, "
                        "no prose and no code fences: " + json.dumps(schema))
                msgs = [{"role": "system", "content": hint},
                        {"role": "user", "content": content}]
                r = client().chat.completions.create(model=model or TEXT_MODEL, messages=msgs)
            return _extract_json(r.choices[0].message.content)
        except Exception as e:  # network, schema, or parse problem
            last = e
            print(f"  retry {attempt + 1}/{retries + 1}: {str(e)[:140]}", flush=True)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"model call '{name}' failed: {last}")


_open_clients = {}


def open_providers():
    """Open-weight providers whose API key is set, in priority order. Set OPEN_PROVIDER
    (for example to Fireworks) in .env to force one."""
    chosen = os.environ.get("OPEN_PROVIDER", "").strip().lower()
    found = [pv for pv in OPEN_PROVIDERS if os.environ.get(pv["env"])]
    if chosen:
        found = [pv for pv in found if pv["name"].lower() == chosen] or found
    return found


def _open_client(provider):
    if provider["name"] not in _open_clients:
        _open_clients[provider["name"]] = OpenAI(base_url=provider["url"],
                                                 api_key=os.environ[provider["env"]])
    return _open_clients[provider["name"]]


def ask_open(provider, system, user, name, schema, required, retries=2):
    """Call Muse Spark Open (open weights) through a hosted partner provider.
    The model always reasons and does not enforce JSON schemas, so the prompt demands
    JSON, the reply is validated, and each retry allows more tokens for reasoning."""
    hint = (system + "\nReply with ONLY a JSON object matching this schema, no prose and "
            "no code fences: " + json.dumps(schema))
    last = None
    for attempt in range(retries + 1):
        try:
            r = _open_client(provider).chat.completions.create(
                model=provider["model"],
                messages=[{"role": "system", "content": hint},
                          {"role": "user", "content": user}],
                max_tokens=OPEN_MAX_TOKENS * (attempt + 1),
                temperature=0.2,
                extra_body={"reasoning_effort": OPEN_REASONING},
            )
            data = _extract_json(r.choices[0].message.content)
            missing = [k for k in required if k not in data]
            if missing:
                raise ValueError("reply is missing " + ", ".join(missing))
            return data
        except Exception as e:
            last = e
            print(f"  {provider['name']} retry {attempt + 1}/{retries + 1}: {str(e)[:140]}", flush=True)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{provider['name']} call '{name}' failed: {last}")


def load_signals():
    signals = json.loads(SIGNALS_PATH.read_text())
    if not signals:
        raise RuntimeError("data/signals.json is empty")
    for s in signals:
        s.setdefault("url", s.get("source", "unknown"))
    return signals


def compute_score(opp, by_id, confidence):
    """Return (score, parts, n_signals, n_threads) from evidence, not model opinion."""
    ids = [i for i in dict.fromkeys(opp.get("evidence_ids", [])) if i in by_id]
    n = len(ids)
    threads = len({by_id[i]["url"] for i in ids})
    parts = {
        "evidence": round(W_EVIDENCE * min(n / EVIDENCE_FULL, 1.0)),
        "sources": round(W_SOURCES * min(threads / SOURCES_FULL, 1.0)),
        "verifier": round(W_VERIFIER * max(0, min(confidence, 10)) / 10),
    }
    return sum(parts.values()), parts, n, threads


# ---------- concept visuals ----------

def _sniff_ext(b):
    if b[:4] == b"RIFF":
        return "webp"
    if b[:3] == b"\xff\xd8\xff":
        return "jpg"
    return "png"


def generate_image(prompt):
    r = client().images.generate(model=IMAGE_MODEL, prompt=prompt)
    item = r.data[0]
    if getattr(item, "b64_json", None):
        return base64.b64decode(item.b64_json)
    if getattr(item, "url", None):
        with urllib.request.urlopen(item.url, timeout=60) as resp:
            return resp.read()
    raise RuntimeError("image response had no data")


def concept_prompt(opp, missing=None):
    feats = "; ".join(opp["required_features"])
    extra = f" Make sure these are clearly visible: {', '.join(missing)}." if missing else ""
    return ("Studio flat-lay product photograph of a single garment on a plain light "
            "background. Show only the garment: no people, no body parts, no skin or "
            "skin-colored fabric, no text, no logos. "
            f"Product: {opp['concept']} It must clearly show: {feats}.{extra}")


NEUTRAL_SYSTEM = (
    "You write prompts for a product catalog image generator. Rewrite the product as a "
    "neutral studio flat-lay photo prompt showing the garment alone on a plain light "
    "background. Describe only fabric, shape, closures, and pockets. Do not mention any "
    "body part, medical term, surgery, scars, or people. No text or logos."
)
NEUTRAL_SCHEMA = {
    "type": "object",
    "properties": {"prompt": {"type": "string"}},
    "required": ["prompt"],
    "additionalProperties": False,
}


def safe_prompt(opp):
    """Neutral rewrite used only when the image filter blocks the first prompt."""
    try:
        r = ask(NEUTRAL_SYSTEM,
                f"Product: {opp['title']}. {opp['concept']} "
                f"Features: {'; '.join(opp['required_features'])}",
                "neutral_prompt", NEUTRAL_SCHEMA, retries=1)
        return r["prompt"]
    except Exception:
        return ("Studio flat-lay product photograph of a soft cotton garment with simple "
                "closures on a plain light background. No people, no text, no logos.")


def render(opp, missing=None):
    """Generate an image. If the content filter blocks the prompt, retry once with a
    neutral rewrite."""
    try:
        return generate_image(concept_prompt(opp, missing))
    except Exception as e:
        msg = str(e).lower()
        if "content" not in msg and "filter" not in msg and "policy" not in msg:
            raise
        print("  prompt was filtered, retrying with a neutral rewrite", flush=True)
        return generate_image(safe_prompt(opp))


def visualize(opp, rank):
    """Generate a concept image, check features with Muse Spark vision,
    regenerate once if any feature is missing. Returns dict or None."""
    CONCEPT_DIR.mkdir(parents=True, exist_ok=True)
    missing, best = None, None
    for attempt in range(1, IMAGE_ATTEMPTS + 1):
        try:
            img = render(opp, missing)
        except Exception as e:
            print(f"  image failed: {str(e)[:140]}", flush=True)
            break
        check = None
        try:
            mime = {"webp": "image/webp", "jpg": "image/jpeg", "png": "image/png"}[_sniff_ext(img)]
            data_url = f"data:{mime};base64," + base64.b64encode(img).decode()
            wanted = "\n".join(f"- {f}" for f in opp["required_features"])
            res = ask(CHECK_SYSTEM, f"Check these features:\n{wanted}", "feature_check",
                      CHECK_SCHEMA, retries=1, image_data_url=data_url)
            feats = [{"name": f["name"], "visible": bool(f["visible"])} for f in res["features"]]
            check = {"features": feats, "all_visible": bool(feats) and all(f["visible"] for f in feats),
                     "attempts": attempt}
        except Exception as e:
            print(f"  feature check skipped: {str(e)[:140]}", flush=True)
        best = (img, check)
        if check and check["all_visible"]:
            break
        if check:
            missing = [f["name"] for f in check["features"] if not f["visible"]]
    if not best:
        return None
    img, check = best
    for old in CONCEPT_DIR.glob(f"concept-{rank}.*"):
        old.unlink()
    path = CONCEPT_DIR / f"concept-{rank}.{_sniff_ext(img)}"
    path.write_bytes(img)
    return {"image": f"/static/concepts/{path.name}?v={int(time.time())}", "feature_check": check}


# ---------- main run ----------

def run(progress=None, with_images=True):
    events = []  # structured trace of the run, saved with the results and streamed live

    def step(stage, msg, pct, event=None):
        print(f"[{stage}] {msg}", flush=True)
        if event is not None:
            events.append(event)
        if progress:
            progress(stage, msg, pct, event)

    signals = load_signals()
    by_id = {s["id"]: s for s in signals}
    thread_count = len({s["url"] for s in signals})
    step("propose", f"Muse Spark is reading {len(signals)} signals", 8,
         {"type": "reading", "signals": len(signals), "threads": thread_count})
    listing = "\n".join(f'[{s["id"]}] {s["text"]}' for s in signals)
    gen = ask(GEN_SYSTEM,
              f"Condition: {CONDITION}.\nReturn up to {MAX_OPPS} distinct opportunities.\n\n"
              f"Signals:\n{listing}", "opportunities", GEN_SCHEMA)
    proposed = gen.get("opportunities", [])[:MAX_OPPS]
    step("propose", f"Muse Spark proposed {len(proposed)} ideas", 14,
         {"type": "proposed", "count": len(proposed),
          "ideas": [{"title": o["title"], "features": o.get("required_features", [])} for o in proposed]})

    verified, rejected = [], []
    compared = agreed = 0
    providers = open_providers()
    used = {}   # (model, provider, open_weights) -> number of opportunities it checked
    down = set()  # providers that failed once are skipped for the rest of the run
    for k, opp in enumerate(proposed, 1):
        step("verify", f"Challenging opportunity {k} of {len(proposed)}",
             15 + int(45 * k / max(len(proposed), 1)))
        cited = [by_id[i] for i in dict.fromkeys(opp.get("evidence_ids", [])) if i in by_id]
        if not cited:
            rejected.append({"title": opp["title"], "reason": "No valid evidence was cited.",
                             "evidence_count": 0})
            step("verify", f"Rejected '{opp['title']}': no valid evidence", 15,
                 {"type": "verdict", "title": opp["title"], "kept": False, "model": None,
                  "confidence": 0, "reason": "No valid evidence was cited."})
            continue
        cited_text = "\n".join(f'[{s["id"]}] {s["text"]}' for s in cited)
        user_msg = (
            f"Opportunity title: {opp['title']}\nPain point: {opp['pain_point']}\n"
            f"Unmet need: {opp['unmet_need']}\nConcept: {opp['concept']}\n"
            f"Required features: {'; '.join(opp['required_features'])}\n\n"
            f"Cited signals:\n{cited_text}")
        # first opinion: Muse Spark through Meta Model API
        v_self = ask(VERIFY_SYSTEM, user_msg, "verification", VERIFY_SCHEMA)
        self_check = {"supported": bool(v_self["supported"]),
                      "confidence": int(v_self["confidence"]), "reason": str(v_self["reason"])}
        # independent opinion: Muse Spark Open (open weights) on a partner, else Muse Wonder
        v_second, who = None, None
        for prov in providers:
            if prov["name"] in down:
                continue
            try:
                raw = ask_open(prov, VERIFY_SYSTEM, user_msg, "verification", VERIFY_SCHEMA,
                               ["supported", "confidence", "reason"])
                v_second = {"supported": bool(raw["supported"]),
                            "confidence": int(raw["confidence"]), "reason": str(raw["reason"])}
                who = {"model": prov["model"], "provider": prov["name"], "open_weights": True}
                break
            except Exception as e:
                down.add(prov["name"])
                print(f"  {prov['name']} unavailable, skipping it from now on:", str(e)[:140], flush=True)
        if v_second is None and "wonder" not in down:
            try:
                raw = ask(VERIFY_SYSTEM, user_msg, "verification", VERIFY_SCHEMA, model=SECOND_MODEL)
                v_second = {"supported": bool(raw["supported"]),
                            "confidence": int(raw["confidence"]), "reason": str(raw["reason"])}
                who = {"model": SECOND_MODEL, "provider": "Meta Model API", "open_weights": False}
            except Exception as e:
                down.add("wonder")
                print("  Muse Wonder unavailable, using the Muse Spark verdict:", str(e)[:140], flush=True)
        final = v_second or self_check
        who = who or {"model": TEXT_MODEL, "provider": "Meta Model API", "open_weights": False}
        agrees = None
        if v_second:
            compared += 1
            agrees = v_second["supported"] == self_check["supported"]
            agreed += int(agrees)
            key = (who["model"], who["provider"], who["open_weights"])
            used[key] = used.get(key, 0) + 1
        score, parts, n, threads = compute_score(opp, by_id, final["confidence"])
        record = {"model": who["model"], "provider": who["provider"],
                  "open_weights": who["open_weights"], "confidence": final["confidence"],
                  "reason": final["reason"], "self_check": self_check if v_second else None,
                  "agrees": agrees}
        verdict = {"type": "verdict", "title": opp["title"], "kept": bool(final["supported"]),
                   "model": who["model"], "provider": who["provider"],
                   "open_weights": who["open_weights"], "confidence": final["confidence"],
                   "reason": final["reason"],
                   "self_supported": self_check["supported"] if v_second else None, "agrees": agrees,
                   "evidence": n}
        step("verify", f"{'Kept' if final['supported'] else 'Rejected'}: {opp['title']}",
             15 + int(45 * k / max(len(proposed), 1)), verdict)
        if not final["supported"]:
            rejected.append({"title": opp["title"], "reason": final["reason"], "evidence_count": n,
                             "verifier_model": who["model"], "provider": who["provider"],
                             "open_weights": who["open_weights"],
                             "self_check": self_check if v_second else None, "agrees": agrees})
            continue
        opp.update({
            "score": score, "score_parts": parts,
            "evidence_count": n, "thread_count": threads,
            "verifier": record,
            "evidence": [{"id": s["id"], "text": s["text"], "url": s["url"]} for s in cited],
        })
        verified.append(opp)

    verified.sort(key=lambda o: o["score"], reverse=True)
    for rank, opp in enumerate(verified, 1):
        opp["rank"] = rank
        opp["visual"] = None

    if with_images:
        for opp in verified[:MAX_IMAGES]:
            step("visualize", f"Muse Image is drawing concept {opp['rank']}", 62 + 6 * opp["rank"])
            opp["visual"] = visualize(opp, opp["rank"])
            fc = (opp["visual"] or {}).get("feature_check") or {}
            step("visualize", f"Drew concept {opp['rank']}", 62 + 6 * opp["rank"],
                 {"type": "image", "rank": opp["rank"], "title": opp["title"],
                  "ok": bool(opp["visual"]), "all_visible": bool(fc.get("all_visible")),
                  "attempts": fc.get("attempts", 1)})

    step("done", f"{len(verified)} of {len(proposed)} opportunities survived verification", 100,
         {"type": "done", "proposed": len(proposed), "verified": len(verified),
          "rejected": len(rejected)})
    top = max(used, key=used.get) if used else None
    out = {
        "trace": events,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "condition": CONDITION,
        "models": {"text_and_vision": TEXT_MODEL, "image": IMAGE_MODEL,
                   "verifier": top[0] if top else TEXT_MODEL},
        "cross_check": {"enabled": bool(used), "model": top[0] if top else None,
                        "provider": top[1] if top else None,
                        "open_weights": top[2] if top else False,
                        "compared": compared, "agreed": agreed},
        "signal_count": len(signals),
        "thread_count": thread_count,
        "proposed_count": len(proposed),
        "verified_count": len(verified),
        "rejected_count": len(rejected),
        "score_weights": {"evidence": W_EVIDENCE, "sources": W_SOURCES, "verifier": W_VERIFIER},
        "opportunities": verified,
        "rejected": rejected,
    }
    tmp = RESULTS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    os.replace(tmp, RESULTS_PATH)
    return out


if __name__ == "__main__":
    run(with_images="--no-images" not in sys.argv)
