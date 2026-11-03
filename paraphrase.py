"""Create the public data file: original-wording notes instead of forum text.

The forum comments in data/signals.json are other people's writing, so they stay on
your computer and are never committed. This script asks Muse Spark to restate each
comment as a one-sentence research note in new words, rejects any note that reuses
four or more words in a row from the comment, and writes data/signals_public.json.
Only that file is used by the app and committed to the public repo.

Usage:  python paraphrase.py
"""
import json
import re
from pathlib import Path

import pipeline

DATA = Path(__file__).parent / "data"
PRIVATE = DATA / "signals.json"
PUBLIC = DATA / "signals_public.json"
MAX_SHARED = 3   # a note may share at most 3 words in a row with the original comment

SYSTEM = (
    "You rewrite one patient forum comment as a short research note for a clothing design "
    "team. Write exactly one sentence of at most 22 words, in third person ('Patients ...' "
    "or 'One patient ...'), stating the clothing-related problem or wish in your own plain "
    "words. Do not reuse any run of four or more consecutive words from the comment. Do not "
    "include names, usernames, brand or store names, dates, quotes, or personal medical "
    "details. If the comment contains no clothing, comfort, or product need, set usable to "
    "false and note to an empty string."
)
SCHEMA = {
    "type": "object",
    "properties": {"note": {"type": "string"}, "usable": {"type": "boolean"}},
    "required": ["note", "usable"],
    "additionalProperties": False,
}


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def longest_shared_run(a, b):
    """Length, in words, of the longest run of words that appears in both texts."""
    a, b = words(a), words(b)
    best, prev = 0, [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def rewrite(text):
    """Return (note, reason). note is None when the comment is dropped."""
    for attempt in range(3):
        extra = "" if attempt == 0 else (
            " Your last attempt reused wording from the comment. Use completely different words.")
        try:
            r = pipeline.ask(SYSTEM + extra, "Comment:\n" + text, "note", SCHEMA, retries=1)
        except Exception as e:
            return None, "model error: " + str(e)[:80]
        if not r.get("usable", False):
            return None, "no clothing need"
        note = " ".join(str(r.get("note", "")).split())
        if note and longest_shared_run(note, text) <= MAX_SHARED:
            return note, ""
    return None, "could not avoid copying wording"


def main():
    source = json.loads(PRIVATE.read_text(encoding="utf-8"))
    public, dropped = [], []
    for k, s in enumerate(source, 1):
        note, why = rewrite(s["text"])
        if note:
            public.append({"id": s["id"], "source": s.get("source", "forum"),
                           "url": s.get("url", ""), "text": note})
        else:
            dropped.append((s["id"], why))
        print(f"[{k}/{len(source)}] {'kept' if note else 'dropped: ' + why}", flush=True)
    PUBLIC.write_text(json.dumps(public, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(public)} notes to {PUBLIC.name}. Dropped {len(dropped)}.")
    for i, why in dropped:
        print(f"  dropped id {i}: {why}")
    print("\nFirst three notes, so you can check the style:")
    for n in public[:3]:
        print(f"  [{n['id']}] {n['text']}")
    print("\nNow open data/signals_public.json and read every note before you publish anything.")


if __name__ == "__main__":
    main()
