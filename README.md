# Threadline Discovery

**An agent that turns what patients say about clothing into evidence-checked product ideas for adaptive fashion teams.**

Built for the Meta Global AI Developer Hackathon. Indie Developer track, "Build for Your World".

- Live site: https://threadline-discovery.onrender.com
- Condition covered: post-mastectomy clothing
- Not medical advice. It supports product decisions for brand and design teams.

## What it does

Adaptive clothing brands often guess what people need. Threadline Discovery reads short notes drawn from public patient forum posts and runs a four-step agent:

1. **Propose.** Muse Spark reads the notes and proposes distinct garment ideas, each citing the notes that support it.
2. **Verify.** Muse Spark gives a first verdict. A second, different Meta model (Muse Wonder) then checks the same claim against only its cited notes and makes the deciding call. Ideas the cited notes do not support are rejected, and the page shows why.
3. **Score.** Code, not a model, scores each surviving idea out of 100: evidence (up to 40, from distinct cited notes), source spread (up to 30, from distinct forum threads), and verifier confidence (up to 30).
4. **Visualize.** Muse Image draws a concept for each idea. Muse Spark's image understanding then checks that the required visible features are in the picture, and the image is redrawn once if they are not.

The site shows the run as a live trace, a map that links every idea to the notes behind it, and a "rejected by the verifier" list.

## Models used

| Model | Where it runs | What it does |
|---|---|---|
| Muse Spark | Meta Model API | Proposes ideas, first verdict, checks features in images |
| Muse Wonder | Meta Model API | Independent second verdict that decides |
| Muse Image | Meta Model API | Draws concept images |

The code can also use Muse Spark Open (open weights) on Baseten or Fireworks as the second verifier when a key is set. That path was not used for the submitted run.

## Data and rights

- The ideas come from forum posts that are other people's writing. **No post text is in this repository or on the site.**
- Each post was restated by Muse Spark as a one-sentence note in new words (`paraphrase.py`). Code rejects any note that shares more than three words in a row with its source, and I read and edited all notes by hand.
- Each note links to the public thread it came from. The original posts stay on my computer.
- Running the checker on `results.json` against the originals also finds no run longer than three words.
- Forum names: Mayo Clinic Connect, Breastcancer.org, Breast Cancer Now, Macmillan.

## What was built during the hackathon

Everything in this repository was written from scratch between Nov 2 and Nov 4, 2026: the agent pipeline (`pipeline.py`), the note-writing and copy-check tool (`paraphrase.py`), the web app (`app.py`), the live trace, the signal map, and the page (`static/index.html`).

The idea builds on an earlier course project of mine, also called Threadline. No code, data, or assets from it are used here. The colors and type follow my Threadline brand style.

## Limits

- Small and narrow: 43 notes from 6 forum threads, one condition.
- Claims are checked against the notes, which are Muse Spark paraphrases, not against the original posts. Paraphrasing can lose or shift meaning.
- The verifier checks whether the notes support an idea. It does not check whether a product already exists on the market.
- Concept images are illustrations, not products.

## Run it yourself

```
git clone https://github.com/rakshakmoorthy/threadline-discovery
cd threadline-discovery
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "MODEL_API_KEY=your-meta-model-api-key" > .env
python pipeline.py        # regenerate the results
uvicorn app:app --reload  # open http://127.0.0.1:8000
```

`data/signals_public.json` (the notes) is included. The first API call of a run uses your Meta Model API credits.
