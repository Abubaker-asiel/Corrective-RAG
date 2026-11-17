# Corrective RAG

A document Q&A system that checks its own retrieval. Each retrieved passage is graded before use, weak searches are rewritten and retried, and answers are generated only from verified passages, with sources.

## How it works

1. **Ingest** (`ingest.py`): upload PDF, DOCX, TXT, MD or CSV, extract text, clean it, split into overlapping chunks, embed into a Chroma collection.
2. **Retrieve**: top-k passages for the question.
3. **Evaluate**: Claude Haiku grades every passage as verified, unclear or rejected, with a one-line reason.
4. **Correct**: if no passage is verified, the query is rewritten and retrieval runs again (up to 2 rewrites). Passages already graded are not graded twice.
5. **Generate**: Claude Sonnet answers from verified passages only and cites them as [1], [2].
6. **Unanswered questions**: if nothing verifies, or the generator finds the answer isn't in the context, the app says so instead of guessing.

The UI shows a review trail for each answer: every search, every passage and its verdict.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

Tests (no API key needed): `pytest tests`

## Deploy on Streamlit Cloud

1. Push this folder to GitHub.
2. Create the app from `app.py`.
3. In Settings, Secrets, add `ANTHROPIC_API_KEY = "sk-ant-..."`.

## Config

Models can be changed with `CRAG_FAST_MODEL` and `CRAG_ANSWER_MODEL`. Chunk size, overlap, top-k and max rewrites are in the sidebar Settings.

The index lives in memory and is rebuilt per session, so re-upload after a restart.
