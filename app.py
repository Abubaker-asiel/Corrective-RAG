import sys

try:  # Streamlit Cloud ships an old sqlite that chromadb rejects
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import html
import os
import uuid

import chromadb
import streamlit as st

from crag import CorrectiveRAG, make_claude
from ingest import SUPPORTED, build_index

FAST_MODEL = os.getenv("CRAG_FAST_MODEL", "claude-haiku-5-5")
ANSWER_MODEL = os.getenv("CRAG_ANSWER_MODEL", "claude-sonnet-5-5")

st.set_page_config(page_title="Corrective RAG", page_icon="🩺", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap');
:root { --ink:#14263A; --mist:#EEF2F5; --harbor:#1B6A8F; --ok:#1E8A5A; --maybe:#C28A14; --no:#B5414F; --line:#D5DEE5; }
html, body, [class*="css"], .stMarkdown, p, li { font-family:'Instrument Sans', sans-serif; color:var(--ink); }
h1, h2, h3, .brand { font-family:'Bricolage Grotesque', sans-serif; letter-spacing:-0.02em; }
.block-container { max-width: 860px; padding-top: 2.2rem; }
.brand { font-size: 2.4rem; font-weight: 700; line-height: 1.05; margin: 0; }
.tagline { color:#4C6176; margin: .35rem 0 1.6rem; max-width: 52ch; }
.q { font-family:'Bricolage Grotesque', sans-serif; font-size:1.25rem; font-weight:500; margin: 1.8rem 0 .6rem; }
[class*="st-key-ans"] { background:#fff; border-left:5px solid var(--ok); padding:1rem 1.2rem .4rem; border-radius:2px 8px 8px 2px; }
[class*="st-key-declined"] { background:#fff; border-left:5px solid var(--no); padding:1rem 1.2rem .4rem; border-radius:2px 8px 8px 2px; }
.src { font-size:.88rem; padding:.45rem 0 .45rem .8rem; border-left:2px solid var(--line); margin:.35rem 0; color:#3B5167; }
.src b { color:var(--harbor); }
.src .snip { color:#5F7488; }
.round { margin:.4rem 0 1.1rem; }
.rq { font-weight:600; margin-bottom:.4rem; }
.rq em { font-style:normal; color:var(--harbor); }
.chunk { border-left:4px solid var(--line); padding:.35rem .8rem; margin:.4rem 0; background:#fff; border-radius:0 6px 6px 0; }
.chunk.correct { border-left-color:var(--ok); }
.chunk.ambiguous { border-left-color:var(--maybe); }
.chunk.incorrect { border-left-color:var(--no); }
.chunk .meta { font-size:.82rem; font-weight:600; }
.chunk.correct .meta { color:var(--ok); } .chunk.ambiguous .meta { color:var(--maybe); } .chunk.incorrect .meta { color:var(--no); }
.chunk .meta span { color:#6A7F92; font-weight:400; }
.chunk .snip { font-size:.88rem; margin-top:.15rem; }
.chunk .why { font-size:.82rem; color:#6A7F92; margin-top:.15rem; }
.outcome { font-size:.9rem; padding:.3rem 0 0 .8rem; color:#3B5167; }
.flow { display:flex; gap:0; margin:0 0 1.8rem; flex-wrap:wrap; }
.flow span { font-size:.88rem; font-weight:600; padding:.45rem .9rem; background:#fff; border:1px solid var(--line); border-right:none; }
.flow span:first-child { border-radius:8px 0 0 8px; }
.flow span:last-child { border-right:1px solid var(--line); border-radius:0 8px 8px 0; }
.flow span.opt { color:var(--maybe); font-weight:500; }
.stamp { font-size:.85rem; font-weight:600; margin-bottom:.5rem; color:var(--ok); }
.stamp.no { color:var(--no); }
.stamp span { font-weight:400; color:#5F7488; }
.bar { display:flex; height:8px; border-radius:4px; overflow:hidden; gap:2px; margin:.2rem 0 .3rem; }
.bar i { display:block; } .bar .c { background:var(--ok); } .bar .a { background:var(--maybe); } .bar .n { background:var(--no); }
.tally { font-size:.82rem; color:#5F7488; margin-bottom:.5rem; }
.empty { background:#fff; border:1px dashed #B7C5D0; border-radius:10px; padding:1.1rem 1.3rem; }
.empty b { font-family:'Bricolage Grotesque', sans-serif; font-size:1.05rem; }
.empty ol { margin:.5rem 0 0 1.1rem; padding:0; } .empty li { margin:.2rem 0; }
section[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LABEL = {"correct": "Verified", "ambiguous": "Unclear", "incorrect": "Rejected"}


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def secret_key() -> str:
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "") or ""
    except Exception:
        return ""


def new_collection():
    if "chroma" not in st.session_state:
        st.session_state.chroma = chromadb.EphemeralClient()
    client = st.session_state.chroma
    if "collection_name" in st.session_state:
        try:
            client.delete_collection(st.session_state.collection_name)
        except Exception:
            pass
    name = f"docs_{uuid.uuid4().hex[:10]}"
    st.session_state.collection_name = name
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def verdict_bar(chunks) -> str:
    n = {g: sum(1 for c in chunks if c.grade == g) for g in ("correct", "ambiguous", "incorrect")}
    seg = "".join(f'<i class="{k}" style="flex:{n[g]}"></i>' for g, k in (("correct", "c"), ("ambiguous", "a"), ("incorrect", "n")) if n[g])
    tally = f'{n["correct"]} verified, {n["ambiguous"]} unclear, {n["incorrect"]} rejected'
    return f'<div class="bar">{seg}</div><div class="tally">{tally}</div>'


def stamp(result) -> str:
    plural = lambda k, w: f"{k} {w}" if k == 1 else f"{k} {w}s"
    rw = f" after {plural(result.rewrites, 'rewrite')}" if result.rewrites else ""
    if result.answerable:
        return f'<div class="stamp">Answered from {plural(len(result.sources), "verified passage")}<span>{rw}</span></div>'
    return f'<div class="stamp no">No verified answer<span> after {plural(len(result.rounds), "search")}</span></div>'


def place(chunk) -> str:
    return f"{chunk.source}, p.{chunk.page}" if chunk.page else chunk.source


def render_result(result, n: int) -> None:
    st.markdown(f'<div class="q">{esc(result.question)}</div>', unsafe_allow_html=True)
    with st.container(key=f"{'ans' if result.answerable else 'declined'}{n}"):
        st.markdown(stamp(result), unsafe_allow_html=True)
        st.markdown(result.answer)
    if result.sources:
        lines = "".join(
            f'<div class="src"><b>[{i}]</b> {esc(place(c))}<div class="snip">{esc(c.text[:170])}...</div></div>'
            for i, c in result.sources
        )
        st.markdown(lines, unsafe_allow_html=True)

    searches = len(result.rounds)
    label = f"Review trail: {searches} {'search' if searches == 1 else 'searches'}, {result.passages_checked} passages checked"
    with st.expander(label):
        for k, rnd in enumerate(result.rounds, 1):
            chunks = "".join(
                f'<div class="chunk {c.grade}"><div class="meta">{LABEL.get(c.grade, "Unclear")} '
                f"<span>{esc(place(c))}</span></div>"
                f'<div class="snip">{esc(c.text[:200])}...</div>'
                f'<div class="why">{esc(c.reason)}</div></div>'
                for c in rnd.chunks
            )
            if rnd.verdict == "rewrite":
                outcome = f"Not enough verified passages. Rewrote the query as: {esc(rnd.note)}"
            elif rnd.verdict == "accepted":
                outcome = "Enough verified passages. Moving on to the answer."
            else:
                outcome = f"Stopped without a verified answer. {esc(rnd.note)}"
            st.markdown(
                f'<div class="round"><div class="rq">Search {k}: <em>{esc(rnd.query)}</em></div>'
                f'{verdict_bar(rnd.chunks)}{chunks}<div class="outcome">{outcome}</div></div>',
                unsafe_allow_html=True,
            )


# sidebar
with st.sidebar:
    st.markdown("### Documents")
    api_key = secret_key() or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Claude API key", type="password")
    files = st.file_uploader("PDF, DOCX, TXT, MD or CSV", type=list(SUPPORTED), accept_multiple_files=True)
    with st.expander("Settings"):
        chunk_size = st.slider("Chunk size (characters)", 400, 1600, 900, 100)
        overlap = st.slider("Chunk overlap", 0, 300, 150, 25)
        top_k = st.slider("Passages per search", 3, 10, 6)
        max_rewrites = st.slider("Max query rewrites", 0, 3, 2)
    if st.button("Build index", type="primary", disabled=not files, use_container_width=True):
        with st.spinner("Reading, cleaning and embedding..."):
            collection = new_collection()
            stats = build_index(collection, [(f.name, f.getvalue()) for f in files], chunk_size, overlap)
        st.session_state.collection = collection if stats.passages else None
        st.session_state.stats = stats
        st.session_state.history = []
    if st.session_state.get("history") and st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    stats = st.session_state.get("stats")
    if stats:
        if stats.passages:
            st.success(f"{stats.passages} passages from {stats.files} file(s)")
        for msg in stats.skipped:
            st.warning(msg)

# main
st.markdown('<h1 class="brand">Corrective RAG</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="tagline">Every retrieved passage is reviewed before it is used. '
    "Weak searches are rewritten and retried, and the answer comes only from verified text.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="flow"><span>Retrieve</span><span>Review each passage</span>'
    '<span class="opt">Rewrite if weak</span><span>Answer with sources</span></div>',
    unsafe_allow_html=True,
)

ready = st.session_state.get("collection") is not None
if not ready:
    st.markdown(
        '<div class="empty"><b>Start with your documents</b><ol>'
        "<li>Upload files in the sidebar</li><li>Build the index</li><li>Ask a question</li></ol></div>",
        unsafe_allow_html=True,
    )
if ready and not api_key:
    st.warning("Add your Claude API key in the sidebar to ask questions.")

for n, res in enumerate(st.session_state.get("history", [])):
    render_result(res, n)

question = st.chat_input("Ask about your documents", disabled=not (ready and api_key))
if question:
    pipe = CorrectiveRAG(
        st.session_state.collection,
        make_claude(api_key),
        FAST_MODEL,
        ANSWER_MODEL,
        top_k=top_k,
        max_rewrites=max_rewrites,
    )
    try:
        with st.status("Working on it...", expanded=True) as status:
            res = pipe.run(question, on_step=st.write)
            status.update(label="Done", state="complete", expanded=False)
        st.session_state.setdefault("history", []).append(res)
        st.rerun()
    except Exception as exc:
        name = type(exc).__name__
        if "Authentication" in name:
            st.error("The API key was rejected. Check it in the sidebar.")
        else:
            st.error(f"Something went wrong: {exc}")
