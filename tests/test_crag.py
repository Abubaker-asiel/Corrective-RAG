"""Pipeline tests with a fake LLM and a toy embedding, so no API key or model download is needed."""
import hashlib
import json
import re
import sys
from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crag import CorrectiveRAG  # noqa: E402
from ingest import build_index, clean, split_text  # noqa: E402


class BagOfWords(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input):
        out = []
        for text in input:
            v = [0.0] * 128
            for w in re.findall(r"[a-z]+", text.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 128] += 1
            out.append(v)
        return out

    @staticmethod
    def name():
        return "bag-of-words"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return BagOfWords()


DOCS = [
    ("api.txt", b"FastAPI is a Python web framework for building APIs. Endpoints are declared with decorators.\n\n"
                b"Docker packages an app with its dependencies into a container image."),
]


def make_collection(name):
    col = chromadb.EphemeralClient().get_or_create_collection(
        name, embedding_function=BagOfWords(), metadata={"hnsw:space": "cosine"}
    )
    build_index(col, DOCS, size=120, overlap=20)
    return col


def fake_llm(rewrites=None, relevant=("fastapi",), answerable=True):
    calls = {"rewrite": 0, "answer": 0}

    def llm(model, system, user, max_tokens=800):
        if "retrieval grader" in system:
            passages = re.findall(r"\[(\d+)\] (.*?)(?=\n\n\[|\Z)", user, re.S)
            return json.dumps([
                {"id": int(i), "grade": "correct" if any(k in t.lower() for k in relevant) else "incorrect", "reason": "test"}
                for i, t in passages
            ])
        if "rewrite search queries" in system:
            calls["rewrite"] += 1
            return (rewrites or ["fastapi web framework"])[min(calls["rewrite"] - 1, len(rewrites or [1]) - 1)]
        calls["answer"] += 1
        return "FastAPI is a Python web framework [1]." if answerable else "NOT_FOUND"

    llm.calls = calls
    return llm


def test_split_and_clean():
    text = clean("hyphen-\nated word\nsecond line\n\n\n\nnew paragraph")
    assert "hyphenated word second line" in text
    chunks = split_text("word " * 500, size=200, overlap=30)
    assert all(len(c) <= 260 for c in chunks) and len(chunks) > 5


def test_answers_without_rewrite():
    llm = fake_llm()
    res = CorrectiveRAG(make_collection("test_a_col"), llm, "fast", "big").run("What is FastAPI?")
    assert res.answerable and res.sources and len(res.rounds) == 1
    assert res.rounds[0].verdict == "accepted" and llm.calls["rewrite"] == 0


def test_rewrites_when_weak():
    llm = fake_llm(relevant=("fastapi",))
    res = CorrectiveRAG(make_collection("test_b_col"), llm, "fast", "big", top_k=1).run("Docker packages an app with its dependencies into a container image")
    assert llm.calls["rewrite"] >= 1
    assert res.rounds[0].verdict == "rewrite" and res.answerable


def test_declines_when_nothing_verified():
    llm = fake_llm(relevant=("zzzz",), rewrites=["other terms", "more terms"])
    res = CorrectiveRAG(make_collection("test_c_col"), llm, "fast", "big", top_k=2).run("capital of France?")
    assert not res.answerable and llm.calls["answer"] == 0
    assert res.rounds[-1].verdict == "stopped" and "couldn't find" in res.answer


def test_declines_when_generator_says_not_found():
    res = CorrectiveRAG(make_collection("test_d_col"), fake_llm(answerable=False), "fast", "big").run("What is FastAPI?")
    assert not res.answerable
