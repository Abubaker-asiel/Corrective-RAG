"""Corrective RAG: retrieve, grade, rewrite and retry when weak, answer from verified context only."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

CORRECT, AMBIGUOUS, INCORRECT = "correct", "ambiguous", "incorrect"
NOT_FOUND = "NOT_FOUND"

GRADER_SYSTEM = """You are the retrieval grader in a question-answering system.
You receive a question and numbered passages. Grade each passage on its own:
- correct: contains information that directly helps answer the question
- ambiguous: related to the topic but does not contain what is needed
- incorrect: unrelated to the question
Judge only what the passage says. Do not use outside knowledge.
Reply with a JSON array only, no prose, one object per passage:
[{"id": 1, "grade": "correct", "reason": "one short sentence"}]"""

REWRITE_SYSTEM = """You rewrite search queries for a vector database of course and technical documents.
The previous search returned weak results. Write ONE better query: use the terms the documents
would likely use, expand abbreviations, drop filler words, keep the original intent.
Reply with the query only."""

ANSWER_SYSTEM = f"""You answer questions using only the numbered context passages provided.
- Use only facts stated in the passages. No outside knowledge.
- Cite the passages you use inline as [1], [2].
- If the passages do not contain the answer, reply with exactly {NOT_FOUND} and nothing else.
- Be direct and concise."""


def make_claude(api_key: str):
    """Return llm(model, system, user, max_tokens) -> str backed by the Claude API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def llm(model: str, system: str, user: str, max_tokens: int = 800) -> str:
        reply = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in reply.content if b.type == "text")

    return llm


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    distance: float
    grade: str | None = None
    reason: str = ""


@dataclass
class Round:
    query: str
    chunks: list[Chunk]
    verdict: str  # accepted | rewrite | stopped
    note: str = ""  # the rewritten query, or why the search stopped


@dataclass
class Result:
    question: str
    answer: str
    answerable: bool
    sources: list[tuple[int, Chunk]] = field(default_factory=list)
    rounds: list[Round] = field(default_factory=list)

    @property
    def rewrites(self) -> int:
        return sum(1 for r in self.rounds if r.verdict == "rewrite")

    @property
    def passages_checked(self) -> int:
        return len({c.id for r in self.rounds for c in r.chunks})


def parse_grades(raw: str) -> dict[int, tuple[str, str]]:
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        return {}
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    out: dict[int, tuple[str, str]] = {}
    for item in items:
        try:
            grade = str(item["grade"]).lower().strip()
            if grade in (CORRECT, AMBIGUOUS, INCORRECT):
                out[int(item["id"])] = (grade, str(item.get("reason", "")).strip())
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
    return out


class CorrectiveRAG:
    def __init__(
        self,
        collection,
        llm,
        fast_model: str,
        answer_model: str,
        top_k: int = 6,
        max_rewrites: int = 2,
        min_verified: int = 1,
        max_context: int = 5,
    ):
        self.collection = collection
        self.llm = llm
        self.fast_model = fast_model
        self.answer_model = answer_model
        self.top_k = top_k
        self.max_rewrites = max_rewrites
        self.min_verified = min_verified
        self.max_context = max_context

    # 1. retrieve
    def retrieve(self, query: str) -> list[Chunk]:
        total = self.collection.count()
        if total == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(self.top_k, total))
        rows = zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        return [
            Chunk(i, doc, meta.get("source", "unknown"), int(meta.get("page", 0)), float(dist))
            for i, doc, meta, dist in rows
        ]

    # 2. evaluate
    def grade(self, question: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        passages = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, 1))
        raw = self.llm(
            self.fast_model,
            GRADER_SYSTEM,
            f"Question: {question}\n\nPassages:\n{passages}",
            900,
        )
        verdicts = parse_grades(raw)
        for i, chunk in enumerate(chunks, 1):
            chunk.grade, chunk.reason = verdicts.get(
                i, (AMBIGUOUS, "The grader gave no verdict for this passage.")
            )

    # 3. correct
    def rewrite(self, question: str, tried: list[str], pool: dict[str, Chunk]) -> str:
        seen = "\n".join(f"- ({c.grade}) {c.text[:120]}" for c in list(pool.values())[:6])
        user = (
            f"Original question: {question}\n"
            f"Queries already tried: {json.dumps(tried)}\n"
            f"What came back:\n{seen}"
        )
        out = self.llm(self.fast_model, REWRITE_SYSTEM, user, 100).strip()
        lines = out.strip("\"'`").splitlines()
        return lines[0].strip("\"'` ") if lines else ""

    def verified(self, pool: dict[str, Chunk]) -> list[Chunk]:
        good = [c for c in pool.values() if c.grade == CORRECT]
        return sorted(good, key=lambda c: c.distance)[: self.max_context]

    # 4. generate
    def generate(self, question: str, context: list[Chunk]) -> str:
        blocks = "\n\n".join(
            f"[{i}] ({c.source}{f', p.{c.page}' if c.page else ''})\n{c.text}"
            for i, c in enumerate(context, 1)
        )
        return self.llm(
            self.answer_model,
            ANSWER_SYSTEM,
            f"Context:\n{blocks}\n\nQuestion: {question}",
            1000,
        ).strip()

    def run(self, question: str, on_step=None) -> Result:
        say = on_step or (lambda _msg: None)
        pool: dict[str, Chunk] = {}
        rounds: list[Round] = []
        tried: list[str] = []
        query = question

        for attempt in range(self.max_rewrites + 1):
            tried.append(query)
            say(f"Searching for: {query}")
            hits = [pool.setdefault(c.id, c) for c in self.retrieve(query)]
            self.grade(question, [c for c in hits if c.grade is None])
            good = sum(1 for c in hits if c.grade == CORRECT)
            say(f"Reviewed {len(hits)} passages, {good} verified")

            if len(self.verified(pool)) >= self.min_verified:
                rounds.append(Round(query, hits, "accepted"))
                break
            if attempt == self.max_rewrites:
                rounds.append(Round(query, hits, "stopped", "Retry limit reached."))
                break
            say("Not enough verified passages, rewriting the query")
            new_query = self.rewrite(question, tried, pool)
            if not new_query or new_query.lower() in {t.lower() for t in tried}:
                rounds.append(Round(query, hits, "stopped", "No new query to try."))
                break
            rounds.append(Round(query, hits, "rewrite", new_query))
            query = new_query

        context = self.verified(pool)
        result = Result(question, "", False, [], rounds)
        if context:
            say("Writing the answer from verified passages")
            answer = self.generate(question, context)
            if not answer.startswith(NOT_FOUND):
                cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
                picked = [(n, context[n - 1]) for n in cited if 1 <= n <= len(context)]
                result.answer = answer
                result.answerable = True
                result.sources = picked or list(enumerate(context, 1))
                return result

        searches = len(rounds)
        result.answer = (
            "I couldn't find a reliable answer to this in the uploaded documents. "
            f"I checked {result.passages_checked} passages across {searches} "
            f"{'search' if searches == 1 else 'searches'} and none of them addressed the question directly."
        )
        return result
