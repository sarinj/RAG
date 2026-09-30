from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_DIRECTORY = Path(__file__).resolve().parents[3] / "evals" / "datasets"
BENCHMARK_PATH = DATASET_DIRECTORY / "pipeline_v2.json"


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    category: str
    question: str
    expected_answer: str | None
    evidence_spans: tuple[tuple[int, int], ...] = ()
    history: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    slug: str
    name: str
    version: str
    document_name: str
    source_sha256: str
    corpus_sha256: str
    corpus_bytes: int
    corpus_text: str
    cases: tuple[EvaluationCase, ...]


def load_datasets() -> dict[str, EvaluationDataset]:
    benchmark_bytes = BENCHMARK_PATH.read_bytes()
    raw = json.loads(benchmark_bytes)
    if not isinstance(raw, dict):
        raise ValueError("The pipeline evaluation benchmark must be an object")

    corpus_name = _required_string(raw, "corpusFile")
    if Path(corpus_name).name != corpus_name:
        raise ValueError("The benchmark corpusFile must name a file in the datasets directory")
    corpus_bytes = (DATASET_DIRECTORY / corpus_name).read_bytes()
    corpus_text = unicodedata.normalize(
        "NFC",
        corpus_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").rstrip(),
    )
    if not corpus_text:
        raise ValueError("The pipeline evaluation corpus cannot be empty")

    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("The pipeline evaluation benchmark must contain cases")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for item in raw_cases:
        if not isinstance(item, dict):
            raise ValueError("Every pipeline evaluation case must be an object")
        case_id = _required_string(item, "id")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case ID: {case_id}")
        seen_ids.add(case_id)

        category = _required_string(item, "category")
        question = _required_string(item, "question")
        expected_answer_value = item.get("expectedAnswer")
        if expected_answer_value is not None and (
            not isinstance(expected_answer_value, str) or not expected_answer_value.strip()
        ):
            raise ValueError(f"Expected answer for {case_id} must be nonempty text or null")
        expected_answer = (
            expected_answer_value.strip() if isinstance(expected_answer_value, str) else None
        )

        raw_history = item.get("history", [])
        if not isinstance(raw_history, list):
            raise ValueError(f"Conversation history for {case_id} must be an array")
        history: list[dict[str, str]] = []
        for index, message in enumerate(raw_history):
            if not isinstance(message, dict):
                raise ValueError(f"Conversation history for {case_id} must contain objects")
            role = "user" if index % 2 == 0 else "assistant"
            content = message.get("content")
            if message.get("role") != role or not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Conversation history for {case_id} must contain ordered user/assistant pairs"
                )
            history.append({"role": role, "content": content.strip()})
        if len(history) % 2:
            raise ValueError(f"Conversation history for {case_id} must contain complete pairs")

        snippets = item.get("evidence", [])
        if not isinstance(snippets, list) or not all(
            isinstance(snippet, str) and snippet.strip() for snippet in snippets
        ):
            raise ValueError(f"Evidence for {case_id} must be an array of nonempty source excerpts")
        if expected_answer is None and snippets:
            raise ValueError(f"Unanswerable case {case_id} cannot have gold evidence")
        if expected_answer is not None and not snippets:
            raise ValueError(f"Answerable case {case_id} must have gold evidence")

        spans = tuple(
            _find_unique_span(corpus_text, snippet, case_id) for snippet in snippets
        )
        cases.append(
            EvaluationCase(
                case_id=case_id,
                category=category,
                question=question,
                expected_answer=expected_answer,
                evidence_spans=spans,
                history=tuple(history),
            )
        )

    corpus_data = corpus_text.encode("utf-8")
    source_hash = hashlib.sha256(benchmark_bytes + b"\0" + corpus_data).hexdigest()
    dataset = EvaluationDataset(
        slug=_required_string(raw, "slug"),
        name=_required_string(raw, "name"),
        version=_required_string(raw, "version"),
        document_name=_required_string(raw, "documentName"),
        source_sha256=source_hash,
        corpus_sha256=hashlib.sha256(corpus_data).hexdigest(),
        corpus_bytes=len(corpus_data),
        corpus_text=corpus_text,
        cases=tuple(cases),
    )
    return {dataset.slug: dataset}


def _required_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"Benchmark field {key} must be nonempty text")
    return item.strip()


def _find_unique_span(text: str, snippet: str, case_id: str) -> tuple[int, int]:
    normalized = unicodedata.normalize("NFC", snippet.strip())
    start = text.find(normalized)
    if start < 0 or text.find(normalized, start + 1) >= 0:
        raise ValueError(f"Evidence for case {case_id} must match the corpus exactly once")
    return start, start + len(normalized)
