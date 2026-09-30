from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

DATASET_DIRECTORY = Path(__file__).resolve().parents[3] / "evals" / "datasets"


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    category: str
    question: str
    expected_answer: str | None
    evidence_spans: tuple[tuple[int, int], ...] = ()


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
    datasets = {
        "astrarail": _load_astrarail(),
        "helios_city": _load_helios_city(),
    }
    label_path = DATASET_DIRECTORY / "retrieval_labels.json"
    label_bytes = label_path.read_bytes()
    label_data = json.loads(label_bytes)
    if not isinstance(label_data, dict) or not isinstance(label_data.get("datasets"), dict):
        raise ValueError("Retrieval labels must contain a datasets object")
    label_version = label_data.get("version")
    if not isinstance(label_version, str) or not label_version.strip():
        raise ValueError("Retrieval labels must have a nonempty version")

    dataset_labels = label_data["datasets"]
    if set(dataset_labels) != set(datasets):
        raise ValueError("Retrieval labels must match the bundled evaluation datasets")
    return {
        slug: _attach_retrieval_labels(dataset, dataset_labels[slug], label_version)
        for slug, dataset in datasets.items()
    }


def _attach_retrieval_labels(
    dataset: EvaluationDataset,
    raw_labels: object,
    label_version: str,
) -> EvaluationDataset:
    if not isinstance(raw_labels, dict):
        raise ValueError(f"Retrieval labels for {dataset.slug} must be an object")
    cases_by_id = {case.case_id: case for case in dataset.cases}
    unknown_ids = set(raw_labels) - set(cases_by_id)
    if unknown_ids:
        raise ValueError(f"Retrieval labels for {dataset.slug} contain unknown cases")

    text = unicodedata.normalize(
        "NFC",
        dataset.corpus_text.replace("\r\n", "\n").replace("\r", "\n"),
    )
    labeled_cases: list[EvaluationCase] = []
    for case in dataset.cases:
        raw_evidence = raw_labels.get(case.case_id, [])
        if not isinstance(raw_evidence, list) or not all(
            isinstance(item, str) and item.strip() for item in raw_evidence
        ):
            raise ValueError(
                f"Retrieval evidence for {dataset.slug} case {case.case_id} must be nonempty text"
            )
        if raw_evidence and case.expected_answer is None:
            raise ValueError(
                f"Unanswerable case {dataset.slug} {case.case_id} cannot have gold evidence"
            )

        spans = tuple(
            _find_unique_span(text, snippet, dataset.slug, case.case_id)
            for snippet in raw_evidence
        )
        labeled_cases.append(replace(case, evidence_spans=spans))

    labels_hash_input = json.dumps(
        {"version": label_version, "labels": raw_labels},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    combined_source_hash = _sha256(
        bytes.fromhex(dataset.source_sha256) + b"\0" + labels_hash_input
    )
    return replace(
        dataset,
        source_sha256=combined_source_hash,
        cases=tuple(labeled_cases),
    )


def _find_unique_span(text: str, snippet: str, dataset_slug: str, case_id: str) -> tuple[int, int]:
    start = text.find(snippet)
    if start < 0 or text.find(snippet, start + 1) >= 0:
        raise ValueError(
            f"Retrieval evidence for {dataset_slug} case {case_id} must match exactly once"
        )
    return start, start + len(snippet)


def _load_astrarail() -> EvaluationDataset:
    source_path = DATASET_DIRECTORY / "astrarail_rag_test (1).txt"
    answer_path = DATASET_DIRECTORY / "astrarail_answers.json"
    source = source_path.read_text(encoding="utf-8")
    answer_data = json.loads(answer_path.read_text(encoding="utf-8"))

    corpus, question_section = _split_once(source, "8. RAG Test Questions")
    answerable_section, unanswerable_section = _split_once(
        question_section,
        "Questions that should NOT be answerable from this document:",
    )
    answerable = _numbered_questions(answerable_section)
    unanswerable = _numbered_questions(unanswerable_section)
    gold_answers = answer_data.get("answers")
    if not isinstance(gold_answers, dict):
        raise ValueError("AstraRail answer key must contain an answers object")
    if {str(index) for index, _ in answerable} != set(gold_answers):
        raise ValueError("AstraRail answer key does not match the answerable questions")

    corpus_text = corpus.rstrip()
    cases = [
        EvaluationCase(str(index), "direct", question, gold_answers[str(index)])
        for index, question in answerable
    ]
    cases.extend(
        EvaluationCase(f"u{index}", "unanswerable", question, None)
        for index, question in unanswerable
    )
    return EvaluationDataset(
        slug="astrarail",
        name="AstraRail Operations Knowledge Base",
        version=str(answer_data.get("version", "1")),
        document_name="astrarail_rag_test.txt",
        source_sha256=_sha256(source_path.read_bytes() + answer_path.read_bytes()),
        corpus_sha256=_sha256(corpus_text.encode("utf-8")),
        corpus_bytes=len(corpus_text.encode("utf-8")),
        corpus_text=corpus_text,
        cases=tuple(cases),
    )


def _load_helios_city() -> EvaluationDataset:
    source_path = DATASET_DIRECTORY / "helios_city_very_long_rag_test (1).txt"
    source = source_path.read_text(encoding="utf-8")
    corpus, question_and_key = _split_once(source, "151. RAG TEST QUESTIONS")
    question_section, answer_key_section = _split_once(
        question_and_key,
        "152. EXPECTED ANSWER KEY",
    )
    question_cases = _parse_helios_questions(question_section)
    expected_answers = _parse_helios_answers(answer_key_section)

    cases: list[EvaluationCase] = []
    for case_id, category, question in question_cases:
        expected_answer = expected_answers.get(case_id)
        if category != "unanswerable" and expected_answer is None:
            raise ValueError(f"Helios answer key is missing case {case_id}")
        cases.append(EvaluationCase(case_id, category, question, expected_answer))
    answerable_ids = {
        case.case_id for case in cases if case.expected_answer is not None
    }
    if set(expected_answers) != answerable_ids:
        raise ValueError("Helios answer key contains unmatched or missing answerable cases")

    corpus_text = corpus.rstrip()
    version_match = re.search(r"(?m)^Version:\s*(\S+)\s*$", source)
    return EvaluationDataset(
        slug="helios_city",
        name="Helios City Digital Operations Knowledge Base",
        version=version_match.group(1) if version_match else "unversioned",
        document_name="helios_city_very_long_rag_test.txt",
        source_sha256=_sha256(source_path.read_bytes()),
        corpus_sha256=_sha256(corpus_text.encode("utf-8")),
        corpus_bytes=len(corpus_text.encode("utf-8")),
        corpus_text=corpus_text,
        cases=tuple(cases),
    )


def _parse_helios_questions(section: str) -> list[tuple[str, str, str]]:
    categories = {
        "Direct factual questions:": "direct",
        "Semantic and paraphrase questions:": "semantic",
        "Multi-hop questions:": "multi_hop",
        "Questions that should NOT be answered from this document:": "unanswerable",
    }
    category: str | None = None
    result: list[tuple[str, str, str]] = []
    for line in section.splitlines():
        stripped = line.strip()
        if stripped in categories:
            category = categories[stripped]
            continue
        match = re.match(r"^(\d+)\.\s+(.+?)\s*$", stripped)
        if match:
            if category is None:
                raise ValueError("Helios question appears before its category heading")
            result.append((match.group(1), category, match.group(2)))
    if len(result) != 100:
        raise ValueError(f"Expected 100 Helios questions, found {len(result)}")
    return result


def _parse_helios_answers(section: str) -> dict[str, str]:
    answers_section, notes_marker, _notes = section.partition("153. RAG EVALUATION NOTES")
    if not notes_marker:
        raise ValueError("Helios dataset is missing its evaluation-notes boundary")
    answers = {
        match.group(1): match.group(2).strip()
        for match in re.finditer(r"(?m)^(\d+)\.\s+(.+?)\s*$", answers_section)
    }
    if set(answers) != {str(index) for index in range(1, 81)}:
        raise ValueError("Helios answer key must contain answers 1 through 80")
    if not re.search(r"(?m)^81-100\.\s+", answers_section):
        raise ValueError("Helios answer key must mark questions 81 through 100 unanswerable")
    return answers


def _numbered_questions(section: str) -> list[tuple[int, str]]:
    return [
        (int(match.group(1)), match.group(2).strip())
        for match in re.finditer(r"(?m)^(\d+)\.\s+(.+?)\s*$", section)
    ]


def _split_once(text: str, marker: str) -> tuple[str, str]:
    before, found, after = text.partition(marker)
    if not found:
        raise ValueError(f"Dataset is missing expected section: {marker}")
    return before, after


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
