from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from typing import Any

from rag_app.config import Settings
from rag_app.errors import ProviderError
from rag_app.providers.ollama import (
    EmbeddingInput,
    GenerationInput,
    OllamaProvider,
    find_context_window,
    find_model,
)

DEFAULT_SAMPLE = "The quick brown fox jumps over the lazy dog."


async def probe(settings: Settings, samples: list[str]) -> dict[str, object]:
    if not settings.generation_model or not settings.embedding_model:
        raise ValueError("Set GENERATION_MODEL and EMBEDDING_MODEL before probing")

    provider = OllamaProvider(settings)
    try:
        models = await provider.list_models()
        generation = find_model(models, settings.generation_model)
        embedding = find_model(models, settings.embedding_model)
        generation_info = await provider.show_model(settings.generation_model)
        embedded = await provider.embed(EmbeddingInput(texts=samples))
        model_info: dict[str, Any] = generation_info.model_info
        generated = await provider.generate(
            GenerationInput(
                prompt=(f"Reply with a short greeting in the language used here: {samples[0]}"),
                max_output_tokens=32,
            )
        )
        return {
            "probeStatus": "ready_for_review",
            "generationModel": {
                "name": generation.name,
                "digest": generation.digest,
                "contextWindowTokens": find_context_window(model_info),
                "sampleOutput": generated.response,
                "promptTokens": generated.prompt_eval_count,
                "outputTokens": generated.eval_count,
                "durationNanoseconds": generated.total_duration,
            },
            "embeddingModel": {
                "name": embedding.name,
                "digest": embedding.digest,
                "dimension": embedded.dimension,
                "sampleCount": len(embedded.vectors),
            },
            "tokenizer": {
                "status": "not-probed",
                "note": (
                    "Ollama exposes model context metadata, not the selected tokenizer's "
                    "local token-count API."
                ),
            },
            "inputBudget": {
                "reportedContextWindowTokens": find_context_window(model_info),
                "exactPromptTokenCount": generated.prompt_eval_count,
                "status": "requires-selected-tokenizer",
            },
            "configuration": {
                "status": "requires-review",
                "note": (
                    "Use these model identifiers and embedding dimension in the reviewed "
                    "shared index config."
                ),
            },
        }
    finally:
        await provider.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe configured local Ollama models")
    parser.add_argument(
        "--sample-text",
        action="append",
        dest="samples",
        help="Representative text to embed; repeat for each required language",
    )
    return parser.parse_args(argv)


async def _async_main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    samples = args.samples or [DEFAULT_SAMPLE]
    try:
        report = await probe(Settings(), samples)
    except (ProviderError, ValueError) as exc:
        print(json.dumps({"probeStatus": "blocked", "reason": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
