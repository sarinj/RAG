from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from rag_app.config import Settings
from rag_app.errors import ProviderError
from rag_app.providers.factory import create_provider
from rag_app.providers.models import EmbeddingInput, GenerationInput
from rag_app.providers.ollama import (
    OllamaProvider,
    find_context_window,
    find_model,
)

DEFAULT_SAMPLE = "The quick brown fox jumps over the lazy dog."


async def probe(settings: Settings, samples: list[str]) -> dict[str, object]:
    if not settings.generation_model or not settings.embedding_model:
        raise ValueError("Set GENERATION_MODEL and EMBEDDING_MODEL before probing")

    provider = create_provider(settings)
    try:
        embedded = await provider.embed(EmbeddingInput(texts=samples))
        generated = await provider.generate(
            GenerationInput(
                prompt=(f"Reply with a short greeting in the language used here: {samples[0]}"),
                max_output_tokens=32,
            )
        )
        generation_digest: str | None = None
        embedding_digest = settings.embedding_model_digest
        context_window: int | None = None
        generation_is_ollama = settings.llm_provider == "ollama"
        embedding_is_ollama = settings.active_embedding_provider == "ollama"
        if generation_is_ollama or embedding_is_ollama:
            metadata_provider = OllamaProvider(
                settings,
                (
                    settings.generation_connection
                    if generation_is_ollama
                    else settings.embedding_connection
                ),
            )
            try:
                models = await metadata_provider.list_models()
                if generation_is_ollama:
                    generation_model = find_model(models, settings.generation_model)
                    generation_info = await metadata_provider.show_model(settings.generation_model)
                    context_window = find_context_window(generation_info.model_info)
                    generation_digest = generation_model.digest
                if embedding_is_ollama:
                    embedding_model = find_model(models, settings.embedding_model)
                    embedding_digest = embedding_model.digest
            finally:
                await metadata_provider.close()
        return {
            "probeStatus": "ready_for_review",
            "provider": settings.llm_provider,
            "embeddingProvider": settings.active_embedding_provider,
            "generationModel": {
                "name": settings.generation_model,
                "digest": generation_digest,
                "contextWindowTokens": context_window,
                "sampleOutput": generated.response,
                "promptTokens": generated.prompt_eval_count,
                "outputTokens": generated.eval_count,
                "durationNanoseconds": generated.total_duration,
            },
            "embeddingModel": {
                "name": settings.embedding_model,
                "digest": embedding_digest,
                "dimension": embedded.dimension,
                "sampleCount": len(embedded.vectors),
            },
            "tokenizer": {
                "status": "not-probed",
                "note": (
                    "The provider does not expose the selected tokenizer's local token-count API."
                ),
            },
            "inputBudget": {
                "reportedContextWindowTokens": context_window,
                "exactPromptTokenCount": generated.prompt_eval_count,
                "status": "requires-selected-tokenizer",
            },
            "configuration": {
                "status": "requires-review",
                "note": (
                    "Use these model identifiers and embedding dimension in the reviewed "
                    "shared index config. For hosted embeddings, set EMBEDDING_MODEL_DIGEST "
                    "to the operator-pinned embedding model fingerprint."
                ),
            },
        }
    finally:
        await provider.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe configured generation and embedding models")
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
