from __future__ import annotations

import json

import respx
from httpx import Response

from rag_app.config import Settings
from rag_app.generation.grounding import CoverageAssessment
from rag_app.providers.models import GenerationInput
from rag_app.providers.openai import OpenAIProvider


async def test_grounding_request_overrides_nano_minimal_reasoning_and_enforces_schema() -> None:
    schema = CoverageAssessment.model_json_schema()
    original_schema = json.dumps(schema, sort_keys=True)
    with respx.mock(base_url="https://api.openai.com/v1") as router:
        route = router.post("/chat/completions").mock(
            return_value=Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": '{"coverage":"complete","missing_aspects":[]}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "completion_tokens": 83,
                        "completion_tokens_details": {"reasoning_tokens": 64},
                    },
                },
            )
        )
        provider = OpenAIProvider(
            Settings(_env_file=None, llm_provider="openai", generation_model="gpt-5-nano")
        )
        try:
            result = await provider.generate(
                GenerationInput(
                    prompt="Assess evidence and return JSON.",
                    max_output_tokens=256,
                    response_format="json",
                    response_schema=schema,
                    reasoning_effort="low",
                    reasoning_token_budget=512,
                    verbosity="low",
                )
            )
        finally:
            await provider.close()
    body = json.loads(route.calls[0].request.content)
    assert body["reasoning_effort"] == "low"
    assert body["verbosity"] == "low"
    assert body["max_completion_tokens"] == 768
    structured = body["response_format"]
    assert structured["type"] == "json_schema" and structured["json_schema"]["strict"]
    sent = structured["json_schema"]["schema"]
    assert sent["required"] == ["coverage", "missing_aspects"]
    assert sent["additionalProperties"] is False
    assert "default" not in sent["properties"]["missing_aspects"]
    assert json.dumps(schema, sort_keys=True) == original_schema
    assert result.reasoning_tokens == 64


async def test_existing_json_requests_and_non_reasoning_models_remain_compatible() -> None:
    with respx.mock(base_url="https://api.openai.com/v1") as router:
        route = router.post("/chat/completions").mock(
            return_value=Response(
                200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
            ),
        )
        provider = OpenAIProvider(
            Settings(_env_file=None, llm_provider="openai", generation_model="gpt-4o-mini")
        )
        try:
            await provider.generate(
                GenerationInput(
                    prompt="Return JSON.",
                    response_format="json",
                    reasoning_effort="low",
                    temperature=0,
                    verbosity="low",
                )
            )
        finally:
            await provider.close()
    body = json.loads(route.calls[0].request.content)
    assert body["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in body
    assert "verbosity" not in body
    assert body["temperature"] == 0


async def test_per_call_model_override_does_not_mutate_the_default_model() -> None:
    settings = Settings(_env_file=None, llm_provider="openai", generation_model="gpt-5-nano")
    with respx.mock(base_url="https://api.openai.com/v1") as router:
        route = router.post("/chat/completions").mock(
            return_value=Response(
                200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
            )
        )
        provider = OpenAIProvider(settings)
        try:
            await provider.generate(
                GenerationInput(prompt="Quality check", model="gpt-5-mini", reasoning_effort="low")
            )
            await provider.generate(GenerationInput(prompt="Plan next question"))
        finally:
            await provider.close()
    bodies = [json.loads(call.request.content) for call in route.calls]
    assert bodies[0]["model"] == "gpt-5-mini" and bodies[0]["reasoning_effort"] == "low"
    assert bodies[1]["model"] == "gpt-5-nano" and bodies[1]["reasoning_effort"] == "minimal"
    assert settings.generation_model == "gpt-5-nano"
