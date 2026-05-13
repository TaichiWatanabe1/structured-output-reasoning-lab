"""Microsoft 純正 `azure-ai-inference` 経由の Runner。

Azure OpenAI デプロイメントエンドポイントを `ChatCompletionsClient` で叩く。
"""

from __future__ import annotations

import time
from typing import Any

from sor_lab.config import AzureOpenAISettings
from sor_lab.prompts import get_prompt
from sor_lab.runners._schema_utils import to_strict_schema
from sor_lab.runners.base import RunnerResult
from sor_lab.runners.openai_sdk import _parse_response
from sor_lab.schemas import get_schema


class AzureSdkRunner:
    """`azure.ai.inference.ChatCompletionsClient` 経由。"""

    name = "azure_sdk"

    def __init__(self, settings: AzureOpenAISettings) -> None:
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential

        self._settings = settings
        # Azure OpenAI deployment は `<endpoint>/openai/deployments/<deployment>` を
        # base にして API version をクエリで渡す形式。`azure-ai-inference` は
        # `endpoint` をそのまま受け取り、内部で URL を組み立てる。
        deployment_endpoint = (
            settings.endpoint.rstrip("/")
            + f"/openai/deployments/{settings.deployment_name}"
        )
        self._client = ChatCompletionsClient(
            endpoint=deployment_endpoint,
            credential=AzureKeyCredential(settings.api_key),
            api_version=settings.api_version,
        )

    def run(
        self,
        condition_id: str,
        question: str,
        temperature: float,
        seed: int | None = None,
    ) -> RunnerResult:
        from azure.ai.inference.models import (
            JsonSchemaFormat,
            SystemMessage,
            UserMessage,
        )

        prompt = get_prompt(condition_id)
        system, user = prompt.render(question)
        messages: list[Any] = [SystemMessage(system), UserMessage(user)]

        schema_cls = get_schema(condition_id)
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "model": self._settings.deployment_name,
        }
        if seed is not None:
            kwargs["seed"] = seed

        if schema_cls is not None:
            strict_schema = to_strict_schema(schema_cls)
            kwargs["response_format"] = JsonSchemaFormat(
                name=schema_cls.__name__,
                schema=strict_schema,
                strict=True,
            )

        start = time.perf_counter()
        resp = self._client.complete(**kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)

        choice = resp.choices[0]
        content = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        return _parse_response(
            raw=content,
            structured=schema_cls is not None,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


__all__ = ["AzureSdkRunner"]
