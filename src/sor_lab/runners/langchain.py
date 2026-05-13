"""LangChain `AzureChatOpenAI` + `with_structured_output` 経由の Runner。"""

from __future__ import annotations

import json
import time
import warnings
from typing import Any

from sor_lab.config import AzureOpenAISettings
from sor_lab.prompts import get_prompt
from sor_lab.runners.base import RunnerResult
from sor_lab.runners.openai_sdk import _parse_response
from sor_lab.schemas import get_schema


class LangChainRunner:
    """`langchain_openai.AzureChatOpenAI` 経由。"""

    name = "langchain"

    def __init__(self, settings: AzureOpenAISettings) -> None:
        from langchain_openai import AzureChatOpenAI

        self._settings = settings
        # temperature / seed は invoke 時に都度上書きする
        self._llm_cls = AzureChatOpenAI
        self._llm_kwargs = {
            "azure_deployment": settings.deployment_name,
            "api_version": settings.api_version,
            "azure_endpoint": settings.endpoint,
            "api_key": settings.api_key,
        }

    def run(
        self,
        condition_id: str,
        question: str,
        temperature: float,
        seed: int | None = None,
    ) -> RunnerResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = get_prompt(condition_id)
        system, user = prompt.render(question)
        messages: list[Any] = [SystemMessage(system), HumanMessage(user)]

        llm_kwargs = dict(self._llm_kwargs)
        llm_kwargs["temperature"] = temperature
        if seed is not None:
            llm_kwargs["seed"] = seed
        llm = self._llm_cls(**llm_kwargs)

        schema_cls = get_schema(condition_id)
        structured = schema_cls is not None

        start = time.perf_counter()
        if structured:
            assert schema_cls is not None
            # `include_raw=True` で AIMessage と parsed 両方を取得する。
            chain = llm.with_structured_output(
                schema_cls, method="json_schema", strict=True, include_raw=True
            )
            # LangChain 内部で Pydantic が `parsed` フィールドの型不一致 warning を
            # 出すことがある (LangChain 側の実装起因)。データは正しく保存される。
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
                result = chain.invoke(messages)
            latency_ms = int((time.perf_counter() - start) * 1000)

            ai_msg = result.get("raw") if isinstance(result, dict) else None
            parsed = result.get("parsed") if isinstance(result, dict) else None

            raw_content = ""
            if ai_msg is not None:
                content = getattr(ai_msg, "content", "")
                raw_content = content if isinstance(content, str) else json.dumps(content)

            usage = _extract_usage(ai_msg)

            if parsed is not None:
                # Pydantic instance を JSON 文字列化して raw に揃える (キー順保持)
                if hasattr(parsed, "model_dump_json"):
                    raw_content = parsed.model_dump_json()  # type: ignore[attr-defined]

            return _parse_response(
                raw=raw_content,
                structured=True,
                latency_ms=latency_ms,
                prompt_tokens=usage[0],
                completion_tokens=usage[1],
            )

        # Plain 系
        ai_msg = llm.invoke(messages)
        latency_ms = int((time.perf_counter() - start) * 1000)
        content = getattr(ai_msg, "content", "")
        raw_content = content if isinstance(content, str) else json.dumps(content)
        usage = _extract_usage(ai_msg)
        return _parse_response(
            raw=raw_content,
            structured=False,
            latency_ms=latency_ms,
            prompt_tokens=usage[0],
            completion_tokens=usage[1],
        )


def _extract_usage(ai_msg: Any | None) -> tuple[int | None, int | None]:
    """LangChain AIMessage の `usage_metadata` または `response_metadata` から
    prompt/completion tokens を取り出す。"""
    if ai_msg is None:
        return None, None
    usage = getattr(ai_msg, "usage_metadata", None)
    if usage:
        return usage.get("input_tokens"), usage.get("output_tokens")
    meta = getattr(ai_msg, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or meta.get("usage") or {}
    return token_usage.get("prompt_tokens"), token_usage.get("completion_tokens")


__all__ = ["LangChainRunner"]
