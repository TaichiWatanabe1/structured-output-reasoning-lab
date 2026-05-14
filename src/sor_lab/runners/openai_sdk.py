"""`openai.AzureOpenAI` (公式 `openai` パッケージ) を使う Runner。"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from sor_lab.config import AzureOpenAISettings, ModelSpec
from sor_lab.evaluation import parse_plain_answer
from sor_lab.prompts import get_prompt
from sor_lab.runners._schema_utils import to_strict_schema
from sor_lab.runners.base import RunnerResult
from sor_lab.schemas import get_schema


class OpenAISdkRunner:
    """OpenAI 公式 SDK の `AzureOpenAI` クライアント経由。

    1 インスタンス = 1 (runner_type, model_spec)。実験ループ側で
    `(runner_name, model_key)` をキーにキャッシュする。
    """

    name = "openai_sdk"

    def __init__(self, settings: AzureOpenAISettings, model_spec: ModelSpec) -> None:
        # 遅延 import (テスト時に SDK 不要パスを通せるようにする)
        from openai import AzureOpenAI

        self._settings = settings
        self._model_spec = model_spec
        self._client = AzureOpenAI(
            api_key=settings.api_key,
            api_version=settings.api_version,
            azure_endpoint=settings.endpoint,
        )

    @property
    def model_spec(self) -> ModelSpec:
        return self._model_spec

    def run(
        self,
        condition_id: str,
        question: str,
        temperature: float,
        seed: int | None = None,
    ) -> RunnerResult:
        prompt = get_prompt(condition_id)
        system, user = prompt.render(question)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        schema_cls = get_schema(condition_id)
        kwargs: dict[str, Any] = {
            "model": self._model_spec.deployment,
            "messages": messages,
        }
        if self._model_spec.reasoning_effort is not None:
            # reasoning モデルは temperature を受け付けないため明示的に渡さない
            kwargs["reasoning_effort"] = self._model_spec.reasoning_effort
        else:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed

        if schema_cls is not None:
            strict_schema = to_strict_schema(schema_cls)
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_cls.__name__,
                    "schema": strict_schema,
                    "strict": True,
                },
            }

        start = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)

        content = resp.choices[0].message.content or ""
        usage = resp.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        model_string = getattr(resp, "model", None)

        return _parse_response(
            raw=content,
            structured=schema_cls is not None,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_string=model_string,
            schema_cls=schema_cls,
        )


def _parse_response(
    raw: str,
    structured: bool,
    latency_ms: int,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    model_string: str | None = None,
    schema_cls: type[BaseModel] | None = None,
) -> RunnerResult:
    """Plain / Structured 共通の parse ロジック。

    Structured では `schema_cls.model_validate` を通って初めて採点対象になる。
    `ValidationError` は `parse_failed=True` として集計から除外される
    (docs/experiment_design.md §8)。
    """
    if not structured:
        answer = parse_plain_answer(raw)
        return RunnerResult(
            raw_response=raw,
            raw_response_keys=None,
            parsed_answer=answer,
            parsed_reasoning=raw,
            parse_failed=answer is None,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_string=model_string,
        )

    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        return RunnerResult(
            raw_response=raw,
            raw_response_keys=None,
            parsed_answer=None,
            parsed_reasoning=None,
            parse_failed=True,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_string=model_string,
        )

    # raw_response_keys は schema 検証 *前* の dict 挿入順を取る (モデル生成順)。
    keys = list(data.keys()) if isinstance(data, dict) else []

    if schema_cls is not None:
        try:
            model = schema_cls.model_validate(data)
        except ValidationError:
            return RunnerResult(
                raw_response=raw,
                raw_response_keys=keys or None,
                parsed_answer=None,
                parsed_reasoning=None,
                parse_failed=True,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model_string=model_string,
            )
        answer_val = getattr(model, "answer", None)
        answer = int(answer_val) if isinstance(answer_val, int) else None
        reasoning: str | list[str] | None
        if hasattr(model, "reasoning_steps"):
            steps = getattr(model, "reasoning_steps", None)
            reasoning = steps if isinstance(steps, list) else None
        elif hasattr(model, "reasoning"):
            r = getattr(model, "reasoning", None)
            reasoning = r if isinstance(r, str) else None
        else:
            reasoning = None
        return RunnerResult(
            raw_response=raw,
            raw_response_keys=keys,
            parsed_answer=answer,
            parsed_reasoning=reasoning,
            parse_failed=answer is None,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_string=model_string,
        )

    # schema_cls が無いケース (旧テスト互換): 緩い抽出のみ。
    answer_raw = data.get("answer") if isinstance(data, dict) else None
    try:
        answer = int(answer_raw) if answer_raw is not None else None
    except (ValueError, TypeError):
        answer = None

    if isinstance(data, dict) and "reasoning_steps" in data:
        steps = data["reasoning_steps"]
        reasoning = steps if isinstance(steps, list) else None
    elif isinstance(data, dict) and "reasoning" in data:
        r = data["reasoning"]
        reasoning = r if isinstance(r, str) else None
    else:
        reasoning = None

    return RunnerResult(
        raw_response=raw,
        raw_response_keys=keys,
        parsed_answer=answer,
        parsed_reasoning=reasoning,
        parse_failed=answer is None,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model_string=model_string,
    )


__all__ = ["OpenAISdkRunner", "_parse_response"]
