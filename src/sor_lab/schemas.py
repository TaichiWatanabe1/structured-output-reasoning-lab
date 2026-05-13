"""Pydantic スキーマ。フィールド宣言順がそのまま生成順になる前提なので、
test_schemas.py で `model_json_schema()["properties"]` のキー順を assert する。

`Field(description=...)` は条件間で文言を揃え、順序以外の要素を統制する。
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, create_model

_ANSWER_DESC = "最終解答（整数）"
_REASONING_DESC = "計算過程を順を追って"
_REASONING_STEPS_DESC = "計算過程のステップ列"


class _StrictModel(BaseModel):
    """OpenAI strict JSON Schema が要求する `additionalProperties: false` を強制。"""

    model_config = ConfigDict(extra="forbid")


class AnswerFirst(_StrictModel):
    answer: int = Field(description=_ANSWER_DESC)
    reasoning: str = Field(description=_REASONING_DESC)


class ReasoningFirst(_StrictModel):
    reasoning: str = Field(description=_REASONING_DESC)
    answer: int = Field(description=_ANSWER_DESC)


def make_step_constrained(n_steps: int, answer_first: bool) -> type[BaseModel]:
    """`reasoning_steps: list[str]` を `max_length=n_steps` で制約したスキーマ。

    `answer_first=True` なら `answer` が先、`False` なら `reasoning_steps` が先。
    生成されるクラス名は `StepConstrained_AF{N}` / `StepConstrained_RF{N}`。
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    steps_field = (
        list[str],
        Field(
            description=_REASONING_STEPS_DESC,
            min_length=1,
            max_length=n_steps,
        ),
    )
    answer_field = (int, Field(description=_ANSWER_DESC))

    if answer_first:
        fields: dict[str, tuple[type, object]] = {
            "answer": answer_field,
            "reasoning_steps": steps_field,
        }
        cls_name = f"StepConstrained_AF{n_steps}"
    else:
        fields = {
            "reasoning_steps": steps_field,
            "answer": answer_field,
        }
        cls_name = f"StepConstrained_RF{n_steps}"

    return create_model(  # type: ignore[call-overload, no-any-return]
        cls_name,
        __base__=_StrictModel,
        **fields,
    )


# ---- Registry ----------------------------------------------------------------

# Structured 出力を使う condition_id ベース → Pydantic クラス または ファクトリ
_BASE_SCHEMAS: dict[str, type[BaseModel]] = {
    "so_answer_first": AnswerFirst,
    "so_reasoning_first": ReasoningFirst,
}


def get_schema(condition_id: str) -> type[BaseModel] | None:
    """`condition_id` から Pydantic スキーマクラスを返す。Plain 系は None。

    `so_step_constrained:n_steps=5:answer_first=False` のような派生 ID を解釈。
    """
    if not condition_id.startswith("so_"):
        return None

    parts = condition_id.split(":")
    base = parts[0]

    if base in _BASE_SCHEMAS:
        return _BASE_SCHEMAS[base]

    if base == "so_step_constrained":
        params: dict[str, str] = {}
        for p in parts[1:]:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            params[k] = v
        n_steps = int(params.get("n_steps", "5"))
        answer_first = params.get("answer_first", "False").lower() == "true"
        return make_step_constrained(n_steps=n_steps, answer_first=answer_first)

    raise KeyError(f"unknown structured condition_id base: {base!r}")


def field_order(schema_cls: type[BaseModel]) -> list[str]:
    """Pydantic クラスの宣言フィールド順を返す。"""
    return list(schema_cls.model_fields.keys())


def json_schema_property_order(schema_cls: type[BaseModel]) -> list[str]:
    """`model_json_schema()` の `properties` キー順を返す。

    `dict` は Python 3.7+ で挿入順保持なので、これが Pydantic 宣言順と一致することを
    test_schemas.py で assert する。
    """
    schema = schema_cls.model_json_schema()
    return list(schema.get("properties", {}).keys())


# Annotated 用にエクスポート (今のところ未使用、将来の type narrowing 用)
__all__ = [
    "AnswerFirst",
    "ReasoningFirst",
    "make_step_constrained",
    "get_schema",
    "field_order",
    "json_schema_property_order",
    "Annotated",
]
