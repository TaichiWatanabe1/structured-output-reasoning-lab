"""スキーマの宣言順が JSON Schema properties 順と一致することを保証する。
これが崩れたら実験の前提 (順序効果) が無意味になるので最重要テスト。"""

from __future__ import annotations

import pytest

from sor_lab.schemas import (
    AnswerFirst,
    ReasoningFirst,
    field_order,
    get_schema,
    json_schema_property_order,
    make_step_constrained,
)


def test_answer_first_field_order() -> None:
    assert field_order(AnswerFirst) == ["answer", "reasoning"]


def test_reasoning_first_field_order() -> None:
    assert field_order(ReasoningFirst) == ["reasoning", "answer"]


def test_answer_first_json_schema_property_order_matches_declaration() -> None:
    assert json_schema_property_order(AnswerFirst) == ["answer", "reasoning"]


def test_reasoning_first_json_schema_property_order_matches_declaration() -> None:
    assert json_schema_property_order(ReasoningFirst) == ["reasoning", "answer"]


def test_strict_models_forbid_extra_keys() -> None:
    schema = AnswerFirst.model_json_schema()
    assert schema.get("additionalProperties") is False


@pytest.mark.parametrize("n_steps", [1, 3, 5, 10])
@pytest.mark.parametrize("answer_first", [True, False])
def test_step_constrained_field_order(n_steps: int, answer_first: bool) -> None:
    cls = make_step_constrained(n_steps=n_steps, answer_first=answer_first)
    expected = (
        ["answer", "reasoning_steps"] if answer_first else ["reasoning_steps", "answer"]
    )
    assert field_order(cls) == expected
    assert json_schema_property_order(cls) == expected


@pytest.mark.parametrize("n_steps", [1, 3, 5, 10])
def test_step_constrained_max_length_enforced(n_steps: int) -> None:
    cls = make_step_constrained(n_steps=n_steps, answer_first=False)
    schema = cls.model_json_schema()
    steps_schema = schema["properties"]["reasoning_steps"]
    assert steps_schema["maxItems"] == n_steps
    assert steps_schema["minItems"] == 1


def test_make_step_constrained_rejects_zero() -> None:
    with pytest.raises(ValueError):
        make_step_constrained(n_steps=0, answer_first=False)


def test_get_schema_plain_returns_none() -> None:
    assert get_schema("plain_direct") is None
    assert get_schema("plain_cot") is None


def test_get_schema_so_answer_first() -> None:
    assert get_schema("so_answer_first") is AnswerFirst


def test_get_schema_so_reasoning_first() -> None:
    assert get_schema("so_reasoning_first") is ReasoningFirst


def test_get_schema_step_constrained_parses_params() -> None:
    cls = get_schema("so_step_constrained:n_steps=5:answer_first=False")
    assert cls is not None
    assert field_order(cls) == ["reasoning_steps", "answer"]

    cls_af = get_schema("so_step_constrained:n_steps=3:answer_first=True")
    assert cls_af is not None
    assert field_order(cls_af) == ["answer", "reasoning_steps"]


def test_descriptions_match_between_schemas() -> None:
    """順序以外の要素 (description) を統制している確認。"""
    af = AnswerFirst.model_json_schema()["properties"]
    rf = ReasoningFirst.model_json_schema()["properties"]
    assert af["answer"]["description"] == rf["answer"]["description"]
    assert af["reasoning"]["description"] == rf["reasoning"]["description"]
