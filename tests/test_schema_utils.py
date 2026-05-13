"""strict JSON Schema 変換のテスト。`properties` キー順保持が最重要。"""

from __future__ import annotations

from sor_lab.runners._schema_utils import schema_property_order, to_strict_schema
from sor_lab.schemas import AnswerFirst, ReasoningFirst, make_step_constrained


def test_to_strict_schema_answer_first_property_order() -> None:
    schema = to_strict_schema(AnswerFirst)
    assert schema_property_order(schema) == ["answer", "reasoning"]


def test_to_strict_schema_reasoning_first_property_order() -> None:
    schema = to_strict_schema(ReasoningFirst)
    assert schema_property_order(schema) == ["reasoning", "answer"]


def test_to_strict_schema_step_constrained_property_order_rf() -> None:
    cls = make_step_constrained(n_steps=5, answer_first=False)
    schema = to_strict_schema(cls)
    assert schema_property_order(schema) == ["reasoning_steps", "answer"]


def test_to_strict_schema_step_constrained_property_order_af() -> None:
    cls = make_step_constrained(n_steps=3, answer_first=True)
    schema = to_strict_schema(cls)
    assert schema_property_order(schema) == ["answer", "reasoning_steps"]


def test_strict_schema_forbids_additional_properties() -> None:
    schema = to_strict_schema(AnswerFirst)
    assert schema["additionalProperties"] is False


def test_strict_schema_requires_all_properties() -> None:
    schema = to_strict_schema(AnswerFirst)
    assert set(schema["required"]) == {"answer", "reasoning"}
