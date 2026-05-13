"""Pydantic `model_json_schema()` を OpenAI/Azure strict 互換に正規化する。

strict 仕様 (OpenAI):
  - 全 object に `additionalProperties: false`
  - 全 property を `required` に列挙
  - `$defs` をインライン化 (必須ではないが、Azure SDK 側で扱いやすくするため)
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel


def to_strict_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """strict JSON Schema を返す。`properties` キー順は宣言順を保持。"""
    raw = model_cls.model_json_schema()
    inlined = _inline_defs(raw)
    normalized = _normalize(inlined)
    return normalized


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.pop("$defs", None) or schema.pop("definitions", None) or {}
    if not defs:
        return schema

    def replace(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str):
                ref = node["$ref"]
                # `#/$defs/Foo` / `#/definitions/Foo`
                if ref.startswith("#/$defs/"):
                    key = ref.removeprefix("#/$defs/")
                elif ref.startswith("#/definitions/"):
                    key = ref.removeprefix("#/definitions/")
                else:
                    return {k: replace(v) for k, v in node.items()}
                target = defs.get(key)
                if target is None:
                    return node
                merged = copy.deepcopy(target)
                # $ref 以外のキー (description など) は保持
                for k, v in node.items():
                    if k != "$ref":
                        merged[k] = replace(v)
                return replace(merged)
            return {k: replace(v) for k, v in node.items()}
        if isinstance(node, list):
            return [replace(v) for v in node]
        return node

    return replace(schema)  # type: ignore[no-any-return]


def _normalize(schema: dict[str, Any]) -> dict[str, Any]:
    """object に `additionalProperties: false` を強制し `required` を全列挙。"""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            new = {k: walk(v) for k, v in node.items()}
            if new.get("type") == "object" and "properties" in new:
                new["additionalProperties"] = False
                new["required"] = list(new["properties"].keys())
            return new
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)  # type: ignore[no-any-return]


def schema_property_order(schema: dict[str, Any]) -> list[str]:
    """正規化後 schema の `properties` キー順を返す。"""
    return list(schema.get("properties", {}).keys())


__all__ = ["to_strict_schema", "schema_property_order"]
