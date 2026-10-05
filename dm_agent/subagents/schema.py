"""Explicit JSON Schema subset; unsupported keywords fail closed."""

from __future__ import annotations

from typing import Any

DEFAULT_REPORT = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "findings", "uncertainties"],
    "additionalProperties": False,
}
TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "null": type(None),
}
KEYWORDS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "description",
}


def check_schema(schema: Any, depth: int = 0) -> None:
    if depth > 10 or not isinstance(schema, dict) or set(schema) - KEYWORDS:
        raise ValueError("unsupported output schema; see docs/subagents.md")
    if not isinstance(schema.get("type"), str) or schema["type"] not in TYPES:
        raise ValueError("schema must declare a supported type")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError("enum must be a non-empty array")
    if schema["type"] == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ValueError("invalid object schema")
        if not all(isinstance(k, str) and k in properties for k in required):
            raise ValueError("required keys must be declared in properties")
        if not isinstance(schema.get("additionalProperties", True), bool):
            raise ValueError("additionalProperties must be boolean")
        for child in properties.values():
            check_schema(child, depth + 1)
    elif set(schema) & {"properties", "required", "additionalProperties"}:
        raise ValueError("object keywords require object type")
    if schema["type"] == "array":
        check_schema(schema.get("items"), depth + 1)
    elif "items" in schema:
        raise ValueError("items requires array type")


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    kind = schema["type"]
    if not isinstance(value, TYPES[kind]) or (
        kind in {"integer", "number"} and isinstance(value, bool)
    ):
        raise ValueError(f"{path}: expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: not in enum")
    if kind == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        if set(schema.get("required", [])) - set(value):
            raise ValueError(f"{path}: missing required fields")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError(f"{path}: unexpected fields")
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key], f"{path}.{key}")
    elif kind == "array" and isinstance(value, list):
        for index, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{index}]")
