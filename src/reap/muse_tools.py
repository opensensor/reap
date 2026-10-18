"""OpenAI function-tool fields and Muse's native ATEM call representation."""

import html
import json
import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Function(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict = Field(default_factory=dict)
    strict: bool | None = None


class Tool(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["function"] = "function"
    function: Function


class CallFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    arguments: dict

    @field_validator("arguments", mode="before")
    @classmethod
    def json_arguments(cls, value):
        return json.loads(value) if isinstance(value, str) else value


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: CallFunction


def active_tools(tools, choice):
    if choice not in (None, "auto", "none"):
        raise ValueError(
            "tool_choice supports auto or none; forced calls are not implemented"
        )
    tools = [t.model_dump(exclude_none=True) for t in tools or []]
    names = [t["function"]["name"] for t in tools]
    if len(set(names)) != len(names):
        raise ValueError("tool names must be unique")
    return tools if choice != "none" else []


_BLOCK = re.compile(r"<atem:function_calls>(.*?)</atem:function_calls>", re.S)
_INVOKE = re.compile(r'<atem:invoke\s+name="([^"]+)">(.*?)</atem:invoke>', re.S)
_PARAMETER = re.compile(
    r'<atem:parameter\s+name="([^"]+)">(.*?)</atem:parameter>', re.S
)


def _argument(raw, schema):
    types = (
        {schema.get("type")}
        if isinstance(schema.get("type"), str)
        else set(schema.get("type", []))
    )
    for branch in schema.get("anyOf", []) + schema.get("oneOf", []):
        kind = branch.get("type")
        if isinstance(kind, str):
            types.add(kind)
    if types == {"string"}:
        return raw
    try:
        value = json.loads(raw)
    except ValueError:
        return raw
    if "string" in types and not (value is None and "null" in types):
        return raw
    return value


def extract_calls(body, tools):
    functions = {t["function"]["name"]: t["function"] for t in tools}
    calls = []
    for block in _BLOCK.finditer(body):
        invokes = list(_INVOKE.finditer(block[1]))
        if not invokes:
            raise ValueError("model produced an empty or malformed tool call")
        for invoke in invokes:
            name = html.unescape(invoke[1])
            if name not in functions:
                raise ValueError(f"model requested an unavailable tool: {name}")
            properties = functions[name].get("parameters", {}).get("properties", {})
            arguments = {}
            for parameter in _PARAMETER.finditer(invoke[2]):
                key = html.unescape(parameter[1])
                if key in arguments:
                    raise ValueError(f"duplicate tool argument: {key}")
                arguments[key] = _argument(parameter[2], properties.get(key, {}))
            calls.append(
                {
                    "id": "call_" + uuid.uuid4().hex,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            arguments, ensure_ascii=False, allow_nan=False
                        ),
                    },
                }
            )
    return calls


def strip_calls(body):
    body = _BLOCK.sub("", body)
    marker = "<atem:function_calls>"
    if marker in body:
        body = body[: body.index(marker)]
    for size in range(min(len(marker) - 1, len(body)), 0, -1):
        if body.endswith(marker[:size]):
            return body[:-size]
    return body
