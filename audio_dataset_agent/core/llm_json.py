# LLM JSON 解析工具：从模型输出中提取可解析 JSON。
import json
from typing import Any


def parse_json_object(response: str | dict[str, Any]) -> dict[str, Any]:
    """从直接输出、代码块或包裹文本中解析 LLM 返回的 JSON 对象。"""
    if isinstance(response, dict):
        return response
    if not isinstance(response, str):
        raise TypeError("response must be a JSON string or dict")

    text = _strip_markdown_json_fence(response.strip())
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = json.loads(_extract_first_json_object(text))

    if not isinstance(payload, dict):
        raise TypeError("response JSON must be an object")
    return payload


def _strip_markdown_json_fence(text: str) -> str:
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if len(lines) < 3:
        return text
    if not lines[-1].strip().startswith("```"):
        return text
    return "\n".join(lines[1:-1]).strip()


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in LLM response")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("unterminated JSON object in LLM response")
