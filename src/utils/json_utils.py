from __future__ import annotations

import json
import re
from typing import Any


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()


def _first_json_like_block(text: str) -> str | None:
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def extract_json_payload(raw_text: str) -> Any:
    candidates = [
        raw_text,
        _strip_code_fence(raw_text),
    ]

    block = _first_json_like_block(candidates[-1])
    if block:
        candidates.append(block)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError("Nao foi possivel extrair JSON valido da resposta do LLM")
