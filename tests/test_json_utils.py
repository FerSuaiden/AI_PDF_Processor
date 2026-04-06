from __future__ import annotations

import pytest

from src.utils import extract_json_payload


def test_extract_json_payload_com_fence_markdown() -> None:
    payload = extract_json_payload(
        """
```json
{"questoes": [{"id": "Q1"}]}
```
"""
    )
    assert payload == {"questoes": [{"id": "Q1"}]}


def test_extract_json_payload_com_texto_antes_e_depois() -> None:
    payload = extract_json_payload(
        "Resposta final:\n"
        "{\"questoes\": [{\"id\": \"Q2\"}]}\n"
        "Fim."
    )
    assert payload["questoes"][0]["id"] == "Q2"


def test_extract_json_payload_invalido() -> None:
    with pytest.raises(ValueError):
        extract_json_payload("sem json aqui")
