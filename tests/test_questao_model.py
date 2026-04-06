from __future__ import annotations

import pytest

from src.models import Questao


def test_questao_valida_com_normalizacao() -> None:
    questao = Questao.model_validate(
        {
            "id": "  Q1  ",
            "enunciado": "  Quanto e 2 + 2?  ",
            "alternativas": {
                "a": " 1 ",
                "b": " 2 ",
                "c": " 3 ",
                "d": " 4 ",
                "e": " 5 ",
            },
            "texto_apoio_vinculado": ["  Texto 1  ", "", "   "],
            "tags_assunto": ["  matematica  ", ""],
            "metadados_banca": {
                "banca": "Banca X",
                "prova": "Prova Y",
                "ano": 2025,
                "disciplina": "Matematica",
            },
        }
    )

    assert questao.id == "Q1"
    assert questao.enunciado == "Quanto e 2 + 2?"
    assert list(questao.alternativas.keys()) == ["A", "B", "C", "D", "E"]
    assert questao.alternativas["D"] == "4"
    assert questao.texto_apoio_vinculado == ["Texto 1"]
    assert questao.tags_assunto == ["matematica"]


def test_questao_rejeita_alternativas_incompletas() -> None:
    with pytest.raises(ValueError):
        Questao.model_validate(
            {
                "id": "Q2",
                "enunciado": "Exemplo",
                "alternativas": {
                    "A": "a",
                    "B": "b",
                    "C": "c",
                    "D": "d",
                },
                "texto_apoio_vinculado": [],
                "tags_assunto": [],
                "metadados_banca": {},
            }
        )


def test_questao_rejeita_campos_obrigatorios_vazios() -> None:
    with pytest.raises(ValueError):
        Questao.model_validate(
            {
                "id": "   ",
                "enunciado": "   ",
                "alternativas": {
                    "A": "a",
                    "B": "b",
                    "C": "c",
                    "D": "d",
                    "E": "e",
                },
                "texto_apoio_vinculado": [],
                "tags_assunto": [],
                "metadados_banca": {},
            }
        )
