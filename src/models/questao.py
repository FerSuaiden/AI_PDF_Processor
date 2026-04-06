from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class MetadadosBanca(BaseModel):
    banca: str | None = Field(default=None, description="Nome da banca")
    prova: str | None = Field(default=None, description="Nome da prova")
    ano: int | None = Field(default=None, description="Ano da prova")
    disciplina: str | None = Field(default=None, description="Disciplina principal")


class Questao(BaseModel):
    id: str = Field(description="Identificador unico da questao")
    enunciado: str = Field(description="Texto completo do enunciado")
    alternativas: dict[str, str] = Field(
        description="Alternativas no formato {'A': '...', 'B': '...', ..., 'E': '...'}"
    )
    texto_apoio_vinculado: list[str] = Field(
        default_factory=list,
        description="Textos de apoio vinculados a questao",
    )
    tags_assunto: list[str] = Field(
        default_factory=list,
        description="Tags de assunto inferidas pelos agentes",
    )
    metadados_banca: MetadadosBanca = Field(
        default_factory=MetadadosBanca,
        description="Metadados da banca/prova",
    )

    @field_validator("id", "enunciado")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Campo obrigatorio nao pode estar vazio")
        return value

    @field_validator("alternativas")
    @classmethod
    def validate_alternatives_keys(cls, value: dict[str, str]) -> dict[str, str]:
        expected = {"A", "B", "C", "D", "E"}
        received = {key.strip().upper() for key in value.keys()}

        if received != expected:
            raise ValueError("Alternativas devem conter exatamente as chaves A, B, C, D e E")

        normalized = {
            key.strip().upper(): text.strip()
            for key, text in value.items()
        }

        for key, text in normalized.items():
            if not text:
                raise ValueError(f"Alternativa {key} nao pode estar vazia")

        return normalized

    @field_validator("texto_apoio_vinculado", "tags_assunto")
    @classmethod
    def strip_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]