from __future__ import annotations

import os
from dataclasses import dataclass


SUPPORTED_INGEST_PROVIDERS = {"docling", "llamaparse"}


def _to_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if normalized else None


@dataclass(frozen=True)
class AppConfig:
    ingest_provider: str = "docling"
    extraction_model: str = "gemini/gemini-1.5-flash"
    review_model: str = "gpt-4o-mini"
    review_fallback_model: str | None = None
    docling_do_ocr: bool = False
    max_retries: int = 5
    base_retry_seconds: float = 1.5
    max_retry_seconds: float = 20.0
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "AppConfig":
        provider = os.getenv("INGEST_PROVIDER", "docling").strip().lower()
        if provider not in SUPPORTED_INGEST_PROVIDERS:
            raise ValueError(
                f"INGEST_PROVIDER invalido: {provider}. Use docling ou llamaparse."
            )

        return cls(
            ingest_provider=provider,
            extraction_model=os.getenv(
                "EXTRACTION_MODEL", "gemini/gemini-1.5-flash"
            ).strip(),
            review_model=os.getenv("REVIEW_MODEL", "gpt-4o-mini").strip(),
            review_fallback_model=_to_optional_str(
                os.getenv("REVIEW_FALLBACK_MODEL")
            ),
            docling_do_ocr=_to_bool(os.getenv("DOCLING_DO_OCR"), False),
            max_retries=_to_int(os.getenv("MAX_RETRIES", "5"), 5),
            base_retry_seconds=_to_float(
                os.getenv("BASE_RETRY_SECONDS", "1.5"), 1.5
            ),
            max_retry_seconds=_to_float(
                os.getenv("MAX_RETRY_SECONDS", "20"), 20.0
            ),
            temperature=_to_float(os.getenv("LLM_TEMPERATURE", "0.0"), 0.0),
        )
