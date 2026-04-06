from __future__ import annotations

from pathlib import Path

from src.config import AppConfig
from src.utils.retry import run_with_retry


class IngestionError(RuntimeError):
    """Erro de ingestao/conversao de PDF."""


def pdf_to_markdown(pdf_path: Path, config: AppConfig) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("O arquivo de entrada deve ser um .pdf")

    if config.ingest_provider == "docling":
        markdown = _pdf_to_markdown_docling(pdf_path, config)
    elif config.ingest_provider == "llamaparse":
        markdown = _pdf_to_markdown_llamaparse(pdf_path, config)
    else:
        raise IngestionError(
            f"Provider de ingestao nao suportado: {config.ingest_provider}"
        )

    markdown = markdown.strip()
    if not markdown:
        raise IngestionError("A ingestao retornou markdown vazio")
    return markdown


def _pdf_to_markdown_docling(pdf_path: Path, config: AppConfig) -> str:
    try:
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import (
            DocumentConverter,
            InputFormat,
            PdfFormatOption,
        )
    except ImportError as error:
        raise IngestionError(
            "Docling nao esta instalado. Instale as dependencias do projeto."
        ) from error

    pipeline_options = PdfPipelineOptions(do_ocr=config.docling_do_ocr)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = run_with_retry(
        converter.convert,
        str(pdf_path),
        max_retries=config.max_retries,
        base_delay_seconds=config.base_retry_seconds,
        max_delay_seconds=config.max_retry_seconds,
    )

    if not hasattr(result, "document"):
        raise IngestionError("Conversao Docling sem atributo document")

    try:
        return result.document.export_to_markdown()
    except Exception as error:
        raise IngestionError("Falha ao exportar markdown via Docling") from error


def _pdf_to_markdown_llamaparse(pdf_path: Path, config: AppConfig) -> str:
    try:
        from llama_parse import LlamaParse
    except ImportError as error:
        raise IngestionError(
            "LlamaParse nao esta instalado. Instale as dependencias do projeto."
        ) from error

    parser = LlamaParse(result_type="markdown")
    documents = run_with_retry(
        parser.load_data,
        str(pdf_path),
        max_retries=config.max_retries,
        base_delay_seconds=config.base_retry_seconds,
        max_delay_seconds=config.max_retry_seconds,
    )

    chunks: list[str] = []
    for doc in documents:
        text = getattr(doc, "text", None)
        if text is None:
            text = getattr(doc, "page_content", None)
        if text is None:
            text = str(doc)
        text = text.strip()
        if text:
            chunks.append(text)

    return "\n\n".join(chunks)
