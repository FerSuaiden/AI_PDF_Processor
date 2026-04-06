#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


def _clean_text(text: str) -> str:
    text = text.replace("\u00ac", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"Concurso\s+Vestibular\s+FUVEST\s+\d{4}\s*–\s*Prova\s+V\d", "", text)
    text = text.replace("#####", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []

    out: list[int] = []
    for value in values:
        if isinstance(value, int):
            out.append(value)
            continue
        if isinstance(value, str) and value.strip().isdigit():
            out.append(int(value.strip()))
    return out


def _question_ids_from_stage2(stage2_payload: dict[str, Any]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()

    pages = stage2_payload.get("pages", [])
    if not isinstance(pages, list):
        return ordered

    for page in pages:
        if not isinstance(page, dict):
            continue
        for segment in page.get("segments", []):
            if not isinstance(segment, dict):
                continue
            for qid in _parse_int_list(segment.get("question_ids", [])):
                if qid not in seen:
                    seen.add(qid)
                    ordered.append(qid)

    return ordered


def _question_pages_from_stage2(stage2_payload: dict[str, Any]) -> dict[int, list[int]]:
    mapping: dict[int, set[int]] = {}

    pages = stage2_payload.get("pages", [])
    if not isinstance(pages, list):
        return {}

    for page in pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number")
        if not isinstance(page_number, int):
            continue
        for segment in page.get("segments", []):
            if not isinstance(segment, dict):
                continue
            for qid in _parse_int_list(segment.get("question_ids", [])):
                mapping.setdefault(qid, set()).add(page_number)

    return {qid: sorted(list(page_set)) for qid, page_set in mapping.items()}


def _extract_alternatives(block: str) -> tuple[str, dict[str, str]]:
    alt_pattern = re.compile(r"\(([A-E])\)\s*(.*?)(?=(?:\([A-E]\)\s)|$)", re.DOTALL)
    matches = list(alt_pattern.finditer(block))
    if not matches:
        return _clean_text(block), {}

    enunciado = _clean_text(block[: matches[0].start()])
    alternativas: dict[str, str] = {}
    for match in matches:
        letter = match.group(1).strip().upper()
        content = _clean_text(match.group(2))
        alternativas[letter] = content

    return enunciado, alternativas


def extract_questions(
    pdf_path: Path,
    start_page: int,
    end_page: int,
    question_ids: list[int] | None,
    question_pages_hint: dict[int, list[int]] | None = None,
) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    if start_page < 1 or end_page < start_page:
        raise ValueError("Intervalo de paginas invalido")

    end_page = min(end_page, total_pages)

    chunks: list[str] = []
    for page_number in range(start_page, end_page + 1):
        page_text = reader.pages[page_number - 1].extract_text() or ""
        page_text = _clean_text(page_text)
        chunks.append(f"\n\n[[PAGE_{page_number}]]\n\n{page_text}\n")

    full_text = "\n".join(chunks)

    marker_pattern = re.compile(r"\{\s*0*(\d{1,3})\s*\}")
    marker_matches = list(marker_pattern.finditer(full_text))

    if not marker_matches:
        raise ValueError("Nao foram encontrados marcadores de questao no texto extraido")

    first_markers: dict[int, re.Match[str]] = {}
    ordered_found_ids: list[int] = []
    for match in marker_matches:
        qid = int(match.group(1))
        if qid not in first_markers:
            first_markers[qid] = match
            ordered_found_ids.append(qid)

    if question_ids:
        target_ids = [qid for qid in question_ids if qid in first_markers]
    else:
        target_ids = ordered_found_ids

    target_ids = sorted(target_ids)
    if not target_ids:
        raise ValueError("Nenhuma questao alvo encontrada no texto extraido")

    questions: list[dict[str, Any]] = []
    for idx, qid in enumerate(target_ids):
        start_match = first_markers[qid]
        start_index = start_match.end()

        end_index = len(full_text)
        for next_qid in target_ids[idx + 1 :]:
            next_match = first_markers[next_qid]
            if next_match.start() > start_index:
                end_index = next_match.start()
                break

        raw_block = full_text[start_index:end_index].strip()
        raw_block = re.sub(r"\[\[PAGE_\d+\]\]", "", raw_block)

        enunciado, alternativas = _extract_alternatives(raw_block)

        questions.append(
            {
                "question_id": qid,
                "page_numbers": (
                    question_pages_hint.get(qid, [])
                    if question_pages_hint is not None
                    else []
                ),
                "enunciado": enunciado,
                "alternativas": alternativas,
            }
        )

    return questions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrai enunciado e alternativas localmente do PDF (sem API)."
    )
    parser.add_argument("pdf_path", help="Caminho do PDF")
    parser.add_argument(
        "--output",
        default="artifacts/test7/questoes_texto_local.json",
        help="Arquivo JSON de saida",
    )
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=7)
    parser.add_argument(
        "--stage2",
        default=None,
        help="JSON stage2 para herdar ordem e paginas por questao",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not pdf_path.exists():
        print(f"Erro: PDF nao encontrado: {pdf_path}")
        return 1

    question_ids: list[int] | None = None
    question_pages_hint: dict[int, list[int]] | None = None
    source_stage2: str | None = None

    if args.stage2:
        stage2_path = Path(args.stage2).expanduser().resolve()
        stage2_payload = json.loads(stage2_path.read_text(encoding="utf-8"))
        if isinstance(stage2_payload, dict):
            question_ids = _question_ids_from_stage2(stage2_payload)
            question_pages_hint = _question_pages_from_stage2(stage2_payload)
            source_stage2 = str(stage2_path)

    questions = extract_questions(
        pdf_path=pdf_path,
        start_page=args.start_page,
        end_page=args.end_page,
        question_ids=question_ids,
        question_pages_hint=question_pages_hint,
    )

    payload = {
        "source_pdf": str(pdf_path),
        "source_stage2": source_stage2,
        "page_range": {
            "start_page": args.start_page,
            "end_page": args.end_page,
        },
        "questions": questions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Arquivo salvo em: {output_path}")
    print(f"Questoes extraidas: {len(questions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())