#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image, ImageChops, ImageFilter, ImageOps

try:
    from pdf2image import convert_from_path
except ImportError:  # Optional import; only needed in stage2.
    convert_from_path = None

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:  # Optional import; only needed when --max-pages is used in stage1.
    PdfReader = None
    PdfWriter = None


DEFAULT_MODEL = "gemini-2.5-flash"
QUESTION_IMAGE_NAME_RE = re.compile(r"^questao_(\d+)\.png$", re.IGNORECASE)


LETTER_LAYOUT = {
    "A": (0, 0),
    "D": (0, 1),
    "B": (1, 0),
    "E": (1, 1),
    "C": (2, 0),
}


PROMPT_STAGE1 = """
Voce recebera um PDF completo de prova.

Objetivo desta etapa (um unico prompt):
1) Identificar textos de apoio que pertencem a mais de uma questao.
2) Identificar questoes em que as alternativas sao visuais (imagem, grafico, tabela, diagrama).
3) Identificar imagens/figuras que ocupam mais de uma coluna ou uma largura muito grande.
4) Detectar fragmentos matematicos e registrar em formato LaTeX quando houver.

Retorne SOMENTE JSON valido, sem markdown, no formato:
{
  "shared_contexts": [
    {
      "context_id": "ctx_001",
            "text_excerpt": "trecho literal completo (sem resumir)",
      "question_ids": [14, 15],
      "page_numbers": [7],
      "confidence": 0.0
    }
  ],
  "questions_with_image_or_table_alternatives": [14, 31],
  "wide_visuals": [
    {
      "page_number": 7,
      "description": "mapa do brasil",
      "related_question_ids": [14],
      "spans_multiple_columns": true,
      "confidence": 0.0
    }
  ],
  "question_hints": [
    {
      "question_id": 14,
      "probable_page": 7,
      "has_shared_context": true,
      "has_visual_alternatives": false,
      "has_wide_visual": true
    }
    ],
    "latex_fragments": [
        {
            "snippet_latex": "x^2 + y^2 = z^2",
            "related_question_ids": [14],
            "page_numbers": [7],
            "confidence": 0.0
        }
  ]
}

Regras:
- Use apenas inteiros para question_id e page_number.
- Nao invente questoes nao existentes.
- Se nao houver dado em uma chave, retorne lista vazia.
- Em shared_contexts.text_excerpt, traga o texto integral do trecho compartilhado (nao resumir, nao truncar).
- Para matematica, prefira sintaxe LaTeX limpa em snippet_latex.
""".strip()


PROMPT_STAGE2_TEMPLATE = """
Voce recebera UMA imagem de pagina de prova e um bloco de hints da etapa 1.

Objetivo desta etapa:
- Definir recortes atomicos por questao para extracao posterior.
- Em paginas com duas colunas, use a coluna apenas como referencia visual; os segmentos devem ser por questao, nao por coluna inteira.
- Se uma questao tiver texto e imagem intercalados, o bbox deve conter a questao completa, incluindo todos os blocos de texto, figuras, tabelas, legendas e alternativas dessa questao.
- Se uma questao continuar em outro trecho da pagina, retorne mais de um segmento para a mesma questao com segment_id distinto.
- Se houver texto de apoio compartilhado, inclua esse texto nos segmentos das questoes relacionadas ou retorne um segmento de contexto com question_ids contendo todas elas.
- Se houver conteudo visual que cruza as colunas, use bbox de pagina larga somente para a questao afetada, nao para todas as questoes da pagina.

Hints da etapa 1 (JSON):
__HINTS_JSON__

Retorne SOMENTE JSON valido neste formato:
{
    "page_number": __PAGE_NUMBER__,
    "layout_mode": "split_lr" | "full_page",
  "reason": "texto curto",
  "segments": [
    {
            "segment_id": "p__PAGE_NUMBER_PADDED___l",
      "bbox": [ymin, xmin, ymax, xmax],
      "question_id": 13,
      "question_ids": [13],
      "segment_type": "question",
      "notes": "texto curto"
    }
  ]
}

Regras do bbox:
- Coordenadas normalizadas no intervalo [0,1000].
- Inteiros.
- ymin < ymax e xmin < xmax.
- Cada segmento de questao deve ter preferencialmente exatamente um question_id.
- Nao agrupe questoes diferentes no mesmo bbox apenas porque estao na mesma coluna.
- O bbox de uma questao comeca no numero da questao e termina antes do numero da proxima questao.
- Para questoes com imagens entre trechos de texto, mantenha o bbox amplo o suficiente para preservar a ordem de leitura.
- Use segment_type="question" para questoes e segment_type="shared_context" para texto/figura de apoio compartilhado.
- So use full_page quando uma unica questao ou contexto realmente ocupar largura grande.
""".strip()


PROMPT_STAGE3_TEMPLATE = """
Voce recebera uma imagem da questao __QUESTION_ID__.

Objetivo:
- Encontrar SOMENTE as areas de ilustracoes da questao __QUESTION_ID__ (grafico, tabela, mapa, figura, foto).
- Incluir titulo, legenda e credito/fonte quando estiverem imediatamente associados a ilustracao.
- Excluir enunciado e alternativas textuais.
- Se a imagem recebida ainda contiver pedacos de outras questoes, ignore tudo que nao pertencer a questao __QUESTION_ID__.
- Se texto e imagem estiverem intercalados, retorne a(s) ilustracao(oes) isolada(s), mas tambem descreva os blocos em ordem de leitura em content_blocks.

Contexto opcional da etapa 2:
__STAGE2_HINT__

Retorne SOMENTE JSON valido neste formato:
{
    "illustrations": [
        {
            "bbox": [ymin, xmin, ymax, xmax],
            "visual_type": "mapa|grafico|tabela|figura|foto|outro",
            "confidence": 0.0
        }
    ],
    "content_blocks": [
        {
            "kind": "text|illustration|alternative|shared_context",
            "bbox": [ymin, xmin, ymax, xmax],
            "text_role": "enunciado|legenda|fonte|alternativa|outro",
            "label": "A|B|C|D|E|null",
            "reading_order": 1
        }
    ]
}

Regras:
- Coordenadas normalizadas em [0,1000], inteiros.
- ymin < ymax e xmin < xmax.
- Se houver mais de uma ilustracao relevante na mesma questao, retorne multiplos itens em illustrations.
- Se nao houver ilustracao, retorne "illustrations": [].
- Se a imagem principal tiver legenda/fonte relevante logo acima/abaixo, amplie o bbox para preservar esse contexto.
- Evite recortes agressivos: inclua margem de seguranca para nao cortar texto ou elementos do visual nas bordas.
- content_blocks ajuda a preservar questoes com texto/imagem/texto; se nao tiver certeza, retorne lista vazia.
""".strip()


PROMPT_STAGE3_ALTERNATIVES_TEMPLATE = """
Voce recebera uma imagem da questao __QUESTION_ID__.

Objetivo:
- Detectar SOMENTE alternativas visuais (A, B, C, D, E) que contenham imagem, grafico, tabela, mapa, diagrama ou figura.
- Excluir alternativas puramente textuais.
- Excluir enunciado, texto de apoio e ilustracoes do corpo da questao que nao pertencam as alternativas.
- Se a imagem recebida ainda contiver pedacos de outras questoes, ignore tudo que nao pertencer a questao __QUESTION_ID__.

Contexto opcional da etapa 2:
__STAGE2_HINT__

Retorne SOMENTE JSON valido neste formato:
{
    "alternatives": {
        "A": {"bbox": [ymin, xmin, ymax, xmax], "confidence": 0.0},
        "B": {"bbox": [ymin, xmin, ymax, xmax], "confidence": 0.0}
    }
}

Regras:
- Use apenas as letras A, B, C, D, E.
- Inclua no JSON somente as letras que forem realmente visuais.
- Se nao houver alternativa visual, retorne "alternatives": {}.
- Coordenadas normalizadas em [0,1000], inteiros.
- ymin < ymax e xmin < xmax.
- Se uma alternativa visual estiver fragmentada, use um unico bbox englobando todos os fragmentos daquela letra.
- Nao inclua no bbox o marcador da letra da alternativa (por exemplo: "(A)", "A)", "A.").
- Inclua por completo o conteudo visual da alternativa e textos internos da propria figura/tabela.
- Prefira margem de seguranca leve para nao cortar texto da alternativa.
""".strip()


PROMPT_STAGE3_ALTERNATIVE_SINGLE_TEMPLATE = """
Voce recebera a imagem de UMA unica alternativa visual da questao (letra __LETTER__).

Objetivo:
- Delimitar SOMENTE o conteudo visual da alternativa (figura, grafico, tabela, diagrama), incluindo textos internos da propria figura.
- Excluir marcador de alternativa como "(__LETTER__)" ou variacoes equivalentes.

Retorne SOMENTE JSON valido no formato:
{
    "bbox": [ymin, xmin, ymax, xmax],
    "confidence": 0.0
}

Regras:
- Coordenadas normalizadas em [0,1000], inteiros.
- ymin < ymax e xmin < xmax.
- Inclua margem de seguranca leve para nao cortar texto relevante da propria alternativa.
- Nao inclua conteudo de alternativas vizinhas.
""".strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned)
    cleaned = re.sub(r"\\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise ValueError("Resposta do Gemini nao contem JSON valido")
        payload = json.loads(match.group(0))

    if not isinstance(payload, dict):
        raise ValueError("JSON retornado deve ser objeto")
    return payload


def _build_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY nao encontrada no ambiente/.env")
    return genai.Client(api_key=api_key)


def _request_json(
    client: genai.Client,
    model: str,
    contents: list[Any],
    temperature: float = 0.0,
) -> dict[str, Any]:
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
    except genai_errors.ClientError as exc:
        message = str(exc)
        if "404 NOT_FOUND" in message:
            raise RuntimeError(
                "Modelo Gemini nao encontrado para generateContent. "
                "Use gemini-2.5-flash ou gemini-2.0-flash."
            ) from exc
        if "429 RESOURCE_EXHAUSTED" in message:
            raise RuntimeError(
                "Cota da Gemini excedida para esta chave/projeto. "
                "Verifique faturamento/limites e tente novamente depois."
            ) from exc
        raise

    text = response.text or ""
    if not text.strip():
        raise ValueError("Resposta vazia do Gemini")
    return _extract_json(text)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Arquivo JSON invalido (esperado objeto): {path}")
    return payload


def _parse_pages(raw: str) -> list[int]:
    pages: set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        if "-" in chunk:
            start_str, end_str = chunk.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Intervalo invalido em --pages: {chunk}")
            for page in range(start, end + 1):
                pages.add(page)
        else:
            pages.add(int(chunk))

    if not pages:
        raise ValueError("--pages vazio")
    return sorted(pages)


def _collect_target_pages(
    stage1_payload: dict[str, Any],
    pages_override: str | None,
) -> list[int]:
    if pages_override:
        return _parse_pages(pages_override)

    pages: set[int] = set()

    for item in stage1_payload.get("wide_visuals", []):
        if isinstance(item, dict):
            page = item.get("page_number")
            if isinstance(page, int) and page > 0:
                pages.add(page)

    for item in stage1_payload.get("question_hints", []):
        if isinstance(item, dict):
            page = item.get("probable_page")
            if isinstance(page, int) and page > 0:
                pages.add(page)

    if not pages:
        raise ValueError(
            "Nao foi possivel inferir paginas alvo pela etapa 1. "
            "Passe --pages (ex: 7,8,10-12)."
        )

    return sorted(pages)


def _extract_int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []

    parsed: list[int] = []
    for value in values:
        if isinstance(value, int):
            parsed.append(value)
            continue

        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                parsed.append(int(stripped))

    return parsed


def _extract_segment_question_ids(segment: dict[str, Any]) -> list[int]:
    question_ids = _extract_int_list(segment.get("question_ids", []))
    question_id = segment.get("question_id")
    if isinstance(question_id, int):
        question_ids.append(question_id)
    elif isinstance(question_id, str) and question_id.strip().isdigit():
        question_ids.append(int(question_id.strip()))

    deduped: list[int] = []
    seen: set[int] = set()
    for qid in question_ids:
        if qid in seen:
            continue
        seen.add(qid)
        deduped.append(qid)
    return deduped


def _clean_extracted_text(text: str) -> str:
    text = text.replace("\u00ac", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(
        r"Concurso\s+Vestibular\s+FUVEST\s+\d{4}\s*–\s*Prova\s+V\d",
        "",
        text,
    )
    text = text.replace("#####", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
            for qid in _extract_segment_question_ids(segment):
                mapping.setdefault(qid, set()).add(page_number)

    return {qid: sorted(list(page_set)) for qid, page_set in mapping.items()}


def _extract_alternatives_from_block(block: str) -> tuple[str, dict[str, str]]:
    alt_pattern = re.compile(r"\(([A-E])\)\s*(.*?)(?=(?:\([A-E]\)\s)|$)", re.DOTALL)
    matches = list(alt_pattern.finditer(block))
    if not matches:
        return _clean_extracted_text(block), {}

    enunciado = _clean_extracted_text(block[: matches[0].start()])
    alternativas: dict[str, str] = {}
    for match in matches:
        letter = match.group(1).strip().upper()
        content = _clean_extracted_text(match.group(2))
        alternativas[letter] = content

    return enunciado, alternativas


def run_questions_text_local(
    pdf_path: Path,
    stage2_path: Path,
    output: Path,
    stage1_path: Path | None = None,
) -> None:
    if PdfReader is None:
        raise RuntimeError("pypdf nao instalado. Instale com: pip install pypdf")

    stage2_payload = _load_json(stage2_path)
    pages = stage2_payload.get("pages", [])
    if not isinstance(pages, list) or not pages:
        raise ValueError("Stage2 invalido para extracao local de texto")

    page_numbers = [
        page.get("page_number")
        for page in pages
        if isinstance(page, dict) and isinstance(page.get("page_number"), int)
    ]
    if not page_numbers:
        raise ValueError("Stage2 sem page_number valido")

    start_page = min(page_numbers)
    end_page = max(page_numbers)

    if stage1_path and stage1_path.exists():
        stage1_payload = _load_json(stage1_path)
        page_scope = stage1_payload.get("page_scope")
        if isinstance(page_scope, dict):
            scope_start = page_scope.get("start_page")
            scope_end = page_scope.get("end_page")
            if isinstance(scope_start, int) and scope_start > 0:
                start_page = max(start_page, scope_start)
            if isinstance(scope_end, int) and scope_end > 0:
                end_page = min(end_page, scope_end)

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError("PDF sem paginas")

    end_page = min(end_page, total_pages)
    if start_page < 1 or end_page < start_page:
        raise ValueError("Intervalo de paginas invalido para extracao local")

    question_ids = _question_ids_from_stage2(stage2_payload)
    question_pages = _question_pages_from_stage2(stage2_payload)

    chunks: list[str] = []
    for page_number in range(start_page, end_page + 1):
        page_text = reader.pages[page_number - 1].extract_text() or ""
        page_text = _clean_extracted_text(page_text)
        chunks.append(f"\n\n[[PAGE_{page_number}]]\n\n{page_text}\n")

    full_text = "\n".join(chunks)
    marker_patterns = [
        re.compile(r"\{\s*0*(\d{1,3})\s*\}"),
        re.compile(r"(?m)^\s*0*(\d{1,3})\s*$"),
    ]

    marker_entries: list[tuple[int, int, int]] = []
    for pattern in marker_patterns:
        for match in pattern.finditer(full_text):
            try:
                qid = int(match.group(1))
            except (TypeError, ValueError):
                continue
            marker_entries.append((match.start(), match.end(), qid))

    if question_ids:
        allowed_ids = set(question_ids)
        marker_entries = [entry for entry in marker_entries if entry[2] in allowed_ids]

    marker_entries.sort(key=lambda entry: entry[0])

    deduped_entries: list[tuple[int, int, int]] = []
    seen_signatures: set[tuple[int, int]] = set()
    for start_idx, end_idx, qid in marker_entries:
        signature = (start_idx, qid)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped_entries.append((start_idx, end_idx, qid))

    if not deduped_entries:
        raise ValueError("Nao foram encontrados marcadores de questao no texto extraido")

    first_markers: dict[int, tuple[int, int]] = {}
    for start_idx, end_idx, qid in deduped_entries:
        if qid not in first_markers or start_idx < first_markers[qid][0]:
            first_markers[qid] = (start_idx, end_idx)

    target_ids: list[int]
    if question_ids:
        target_ids = [qid for qid in question_ids if qid in first_markers]
    else:
        target_ids = sorted(first_markers.keys())

    if not target_ids:
        raise ValueError("Nenhuma questao alvo encontrada no texto extraido")

    questions: list[dict[str, Any]] = []
    for idx, qid in enumerate(target_ids):
        start_index = first_markers[qid][1]

        end_index = len(full_text)
        for next_qid in target_ids[idx + 1 :]:
            next_start = first_markers[next_qid][0]
            if next_start > start_index:
                end_index = next_start
                break

        raw_block = full_text[start_index:end_index].strip()
        raw_block = re.sub(r"\[\[PAGE_\d+\]\]", "", raw_block)

        enunciado, alternativas = _extract_alternatives_from_block(raw_block)
        questions.append(
            {
                "question_id": qid,
                "page_numbers": question_pages.get(qid, []),
                "enunciado": enunciado,
                "alternativas": alternativas,
            }
        )

    payload = {
        "source_pdf": str(pdf_path),
        "source_stage2": str(stage2_path),
        "page_range": {
            "start_page": start_page,
            "end_page": end_page,
        },
        "questions": questions,
    }

    _save_json(output, payload)
    print(f"Texto local salvo em: {output}")
    print(f"Questoes com texto extraidas: {len(questions)}")


def _bbox_pixels_to_norm(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> list[int]:
    left, top, right, bottom = bbox
    return [
        int((top / height) * 1000),
        int((left / width) * 1000),
        int((bottom / height) * 1000),
        int((right / width) * 1000),
    ]


def _grid_bboxes_for_visual_alternatives(
    width: int,
    height: int,
) -> dict[str, tuple[int, int, int, int]]:
    # Heuristica local para a regiao de alternativas visuais: metade inferior da pagina.
    y0 = int(height * 0.515)
    y1 = int(height * 0.905)
    x0 = int(width * 0.11)
    x1 = int(width * 0.915)

    region_w = max(1, x1 - x0)
    region_h = max(1, y1 - y0)

    col_gap = int(region_w * 0.055)
    row_gap = int(region_h * 0.075)

    col_w = (region_w - col_gap) // 2
    row_h = (region_h - 2 * row_gap) // 3

    boxes: dict[str, tuple[int, int, int, int]] = {}
    for letter, (row_idx, col_idx) in LETTER_LAYOUT.items():
        left = x0 + col_idx * (col_w + col_gap)
        top = y0 + row_idx * (row_h + row_gap)
        right = left + col_w
        bottom = top + row_h

        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))
        boxes[letter] = (left, top, right, bottom)

    return boxes


def _normalize_stage2_page_payload(
    payload: dict[str, Any],
    page_number: int,
) -> dict[str, Any]:
    layout_mode = str(payload.get("layout_mode", "")).strip().lower()
    if layout_mode == "split_2":
        layout_mode = "split_lr"
    if layout_mode not in {"split_lr", "full_page"}:
        layout_mode = "full_page"

    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError(
            f"Stage2 pagina {page_number}: campo segments ausente ou invalido"
        )

    normalized_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue

        bbox = _validate_bbox({"bbox": segment.get("bbox")})

        if layout_mode == "split_lr":
            default_suffix = "l" if index == 0 else "r"
        else:
            default_suffix = "full"

        raw_segment_id = segment.get("segment_id")
        if isinstance(raw_segment_id, str) and raw_segment_id.strip():
            segment_id = raw_segment_id.strip()
        else:
            segment_id = f"p{page_number:03d}_{default_suffix}"

        normalized_segments.append(
            {
                "segment_id": segment_id,
                "bbox": list(bbox),
                "question_ids": _extract_segment_question_ids(segment),
                "segment_type": str(
                    segment.get("segment_type", "question")
                ).strip() or "question",
                "notes": str(segment.get("notes", "")).strip(),
            }
        )

    if not normalized_segments:
        raise ValueError(f"Stage2 pagina {page_number}: nenhum segmento valido")

    return {
        "page_number": page_number,
        "layout_mode": layout_mode,
        "reason": str(payload.get("reason", "")).strip(),
        "segments": normalized_segments,
    }


def _build_stage1_pdf_part(
    pdf_path: Path,
    max_pages: int | None,
) -> tuple[types.Part, dict[str, int]]:
    if max_pages is None:
        pdf_bytes = pdf_path.read_bytes()
        return (
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            {"start_page": 1, "end_page": -1, "total_pdf_pages": -1},
        )

    if max_pages <= 0:
        raise ValueError("--max-pages deve ser maior que 0")

    if PdfReader is None or PdfWriter is None:
        raise RuntimeError(
            "pypdf nao instalado. Instale com: pip install pypdf"
        )

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError("PDF sem paginas")

    end_page = min(max_pages, total_pages)
    writer = PdfWriter()
    for page_index in range(end_page):
        writer.add_page(reader.pages[page_index])

    buffer = io.BytesIO()
    writer.write(buffer)
    pdf_bytes = buffer.getvalue()

    return (
        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
        {"start_page": 1, "end_page": end_page, "total_pdf_pages": total_pages},
    )


def _validate_bbox(payload: dict[str, Any]) -> tuple[int, int, int, int]:
    bbox = payload.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"bbox invalido: {bbox}")

    try:
        ymin, xmin, ymax, xmax = [int(v) for v in bbox]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bbox deve conter 4 inteiros: {bbox}") from exc

    for coord in (ymin, xmin, ymax, xmax):
        if coord < 0 or coord > 1000:
            raise ValueError(f"Coordenada fora do intervalo [0,1000]: {coord}")

    if ymin >= ymax or xmin >= xmax:
        raise ValueError(f"bbox sem area valida: {bbox}")

    return ymin, xmin, ymax, xmax


def _bbox_to_pixels(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = bbox
    left = int((xmin / 1000) * width)
    top = int((ymin / 1000) * height)
    right = int((xmax / 1000) * width)
    bottom = int((ymax / 1000) * height)

    left = max(0, min(left, width - 1))
    right = max(left + 1, min(right, width))
    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height))
    return left, top, right, bottom


def _expand_bbox_norm(
    bbox: tuple[int, int, int, int],
    pad_top: int,
    pad_left: int,
    pad_bottom: int,
    pad_right: int,
) -> tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = bbox
    expanded = (
        max(0, ymin - max(0, pad_top)),
        max(0, xmin - max(0, pad_left)),
        min(1000, ymax + max(0, pad_bottom)),
        min(1000, xmax + max(0, pad_right)),
    )
    if expanded[0] >= expanded[2] or expanded[1] >= expanded[3]:
        return bbox
    return expanded


def _filter_stage1_hints_for_page(
    stage1_payload: dict[str, Any],
    page_number: int,
) -> dict[str, Any]:
    shared = []
    for item in stage1_payload.get("shared_contexts", []):
        if not isinstance(item, dict):
            continue
        pages = item.get("page_numbers")
        if isinstance(pages, list) and page_number in pages:
            shared.append(item)

    wide = []
    for item in stage1_payload.get("wide_visuals", []):
        if isinstance(item, dict) and item.get("page_number") == page_number:
            wide.append(item)

    hints = []
    for item in stage1_payload.get("question_hints", []):
        if isinstance(item, dict) and item.get("probable_page") == page_number:
            hints.append(item)

    return {
        "page_number": page_number,
        "shared_contexts": shared,
        "wide_visuals": wide,
        "question_hints": hints,
        "questions_with_image_or_table_alternatives": stage1_payload.get(
            "questions_with_image_or_table_alternatives", []
        ),
    }


def _extract_question_number_from_filename(path: Path) -> int | None:
    match = QUESTION_IMAGE_NAME_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _bbox_area_norm(bbox: tuple[int, int, int, int]) -> int:
    ymin, xmin, ymax, xmax = bbox
    return max(0, ymax - ymin) * max(0, xmax - xmin)


def _bbox_iou_norm(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> float:
    aymin, axmin, aymax, axmax = bbox_a
    bymin, bxmin, bymax, bxmax = bbox_b

    inter_ymin = max(aymin, bymin)
    inter_xmin = max(axmin, bxmin)
    inter_ymax = min(aymax, bymax)
    inter_xmax = min(axmax, bxmax)

    inter_h = max(0, inter_ymax - inter_ymin)
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_area = inter_h * inter_w
    if inter_area <= 0:
        return 0.0

    area_a = _bbox_area_norm(bbox_a)
    area_b = _bbox_area_norm(bbox_b)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _extract_stage3_regions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_regions = payload.get("illustrations")
    if isinstance(raw_regions, list):
        regions: list[dict[str, Any]] = []
        for raw_region in raw_regions:
            if isinstance(raw_region, dict):
                bbox = _validate_bbox({"bbox": raw_region.get("bbox")})
                visual_type = raw_region.get("visual_type")
                confidence = _coerce_confidence(raw_region.get("confidence"))
            elif isinstance(raw_region, list):
                bbox = _validate_bbox({"bbox": raw_region})
                visual_type = payload.get("visual_type")
                confidence = _coerce_confidence(payload.get("confidence"))
            else:
                continue

            regions.append(
                {
                    "bbox_norm_0_1000": list(bbox),
                    "visual_type": str(visual_type).strip() if visual_type else None,
                    "confidence": confidence,
                }
            )

        deduped: list[dict[str, Any]] = []
        for region in regions:
            bbox_tuple = tuple(region["bbox_norm_0_1000"])
            if any(
                _bbox_iou_norm(bbox_tuple, tuple(saved["bbox_norm_0_1000"])) >= 0.9
                for saved in deduped
            ):
                continue
            deduped.append(region)

        deduped.sort(
            key=lambda item: (
                item["confidence"] if isinstance(item.get("confidence"), float) else -1.0,
                _bbox_area_norm(tuple(item["bbox_norm_0_1000"])),
            ),
            reverse=True,
        )
        return deduped

    # Compatibilidade retroativa: payload antigo com bbox unico.
    bbox = _validate_bbox(payload)
    return [
        {
            "bbox_norm_0_1000": list(bbox),
            "visual_type": (
                str(payload.get("visual_type")).strip()
                if payload.get("visual_type")
                else None
            ),
            "confidence": _coerce_confidence(payload.get("confidence")),
        }
    ]


def _extract_stage3_content_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_blocks = payload.get("content_blocks")
    if not isinstance(raw_blocks, list):
        return []

    blocks: list[dict[str, Any]] = []
    for raw_block in raw_blocks:
        if not isinstance(raw_block, dict):
            continue

        try:
            bbox = _validate_bbox({"bbox": raw_block.get("bbox")})
        except ValueError:
            continue

        reading_order = raw_block.get("reading_order")
        try:
            reading_order_int = int(reading_order)
        except (TypeError, ValueError):
            reading_order_int = len(blocks) + 1

        label = raw_block.get("label")
        if label is not None:
            label = str(label).strip().upper() or None
            if label == "NULL":
                label = None

        blocks.append(
            {
                "kind": str(raw_block.get("kind", "")).strip() or None,
                "bbox_norm_0_1000": list(bbox),
                "text_role": str(raw_block.get("text_role", "")).strip() or None,
                "label": label,
                "reading_order": reading_order_int,
            }
        )

    blocks.sort(key=lambda item: item["reading_order"])
    return blocks


def _extract_stage3_alternatives(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_alternatives = payload.get("alternatives")
    if not isinstance(raw_alternatives, dict):
        return {}

    extracted: dict[str, dict[str, Any]] = {}
    for letter in ("A", "B", "C", "D", "E"):
        raw_item = raw_alternatives.get(letter)
        if not isinstance(raw_item, dict):
            continue

        try:
            bbox = _validate_bbox({"bbox": raw_item.get("bbox")})
        except ValueError:
            continue

        extracted[letter] = {
            "bbox_norm_0_1000": list(bbox),
            "confidence": _coerce_confidence(raw_item.get("confidence")),
        }

    return extracted


def _extract_stage3_single_bbox(payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        bbox = _validate_bbox(payload)
    except ValueError:
        return None

    return {
        "bbox_norm_0_1000": list(bbox),
        "confidence": _coerce_confidence(payload.get("confidence")),
    }


def _connected_components_from_mask(mask: Image.Image) -> list[tuple[int, int, int, int, int]]:
    width, height = mask.size
    pixels = mask.tobytes()
    visited = bytearray(width * height)
    components: list[tuple[int, int, int, int, int]] = []

    for start_index, value in enumerate(pixels):
        if value == 0 or visited[start_index]:
            continue

        queue: deque[int] = deque([start_index])
        visited[start_index] = 1

        min_x = max_x = start_index % width
        min_y = max_y = start_index // width
        area = 0

        while queue:
            index = queue.pop()
            x = index % width
            y = index // width
            area += 1

            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

            if x > 0:
                left_index = index - 1
                if pixels[left_index] != 0 and not visited[left_index]:
                    visited[left_index] = 1
                    queue.append(left_index)
            if x < width - 1:
                right_index = index + 1
                if pixels[right_index] != 0 and not visited[right_index]:
                    visited[right_index] = 1
                    queue.append(right_index)
            if y > 0:
                up_index = index - width
                if pixels[up_index] != 0 and not visited[up_index]:
                    visited[up_index] = 1
                    queue.append(up_index)
            if y < height - 1:
                down_index = index + width
                if pixels[down_index] != 0 and not visited[down_index]:
                    visited[down_index] = 1
                    queue.append(down_index)

        components.append((min_x, min_y, max_x + 1, max_y + 1, area))

    return components


def _bbox_iou_pixels(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> float:
    left_a, top_a, right_a, bottom_a = bbox_a
    left_b, top_b, right_b, bottom_b = bbox_b

    inter_left = max(left_a, left_b)
    inter_top = max(top_a, top_b)
    inter_right = min(right_a, right_b)
    inter_bottom = min(bottom_a, bottom_b)

    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, right_a - left_a) * max(0, bottom_a - top_a)
    area_b = max(0, right_b - left_b) * max(0, bottom_b - top_b)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _bbox_overlap_ratio_pixels(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> float:
    left_a, top_a, right_a, bottom_a = bbox_a
    left_b, top_b, right_b, bottom_b = bbox_b

    inter_left = max(left_a, left_b)
    inter_top = max(top_a, top_b)
    inter_right = min(right_a, right_b)
    inter_bottom = min(bottom_a, bottom_b)

    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(1, (right_a - left_a) * (bottom_a - top_a))
    return inter_area / area_a


def _expand_bbox_pixels(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    pad_top: int,
    pad_left: int,
    pad_bottom: int,
    pad_right: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    expanded = (
        max(0, left - max(0, pad_left)),
        max(0, top - max(0, pad_top)),
        min(width, right + max(0, pad_right)),
        min(height, bottom + max(0, pad_bottom)),
    )
    if expanded[0] >= expanded[2] or expanded[1] >= expanded[3]:
        return bbox
    return expanded


def _edge_dark_ratios(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    threshold: int = 208,
    strip_ratio: float = 0.035,
) -> dict[str, float]:
    left, top, right, bottom = bbox
    crop = image.crop((left, top, right, bottom)).convert("L")
    crop = ImageOps.autocontrast(crop)

    width, height = crop.size
    if width <= 1 or height <= 1:
        return {"top": 0.0, "left": 0.0, "bottom": 0.0, "right": 0.0}

    strip = max(2, int(min(width, height) * max(0.01, strip_ratio)))
    pixels = crop.tobytes()

    def _ratio_in_rect(x0: int, y0: int, x1: int, y1: int) -> float:
        dark = 0
        total = 0
        for yy in range(y0, y1):
            row_start = yy * width
            for xx in range(x0, x1):
                total += 1
                if pixels[row_start + xx] < threshold:
                    dark += 1
        return dark / max(1, total)

    top_ratio = _ratio_in_rect(0, 0, width, min(height, strip))
    bottom_ratio = _ratio_in_rect(0, max(0, height - strip), width, height)
    left_ratio = _ratio_in_rect(0, 0, min(width, strip), height)
    right_ratio = _ratio_in_rect(max(0, width - strip), 0, width, height)

    return {
        "top": top_ratio,
        "left": left_ratio,
        "bottom": bottom_ratio,
        "right": right_ratio,
    }


def _expand_bbox_for_dark_edges(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    max_iterations: int,
    edge_threshold: float,
    step_top: float,
    step_left: float,
    step_bottom: float,
    step_right: float,
) -> tuple[int, int, int, int]:
    width, height = image.size
    current = bbox

    for _ in range(max_iterations):
        left, top, right, bottom = current
        box_w = max(1, right - left)
        box_h = max(1, bottom - top)
        ratios = _edge_dark_ratios(image, current)

        pad_top = int(box_h * step_top) if ratios["top"] >= edge_threshold else 0
        pad_left = int(box_w * step_left) if ratios["left"] >= edge_threshold else 0
        pad_bottom = int(box_h * step_bottom) if ratios["bottom"] >= edge_threshold else 0
        pad_right = int(box_w * step_right) if ratios["right"] >= edge_threshold else 0

        if pad_top == 0 and pad_left == 0 and pad_bottom == 0 and pad_right == 0:
            break

        current = _expand_bbox_pixels(
            current,
            width,
            height,
            pad_top=pad_top,
            pad_left=pad_left,
            pad_bottom=pad_bottom,
            pad_right=pad_right,
        )

    return current


def _is_alternative_bbox_suspect(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    grid_bbox: tuple[int, int, int, int],
) -> bool:
    box_w = max(1, bbox[2] - bbox[0])
    box_h = max(1, bbox[3] - bbox[1])
    box_area = box_w * box_h

    grid_w = max(1, grid_bbox[2] - grid_bbox[0])
    grid_h = max(1, grid_bbox[3] - grid_bbox[1])
    grid_area = grid_w * grid_h

    too_small_vs_grid = box_area < int(grid_area * 0.62)
    too_narrow = box_w < int(grid_w * 0.70)
    too_short = box_h < int(grid_h * 0.68)

    edge_ratios = _edge_dark_ratios(image, bbox)
    likely_cut = edge_ratios["top"] >= 0.12 or edge_ratios["left"] >= 0.12

    return too_small_vs_grid or ((too_narrow or too_short) and likely_cut)


def _grid_fallback_bbox_for_alternative(
    image: Image.Image,
    grid_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = grid_bbox
    grid_w = max(1, right - left)
    grid_h = max(1, bottom - top)

    return _expand_bbox_pixels(
        grid_bbox,
        image.width,
        image.height,
        pad_top=int(grid_h * 0.10),
        pad_left=int(grid_w * 0.07),
        pad_bottom=int(grid_h * 0.04),
        pad_right=int(grid_w * 0.05),
    )


def _detect_local_visual_alternative_bboxes(
    image: Image.Image,
) -> dict[str, tuple[int, int, int, int]]:
    width, height = image.size
    if width <= 0 or height <= 0:
        return {}

    y_start = int(height * 0.46)
    lower = image.crop((0, y_start, width, height)).convert("RGB")
    r_channel, g_channel, b_channel = lower.split()

    # Segmenta os quadrados azul-ciano das tabelas de alternativas.
    r_mask = r_channel.point(lambda value: 255 if value <= 175 else 0, mode="L")
    g_mask = g_channel.point(lambda value: 255 if value >= 110 else 0, mode="L")
    b_mask = b_channel.point(lambda value: 255 if value >= 165 else 0, mode="L")
    bg_delta = ImageChops.subtract(b_channel, g_channel)
    delta_mask = bg_delta.point(lambda value: 255 if value <= 100 else 0, mode="L")

    mask = ImageChops.multiply(r_mask, g_mask)
    mask = ImageChops.multiply(mask, b_mask)
    mask = ImageChops.multiply(mask, delta_mask)

    mask = mask.filter(ImageFilter.MaxFilter(9))
    mask = mask.filter(ImageFilter.MaxFilter(9))
    mask = mask.filter(ImageFilter.MinFilter(5))

    components = _connected_components_from_mask(mask)
    if not components:
        return {}

    min_area = int((lower.width * lower.height) * 0.002)
    candidates: list[tuple[int, int, int, int, int]] = []
    for min_x, min_y, max_x, max_y, area in components:
        comp_width = max_x - min_x
        comp_height = max_y - min_y
        if area < min_area:
            continue
        if comp_width < int(lower.width * 0.08):
            continue
        if comp_height < int(lower.height * 0.07):
            continue

        fill_ratio = area / max(1, comp_width * comp_height)
        if fill_ratio < 0.15:
            continue

        candidates.append((min_x, min_y + y_start, max_x, max_y + y_start, area))

    if len(candidates) < 5:
        return {}

    candidates.sort(key=lambda item: item[4], reverse=True)
    candidates = candidates[:5]

    boxes_with_centers: list[tuple[tuple[int, int, int, int], float, float]] = []
    for min_x, min_y, max_x, max_y, _ in candidates:
        bbox = (min_x, min_y, max_x, max_y)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        boxes_with_centers.append((bbox, center_x, center_y))

    min_center_x = min(item[1] for item in boxes_with_centers)
    max_center_x = max(item[1] for item in boxes_with_centers)
    split_x = (min_center_x + max_center_x) / 2

    left_column = [item for item in boxes_with_centers if item[1] <= split_x]
    right_column = [item for item in boxes_with_centers if item[1] > split_x]

    if len(left_column) < 3 or len(right_column) < 2:
        boxes_with_centers.sort(key=lambda item: item[1])
        left_column = boxes_with_centers[:3]
        right_column = boxes_with_centers[3:]

    left_column.sort(key=lambda item: item[2])
    right_column.sort(key=lambda item: item[2])

    if len(left_column) < 3 or len(right_column) < 2:
        return {}

    letter_to_box = {
        "A": left_column[0][0],
        "B": left_column[1][0],
        "C": left_column[2][0],
        "D": right_column[0][0],
        "E": right_column[1][0],
    }

    expanded_boxes: dict[str, tuple[int, int, int, int]] = {}
    for letter, bbox in letter_to_box.items():
        box_w = max(1, bbox[2] - bbox[0])
        box_h = max(1, bbox[3] - bbox[1])
        expanded_boxes[letter] = _expand_bbox_pixels(
            bbox,
            width,
            height,
            pad_top=int(box_h * 0.10),
            pad_left=int(box_w * 0.10),
            pad_bottom=int(box_h * 0.06),
            pad_right=int(box_w * 0.06),
        )

    return expanded_boxes


def _tighten_bbox_to_dark_content(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    crop = image.crop((left, top, right, bottom)).convert("L")
    crop = ImageOps.autocontrast(crop)
    mask = crop.point(lambda pixel: 255 if pixel < 215 else 0, mode="L")

    components = _connected_components_from_mask(mask)
    if not components:
        return bbox

    cell_width, cell_height = crop.size
    cell_area = max(1, cell_width * cell_height)
    min_component_area = int(cell_area * 0.01)
    min_component_width = int(cell_width * 0.12)
    min_component_height = int(cell_height * 0.12)

    selected: list[tuple[int, int, int, int, int]] = []
    for min_x, min_y, max_x, max_y, area in components:
        comp_width = max_x - min_x
        comp_height = max_y - min_y
        if area < min_component_area:
            continue
        if comp_width < min_component_width or comp_height < min_component_height:
            continue

        selected.append((min_x, min_y, max_x, max_y, area))

    if not selected:
        return bbox

    min_x = min(item[0] for item in selected)
    min_y = min(item[1] for item in selected)
    max_x = max(item[2] for item in selected)
    max_y = max(item[3] for item in selected)

    pad_x = max(4, int((max_x - min_x) * 0.06))
    pad_y = max(4, int((max_y - min_y) * 0.06))
    min_x = max(0, min_x - pad_x)
    min_y = max(0, min_y - pad_y)
    max_x = min(cell_width, max_x + pad_x)
    max_y = min(cell_height, max_y + pad_y)

    tightened = (left + min_x, top + min_y, left + max_x, top + max_y)
    tightened_area = max(1, (tightened[2] - tightened[0]) * (tightened[3] - tightened[1]))
    original_area = max(1, (right - left) * (bottom - top))
    if tightened_area < int(original_area * 0.12):
        return bbox
    return tightened


def _tighten_visual_alternative_bbox(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    tightened = _tighten_bbox_to_dark_content(image, bbox)

    left, top, right, bottom = tightened
    crop = image.crop((left, top, right, bottom)).convert("L")
    crop = ImageOps.autocontrast(crop)
    mask = crop.point(lambda pixel: 255 if pixel < 212 else 0, mode="L")

    components = _connected_components_from_mask(mask)
    if not components:
        return tightened

    cell_width, cell_height = crop.size
    cell_area = max(1, cell_width * cell_height)
    min_component_area = int(cell_area * 0.004)

    kept: list[tuple[int, int, int, int, int]] = []
    for min_x, min_y, max_x, max_y, area in components:
        comp_width = max_x - min_x
        comp_height = max_y - min_y
        if area < min_component_area:
            continue

        near_left_edge = min_x <= int(cell_width * 0.06)
        compact_component = (
            comp_width <= int(cell_width * 0.08)
            and comp_height <= int(cell_height * 0.20)
        )
        if near_left_edge and compact_component:
            # Remove apenas marcadores muito pequenos colados na borda esquerda.
            continue

        kept.append((min_x, min_y, max_x, max_y, area))

    if not kept:
        return tightened

    min_x = min(item[0] for item in kept)
    min_y = min(item[1] for item in kept)
    max_x = max(item[2] for item in kept)
    max_y = max(item[3] for item in kept)

    pad_x = max(3, int((max_x - min_x) * 0.05))
    pad_y = max(3, int((max_y - min_y) * 0.05))
    min_x = max(0, min_x - pad_x)
    min_y = max(0, min_y - pad_y)
    max_x = min(cell_width, max_x + pad_x)
    max_y = min(cell_height, max_y + pad_y)

    refined = (left + min_x, top + min_y, left + max_x, top + max_y)
    refined_area = max(1, (refined[2] - refined[0]) * (refined[3] - refined[1]))
    original_area = max(1, (right - left) * (bottom - top))
    if refined_area < int(original_area * 0.45):
        return tightened
    return refined


def _component_distance_pixels(
    component_a: tuple[int, int, int, int, int],
    component_b: tuple[int, int, int, int, int],
) -> tuple[int, int]:
    a_left, a_top, a_right, a_bottom, _ = component_a
    b_left, b_top, b_right, b_bottom, _ = component_b

    dx = max(0, max(a_left, b_left) - min(a_right, b_right))
    dy = max(0, max(a_top, b_top) - min(a_bottom, b_bottom))
    return dx, dy


def _tighten_illustration_bbox_to_primary_cluster(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    crop = image.crop((left, top, right, bottom)).convert("L")
    crop = ImageOps.autocontrast(crop)
    mask = crop.point(lambda pixel: 255 if pixel < 195 else 0, mode="L")
    mask = mask.filter(ImageFilter.MaxFilter(3))

    components = _connected_components_from_mask(mask)
    if not components:
        return bbox

    cell_width, cell_height = crop.size
    cell_area = max(1, cell_width * cell_height)
    min_component_area = max(24, int(cell_area * 0.0005))

    ranked: list[tuple[int, int, int, int, int]] = []
    for component in components:
        if component[4] >= min_component_area:
            ranked.append(component)

    if not ranked:
        return bbox

    ranked.sort(key=lambda item: item[4], reverse=True)

    cluster: list[tuple[int, int, int, int, int]] = [ranked[0]]
    used_indexes = {0}
    gap_x = max(10, int(cell_width * 0.05))
    gap_y = max(12, int(cell_height * 0.08))

    changed = True
    while changed:
        changed = False
        for index, component in enumerate(ranked):
            if index in used_indexes:
                continue

            near_cluster = False
            for kept in cluster:
                distance_x, distance_y = _component_distance_pixels(component, kept)
                if distance_x <= gap_x and distance_y <= gap_y:
                    near_cluster = True
                    break

            if near_cluster:
                cluster.append(component)
                used_indexes.add(index)
                changed = True

    min_x = min(item[0] for item in cluster)
    min_y = min(item[1] for item in cluster)
    max_x = max(item[2] for item in cluster)
    max_y = max(item[3] for item in cluster)

    pad_x = max(4, int((max_x - min_x) * 0.05))
    pad_y = max(6, int((max_y - min_y) * 0.06))
    min_x = max(0, min_x - pad_x)
    min_y = max(0, min_y - pad_y)
    max_x = min(cell_width, max_x + pad_x)
    max_y = min(cell_height, max_y + pad_y)

    refined = (left + min_x, top + min_y, left + max_x, top + max_y)
    refined_area = max(1, (refined[2] - refined[0]) * (refined[3] - refined[1]))
    original_area = max(1, (right - left) * (bottom - top))
    if refined_area < int(original_area * 0.55):
        return bbox
    return refined


def _detect_local_illustration_bboxes(
    image: Image.Image,
    max_regions: int,
    excluded_regions_px: list[tuple[int, int, int, int]] | None = None,
    avoid_top_text_blocks: bool = False,
) -> list[tuple[int, int, int, int]]:
    width, height = image.size
    if width <= 0 or height <= 0:
        return []

    max_dimension = max(width, height)
    scale = max_dimension / 1200 if max_dimension > 1200 else 1.0
    work_width = max(1, int(width / scale))
    work_height = max(1, int(height / scale))

    work = image.convert("L")
    if scale > 1.0:
        work = work.resize((work_width, work_height), Image.Resampling.BILINEAR)

    work = ImageOps.autocontrast(work)
    binary = work.point(lambda pixel: 255 if pixel < 195 else 0, mode="L")
    binary = binary.filter(ImageFilter.MaxFilter(5))
    binary = binary.filter(ImageFilter.MinFilter(3))
    binary = binary.filter(ImageFilter.MaxFilter(3))

    components = _connected_components_from_mask(binary)
    if not components:
        return []

    total_area = work_width * work_height
    min_component_area = int(total_area * 0.010)
    min_component_width = int(work_width * 0.10)
    min_component_height = int(work_height * 0.07)

    candidates: list[tuple[int, int, int, int, int]] = []
    for min_x, min_y, max_x, max_y, area in components:
        comp_width = max_x - min_x
        comp_height = max_y - min_y
        if area < min_component_area:
            continue
        if comp_width < min_component_width or comp_height < min_component_height:
            continue

        bbox_area = comp_width * comp_height
        fill_ratio = area / max(1, bbox_area)
        if fill_ratio < 0.16:
            continue

        if min_y < int(work_height * 0.04) and comp_height < int(work_height * 0.20):
            continue

        candidates.append((min_x, min_y, max_x, max_y, area))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[4], reverse=True)

    selected: list[tuple[int, int, int, int]] = []
    for min_x, min_y, max_x, max_y, _ in candidates:
        bbox_norm = (
            int((min_y / work_height) * 1000),
            int((min_x / work_width) * 1000),
            int((max_y / work_height) * 1000),
            int((max_x / work_width) * 1000),
        )
        bbox_norm = _validate_bbox({"bbox": list(bbox_norm)})

        if any(_bbox_iou_norm(bbox_norm, existing) >= 0.55 for existing in selected):
            continue

        selected.append(bbox_norm)
        if len(selected) >= max_regions:
            break

    bboxes_px: list[tuple[int, int, int, int]] = []
    for bbox_norm in selected:
        left, top, right, bottom = _bbox_to_pixels(bbox_norm, width, height)

        x_margin = max(4, int((right - left) * 0.02))
        y_margin = max(4, int((bottom - top) * 0.02))
        left = max(0, left - x_margin)
        top = max(0, top - y_margin)
        right = min(width, right + x_margin)
        bottom = min(height, bottom + y_margin)
        if right - left < 2 or bottom - top < 2:
            continue
        bbox_px = (left, top, right, bottom)

        if excluded_regions_px:
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            should_exclude = False
            for excluded in excluded_regions_px:
                ex_left, ex_top, ex_right, ex_bottom = excluded
                if (
                    ex_left <= center_x <= ex_right
                    and ex_top <= center_y <= ex_bottom
                ):
                    should_exclude = True
                    break
                if _bbox_overlap_ratio_pixels(bbox_px, excluded) >= 0.25:
                    should_exclude = True
                    break
                if _bbox_iou_pixels(bbox_px, excluded) >= 0.18:
                    should_exclude = True
                    break
            if should_exclude:
                continue

        if avoid_top_text_blocks:
            bbox_height = bottom - top
            if top < int(height * 0.12) and bbox_height < int(height * 0.18):
                continue

        bboxes_px.append(bbox_px)

    return bboxes_px


def _stage2_hint_for_question(stage2_payload: dict[str, Any], question_number: int) -> str:
    pages = stage2_payload.get("pages", [])
    if not isinstance(pages, list):
        return "Sem hint da etapa 2"

    for page_item in pages:
        if not isinstance(page_item, dict):
            continue
        for segment in page_item.get("segments", []):
            if not isinstance(segment, dict):
                continue
            if question_number in _extract_segment_question_ids(segment):
                return json.dumps(segment, ensure_ascii=False)

    return "Sem hint da etapa 2"


def _question_ids_from_stage2(stage2_payload: dict[str, Any]) -> list[int]:
    question_ids: set[int] = set()

    pages = stage2_payload.get("pages", [])
    if not isinstance(pages, list):
        return []

    for page_item in pages:
        if not isinstance(page_item, dict):
            continue
        for segment in page_item.get("segments", []):
            if not isinstance(segment, dict):
                continue
            for question_id in _extract_segment_question_ids(segment):
                question_ids.add(question_id)

    return sorted(question_ids)


def _compose_question_segments(crops: list[Image.Image]) -> Image.Image:
    if len(crops) == 1:
        return crops[0]

    normalized = [crop.convert("RGB") for crop in crops]
    gap = max(10, int(max(crop.height for crop in normalized) * 0.015))
    width = max(crop.width for crop in normalized)
    height = sum(crop.height for crop in normalized) + gap * (len(normalized) - 1)
    composed = Image.new("RGB", (width, height), "white")

    cursor_y = 0
    for crop in normalized:
        composed.paste(crop, (0, cursor_y))
        cursor_y += crop.height + gap

    return composed


def _build_question_images_from_stage2_segments(
    pdf_path: Path,
    stage2_path: Path,
    output_dir: Path,
    dpi: int = 220,
) -> int:
    if convert_from_path is None:
        raise RuntimeError(
            "pdf2image nao instalado. Instale com: pip install pdf2image"
        )

    stage2_payload = _load_json(stage2_path)
    pages = stage2_payload.get("pages", [])
    if not isinstance(pages, list) or not pages:
        raise ValueError("Stage2 invalido para gerar imagens de questao")

    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("questao_*.png"):
        existing.unlink(missing_ok=True)

    crops_by_question: dict[int, list[tuple[int, int, int, Image.Image]]] = {}

    for page_item in pages:
        if not isinstance(page_item, dict):
            continue

        page_number = page_item.get("page_number")
        if not isinstance(page_number, int) or page_number <= 0:
            continue

        page_images = convert_from_path(
            str(pdf_path),
            first_page=page_number,
            last_page=page_number,
            dpi=dpi,
        )
        if not page_images:
            continue

        page_image = page_images[0]
        width, height = page_image.width, page_image.height

        segments = page_item.get("segments", [])
        if not isinstance(segments, list):
            continue

        for segment in segments:
            if not isinstance(segment, dict):
                continue

            try:
                bbox_norm = _validate_bbox({"bbox": segment.get("bbox")})
            except ValueError:
                continue

            qids = _extract_segment_question_ids(segment)
            if not qids:
                continue

            left, top, right, bottom = _bbox_to_pixels(bbox_norm, width, height)
            if right <= left or bottom <= top:
                continue

            segment_crop = page_image.crop((left, top, right, bottom))
            for qid in qids:
                crops_by_question.setdefault(qid, []).append(
                    (page_number, top, left, segment_crop.copy())
                )

    generated = 0
    for qid, entries in sorted(crops_by_question.items()):
        entries.sort(key=lambda item: (item[0], item[1], item[2]))
        output_image = output_dir / f"questao_{qid:02d}.png"
        composed = _compose_question_segments([entry[3] for entry in entries])
        composed.save(output_image)
        generated += 1

    return generated


def run_stage1(
    pdf_path: Path,
    output: Path,
    model: str,
    max_pages: int | None,
) -> None:
    client = _build_client()

    pdf_part, page_scope = _build_stage1_pdf_part(pdf_path, max_pages)

    payload = _request_json(
        client=client,
        model=model,
        contents=[PROMPT_STAGE1, pdf_part],
    )

    enriched = {
        "stage": "stage1",
        "model": model,
        "created_at": _now_iso(),
        "source_pdf": str(pdf_path),
        "page_scope": page_scope,
        "data": payload,
    }
    _save_json(output, enriched)
    print(f"Stage1 salvo em: {output}")


def run_stage2(
    pdf_path: Path,
    stage1_path: Path,
    output: Path,
    model: str,
    pages_override: str | None,
    dpi: int,
) -> None:
    if convert_from_path is None:
        raise RuntimeError(
            "pdf2image nao instalado. Instale com: pip install pdf2image"
        )

    stage1_payload = _load_json(stage1_path)
    stage1_data = stage1_payload.get("data")
    if not isinstance(stage1_data, dict):
        raise ValueError("Arquivo stage1 invalido: chave data ausente")

    target_pages = _collect_target_pages(stage1_data, pages_override)
    client = _build_client()

    page_results: list[dict[str, Any]] = []
    for page_number in target_pages:
        page_images = convert_from_path(
            str(pdf_path),
            first_page=page_number,
            last_page=page_number,
            dpi=dpi,
        )
        if not page_images:
            raise RuntimeError(f"Falha ao renderizar pagina {page_number}")

        page_image = page_images[0]
        page_hints = _filter_stage1_hints_for_page(stage1_data, page_number)
        prompt = (
            PROMPT_STAGE2_TEMPLATE
            .replace("__HINTS_JSON__", json.dumps(page_hints, ensure_ascii=False))
            .replace("__PAGE_NUMBER__", str(page_number))
            .replace("__PAGE_NUMBER_PADDED__", f"{page_number:03d}")
        )

        raw_page_payload = _request_json(
            client=client,
            model=model,
            contents=[prompt, page_image],
        )
        page_payload = _normalize_stage2_page_payload(raw_page_payload, page_number)
        page_results.append(page_payload)
        print(f"Stage2 pagina {page_number}: OK")

    enriched = {
        "stage": "stage2",
        "model": model,
        "created_at": _now_iso(),
        "source_pdf": str(pdf_path),
        "source_stage1": str(stage1_path),
        "pages": page_results,
    }
    _save_json(output, enriched)
    print(f"Stage2 salvo em: {output}")


def run_stage3(
    question_images_dir: Path,
    output_dir: Path,
    output_metadata: Path,
    model: str,
    stage2_path: Path | None,
    only_stage2_questions: bool,
    resume: bool,
    limit: int | None,
) -> None:
    client = _build_client()

    stage2_payload: dict[str, Any] = {}
    if stage2_path:
        stage2_payload = _load_json(stage2_path)

    image_paths = sorted(
        image_path
        for image_path in question_images_dir.glob("questao_*.png")
        if _extract_question_number_from_filename(image_path) is not None
    )

    filtered_question_ids: list[int] | None = None
    if stage2_payload and only_stage2_questions:
        filtered_question_ids = _question_ids_from_stage2(stage2_payload)
        if not filtered_question_ids:
            raise ValueError(
                "Stage2 nao retornou question_ids para filtrar o stage3. "
                "Use --all-questions para ignorar esse filtro."
            )

        allowed = set(filtered_question_ids)
        filtered_image_paths: list[Path] = []
        for image_path in image_paths:
            question_number = _extract_question_number_from_filename(image_path)
            if question_number is not None and question_number in allowed:
                filtered_image_paths.append(image_path)
        image_paths = filtered_image_paths

    if limit is not None:
        image_paths = image_paths[:limit]

    if not image_paths:
        raise ValueError(
            f"Nenhuma imagem de questao encontrada em: {question_images_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    if resume and output_metadata.exists():
        previous = _load_json(output_metadata)
        previous_items = previous.get("items")
        if isinstance(previous_items, list):
            items = [item for item in previous_items if isinstance(item, dict)]

    processed_ids = {
        item.get("question_id")
        for item in items
        if isinstance(item.get("question_id"), int)
    }

    def save_progress() -> None:
        enriched = {
            "stage": "stage3",
            "model": model,
            "created_at": _now_iso(),
            "question_images_dir": str(question_images_dir),
            "output_dir": str(output_dir),
            "source_stage2": str(stage2_path) if stage2_path else None,
            "filtered_question_ids": filtered_question_ids,
            "items": items,
        }
        _save_json(output_metadata, enriched)

    for image_path in image_paths:
        question_number = _extract_question_number_from_filename(image_path)
        if question_number is None:
            continue
        if question_number in processed_ids:
            continue

        hint = (
            _stage2_hint_for_question(stage2_payload, question_number)
            if stage2_payload
            else "Sem hint da etapa 2"
        )
        prompt = PROMPT_STAGE3_TEMPLATE.replace("__STAGE2_HINT__", hint)
        prompt = prompt.replace("__QUESTION_ID__", str(question_number))

        with Image.open(image_path) as image:
            payload = _request_json(
                client=client,
                model=model,
                contents=[prompt, image],
            )

            regions = _extract_stage3_regions(payload)
            content_blocks = _extract_stage3_content_blocks(payload)
            region_items: list[dict[str, Any]] = []

            for index, region in enumerate(regions, start=1):
                bbox_norm = _expand_bbox_norm(
                    tuple(region["bbox_norm_0_1000"]),
                    pad_top=20,
                    pad_left=60,
                    pad_bottom=40,
                    pad_right=18,
                )
                bbox_px = _bbox_to_pixels(bbox_norm, image.width, image.height)
                bbox_px = _expand_bbox_for_dark_edges(
                    image,
                    bbox_px,
                    max_iterations=3,
                    edge_threshold=0.11,
                    step_top=0.09,
                    step_left=0.14,
                    step_bottom=0.08,
                    step_right=0.07,
                )
                bbox_norm = _bbox_pixels_to_norm(bbox_px, image.width, image.height)
                cropped = image.crop(bbox_px)

                if len(regions) == 1:
                    output_name = f"questao_{question_number:02d}_ilustracao.png"
                else:
                    output_name = (
                        f"questao_{question_number:02d}_ilustracao_{index:02d}.png"
                    )
                output_path = output_dir / output_name
                cropped.save(output_path)

                region_items.append(
                    {
                        "output_image": str(output_path),
                        "bbox_norm_0_1000": list(bbox_norm),
                        "bbox_pixels": list(bbox_px),
                        "visual_type": region.get("visual_type"),
                        "confidence": region.get("confidence"),
                        "rank": index,
                    }
                )

        item = {
            "question_id": question_number,
            "input_image": str(image_path),
            "regions": region_items,
            "content_blocks": content_blocks,
        }
        if region_items:
            item.update(
                {
                    "output_image": region_items[0]["output_image"],
                    "bbox_norm_0_1000": region_items[0]["bbox_norm_0_1000"],
                    "bbox_pixels": region_items[0]["bbox_pixels"],
                    "visual_type": region_items[0]["visual_type"],
                    "confidence": region_items[0]["confidence"],
                }
            )
        items.append(item)
        processed_ids.add(question_number)
        save_progress()
        if region_items:
            print(f"Stage3 questao {question_number}: {len(region_items)} ilustracao(oes)")
        else:
            print(f"Stage3 questao {question_number}: sem ilustracao detectada")

    save_progress()
    print(f"Stage3 metadados salvos em: {output_metadata}")


def run_stage3_alternatives(
    stage1_path: Path,
    question_images_dir: Path,
    output_dir: Path,
    output_metadata: Path,
    model: str,
    stage2_path: Path | None,
    only_stage2_questions: bool,
    resume: bool,
    limit: int | None,
    per_letter_refine: bool,
) -> None:
    client = _build_client()

    stage1_payload = _load_json(stage1_path)
    stage1_data = stage1_payload.get("data")
    if not isinstance(stage1_data, dict):
        raise ValueError("Arquivo stage1 invalido: chave data ausente")

    question_ids = sorted(
        set(
            _extract_int_list(
                stage1_data.get("questions_with_image_or_table_alternatives", [])
            )
        )
    )
    if not question_ids:
        raise ValueError(
            "Stage1 nao retornou questoes com alternativas visuais para processar"
        )

    stage2_payload: dict[str, Any] = {}
    filtered_by_stage2: list[int] | None = None
    if stage2_path:
        stage2_payload = _load_json(stage2_path)
        if only_stage2_questions:
            filtered_by_stage2 = _question_ids_from_stage2(stage2_payload)
            if not filtered_by_stage2:
                raise ValueError(
                    "Stage2 nao retornou question_ids para filtrar alternativas visuais. "
                    "Use --all-questions para ignorar esse filtro."
                )
            allowed = set(filtered_by_stage2)
            question_ids = [question_id for question_id in question_ids if question_id in allowed]

    if limit is not None:
        question_ids = question_ids[:limit]

    if not question_ids:
        raise ValueError("Nenhuma questao com alternativa visual apos filtros")

    output_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    if resume and output_metadata.exists():
        previous = _load_json(output_metadata)
        previous_items = previous.get("items")
        if isinstance(previous_items, list):
            items = [item for item in previous_items if isinstance(item, dict)]

    processed_ids = {
        item.get("question_id")
        for item in items
        if isinstance(item.get("question_id"), int)
    }

    def save_progress() -> None:
        payload = {
            "stage": "stage3_alternatives",
            "model": model,
            "created_at": _now_iso(),
            "source_stage1": str(stage1_path),
            "source_stage2": str(stage2_path) if stage2_path else None,
            "question_images_dir": str(question_images_dir),
            "output_dir": str(output_dir),
            "filtered_question_ids": question_ids,
            "filtered_by_stage2": filtered_by_stage2,
            "items": items,
        }
        _save_json(output_metadata, payload)

    for question_id in question_ids:
        if resume and not per_letter_refine and question_id in processed_ids:
            continue

        input_image = question_images_dir / f"questao_{question_id:02d}.png"
        if not input_image.exists():
            print(
                "Stage3 alternatives questao "
                f"{question_id}: imagem nao encontrada"
            )
            continue

        hint = (
            _stage2_hint_for_question(stage2_payload, question_id)
            if stage2_payload
            else "Sem hint da etapa 2"
        )
        prompt = PROMPT_STAGE3_ALTERNATIVES_TEMPLATE.replace("__STAGE2_HINT__", hint)
        prompt = prompt.replace("__QUESTION_ID__", str(question_id))

        with Image.open(input_image) as image:
            grid_boxes = _grid_bboxes_for_visual_alternatives(image.width, image.height)
            payload = _request_json(
                client=client,
                model=model,
                contents=[prompt, image],
            )

            alternatives_payload = _extract_stage3_alternatives(payload)
            alternatives: dict[str, dict[str, Any]] = {}
            for letter in ("A", "B", "C", "D", "E"):
                alt_meta = alternatives_payload.get(letter)
                if not isinstance(alt_meta, dict):
                    continue

                base_bbox_norm = _expand_bbox_norm(
                    tuple(alt_meta["bbox_norm_0_1000"]),
                    pad_top=8,
                    pad_left=4,
                    pad_bottom=16,
                    pad_right=10,
                )
                base_bbox_px = _bbox_to_pixels(base_bbox_norm, image.width, image.height)
                bbox_px = base_bbox_px

                if per_letter_refine:
                    base_crop = image.crop(bbox_px)
                    refine_prompt = PROMPT_STAGE3_ALTERNATIVE_SINGLE_TEMPLATE.replace(
                        "__LETTER__",
                        letter,
                    )
                    try:
                        refined_payload = _request_json(
                            client=client,
                            model=model,
                            contents=[refine_prompt, base_crop],
                        )
                        refined_local = _extract_stage3_single_bbox(refined_payload)
                    except Exception:
                        refined_local = None

                    if isinstance(refined_local, dict):
                        local_bbox_norm = tuple(refined_local["bbox_norm_0_1000"])
                        local_bbox_px = _bbox_to_pixels(
                            local_bbox_norm,
                            base_crop.width,
                            base_crop.height,
                        )
                        candidate_bbox_px = (
                            bbox_px[0] + local_bbox_px[0],
                            bbox_px[1] + local_bbox_px[1],
                            bbox_px[0] + local_bbox_px[2],
                            bbox_px[1] + local_bbox_px[3],
                        )

                        base_area = max(
                            1,
                            (base_bbox_px[2] - base_bbox_px[0])
                            * (base_bbox_px[3] - base_bbox_px[1]),
                        )
                        candidate_area = max(
                            1,
                            (candidate_bbox_px[2] - candidate_bbox_px[0])
                            * (candidate_bbox_px[3] - candidate_bbox_px[1]),
                        )
                        if candidate_area >= int(base_area * 0.45):
                            bbox_px = candidate_bbox_px

                bbox_px = _expand_bbox_for_dark_edges(
                    image,
                    bbox_px,
                    max_iterations=2,
                    edge_threshold=0.12,
                    step_top=0.10,
                    step_left=0.08,
                    step_bottom=0.08,
                    step_right=0.06,
                )

                grid_bbox = grid_boxes.get(letter)
                if (
                    isinstance(grid_bbox, tuple)
                    and _is_alternative_bbox_suspect(image, bbox_px, grid_bbox)
                ):
                    bbox_px = _grid_fallback_bbox_for_alternative(image, grid_bbox)

                bbox_norm = _bbox_pixels_to_norm(bbox_px, image.width, image.height)
                output_image = output_dir / f"questao_{question_id:02d}_alt_{letter}.png"
                image.crop(bbox_px).save(output_image)

                alternatives[letter] = {
                    "output_image": str(output_image),
                    "bbox_pixels": list(bbox_px),
                    "bbox_norm_0_1000": bbox_norm,
                    "confidence": alt_meta.get("confidence"),
                }

        items = [
            item
            for item in items
            if not (
                isinstance(item, dict)
                and item.get("question_id") == question_id
            )
        ]
        items.append(
            {
                "question_id": question_id,
                "input_image": str(input_image),
                "mode": (
                    "gemini_visual_only_per_letter_refine"
                    if per_letter_refine
                    else "gemini_visual_only"
                ),
                "alternatives": alternatives,
            }
        )
        processed_ids.add(question_id)
        save_progress()
        print(
            "Stage3 alternatives questao "
            f"{question_id}: {len(alternatives)} alternativa(s) visual(is)"
        )

    save_progress()
    print(f"Stage3 alternatives salvo em: {output_metadata}")


def run_stage3_local_alternatives(
    stage1_path: Path,
    question_images_dir: Path,
    output_dir: Path,
    output_metadata: Path,
) -> None:
    stage1_payload = _load_json(stage1_path)
    stage1_data = stage1_payload.get("data")
    if not isinstance(stage1_data, dict):
        raise ValueError("Arquivo stage1 invalido: chave data ausente")

    question_ids = sorted(
        set(
            _extract_int_list(
                stage1_data.get("questions_with_image_or_table_alternatives", [])
            )
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []

    for question_id in question_ids:
        input_image = question_images_dir / f"questao_{question_id:02d}.png"
        if not input_image.exists():
            print(f"Stage3 local alts questao {question_id}: imagem nao encontrada")
            continue

        with Image.open(input_image) as image:
            width, height = image.size
            detected_boxes = _detect_local_visual_alternative_bboxes(image)
            grid_boxes = _grid_bboxes_for_visual_alternatives(width, height)

            alternatives: dict[str, dict[str, Any]] = {}
            for letter in ("A", "B", "C", "D", "E"):
                if letter in detected_boxes:
                    bbox = detected_boxes[letter]
                else:
                    bbox = _grid_fallback_bbox_for_alternative(image, grid_boxes[letter])
                output_image = output_dir / f"questao_{question_id:02d}_alt_{letter}.png"

                image.crop(bbox).save(output_image)

                alternatives[letter] = {
                    "output_image": str(output_image),
                    "bbox_pixels": list(bbox),
                    "bbox_norm_0_1000": _bbox_pixels_to_norm(bbox, width, height),
                }

        items.append(
            {
                "question_id": question_id,
                "input_image": str(input_image),
                "mode": "grid_2x3_lower",
                "alternatives": alternatives,
            }
        )
        print(f"Stage3 local alts questao {question_id}: OK")

    payload = {
        "stage": "stage3_local_alternatives",
        "created_at": _now_iso(),
        "source_stage1": str(stage1_path),
        "source_question_images_dir": str(question_images_dir),
        "output_dir": str(output_dir),
        "items": items,
    }
    _save_json(output_metadata, payload)
    print(f"Stage3 local alts salvo em: {output_metadata}")


def run_stage3_local_illustrations(
    question_images_dir: Path,
    output_dir: Path,
    output_metadata: Path,
    resume: bool,
    limit: int | None,
    max_regions: int,
    stage2_path: Path | None,
    only_stage2_questions: bool,
    stage1_path: Path | None,
) -> None:
    if max_regions <= 0:
        raise ValueError("max_regions deve ser maior que 0")

    stage2_payload: dict[str, Any] = {}
    if stage2_path:
        stage2_payload = _load_json(stage2_path)

    question_ids_with_visual_alternatives: set[int] = set()
    if stage1_path and stage1_path.exists():
        stage1_payload = _load_json(stage1_path)
        stage1_data = stage1_payload.get("data")
        if isinstance(stage1_data, dict):
            question_ids_with_visual_alternatives = set(
                _extract_int_list(
                    stage1_data.get("questions_with_image_or_table_alternatives", [])
                )
            )

    image_paths = sorted(
        image_path
        for image_path in question_images_dir.glob("questao_*.png")
        if _extract_question_number_from_filename(image_path) is not None
    )

    filtered_question_ids: list[int] | None = None
    if stage2_payload and only_stage2_questions:
        filtered_question_ids = _question_ids_from_stage2(stage2_payload)
        if not filtered_question_ids:
            raise ValueError(
                "Stage2 nao retornou question_ids para filtrar stage3 local. "
                "Use --all-questions para ignorar esse filtro."
            )

        allowed = set(filtered_question_ids)
        filtered_image_paths: list[Path] = []
        for image_path in image_paths:
            question_number = _extract_question_number_from_filename(image_path)
            if question_number is not None and question_number in allowed:
                filtered_image_paths.append(image_path)
        image_paths = filtered_image_paths

    if limit is not None:
        image_paths = image_paths[:limit]

    if not image_paths:
        raise ValueError(
            f"Nenhuma imagem de questao encontrada em: {question_images_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    if resume and output_metadata.exists():
        previous = _load_json(output_metadata)
        previous_items = previous.get("items")
        if isinstance(previous_items, list):
            items = [item for item in previous_items if isinstance(item, dict)]

    processed_ids = {
        item.get("question_id")
        for item in items
        if isinstance(item.get("question_id"), int)
    }

    def save_progress() -> None:
        payload = {
            "stage": "stage3_local_illustrations",
            "created_at": _now_iso(),
            "question_images_dir": str(question_images_dir),
            "output_dir": str(output_dir),
            "max_regions": max_regions,
            "source_stage2": str(stage2_path) if stage2_path else None,
            "source_stage1": str(stage1_path) if stage1_path else None,
            "filtered_question_ids": filtered_question_ids,
            "question_ids_with_visual_alternatives": sorted(
                question_ids_with_visual_alternatives
            ),
            "items": items,
        }
        _save_json(output_metadata, payload)

    for image_path in image_paths:
        question_number = _extract_question_number_from_filename(image_path)
        if question_number is None:
            continue
        if question_number in processed_ids:
            continue

        with Image.open(image_path) as image:
            excluded_regions_px: list[tuple[int, int, int, int]] = []
            avoid_top_text_blocks = False

            if question_number in question_ids_with_visual_alternatives:
                alternative_boxes = _grid_bboxes_for_visual_alternatives(
                    image.width,
                    image.height,
                )
                union_left = min(bbox[0] for bbox in alternative_boxes.values())
                union_top = min(bbox[1] for bbox in alternative_boxes.values())
                union_right = max(bbox[2] for bbox in alternative_boxes.values())
                union_bottom = max(bbox[3] for bbox in alternative_boxes.values())
                excluded_regions_px.append(
                    (union_left, union_top, union_right, union_bottom)
                )
                avoid_top_text_blocks = True

            bboxes_px = _detect_local_illustration_bboxes(
                image=image,
                max_regions=max_regions,
                excluded_regions_px=excluded_regions_px,
                avoid_top_text_blocks=avoid_top_text_blocks,
            )

            region_items: list[dict[str, Any]] = []
            for index, bbox_px in enumerate(bboxes_px, start=1):
                box_w = max(1, bbox_px[2] - bbox_px[0])
                box_h = max(1, bbox_px[3] - bbox_px[1])
                bbox_px = _expand_bbox_pixels(
                    bbox_px,
                    image.width,
                    image.height,
                    pad_top=int(box_h * 0.08),
                    pad_left=int(box_w * 0.10),
                    pad_bottom=int(box_h * 0.20),
                    pad_right=int(box_w * 0.08),
                )
                bbox_px = _tighten_illustration_bbox_to_primary_cluster(image, bbox_px)

                if len(bboxes_px) == 1:
                    output_name = f"questao_{question_number:02d}_ilustracao.png"
                else:
                    output_name = (
                        f"questao_{question_number:02d}_ilustracao_{index:02d}.png"
                    )

                output_path = output_dir / output_name
                image.crop(bbox_px).save(output_path)
                region_items.append(
                    {
                        "output_image": str(output_path),
                        "bbox_pixels": list(bbox_px),
                        "bbox_norm_0_1000": _bbox_pixels_to_norm(
                            bbox_px,
                            image.width,
                            image.height,
                        ),
                        "visual_type": "local_detected",
                        "confidence": None,
                        "rank": index,
                    }
                )

        item = {
            "question_id": question_number,
            "input_image": str(image_path),
            "regions": region_items,
            "detection_mode": "local_connected_components",
        }
        if region_items:
            item.update(
                {
                    "output_image": region_items[0]["output_image"],
                    "bbox_norm_0_1000": region_items[0]["bbox_norm_0_1000"],
                    "bbox_pixels": region_items[0]["bbox_pixels"],
                    "visual_type": region_items[0]["visual_type"],
                    "confidence": region_items[0]["confidence"],
                }
            )

        items.append(item)
        processed_ids.add(question_number)
        save_progress()

        if region_items:
            print(
                "Stage3 local ilustracoes questao "
                f"{question_number}: {len(region_items)} ilustracao(oes)"
            )
        else:
            print(f"Stage3 local ilustracoes questao {question_number}: sem ilustracao")

    save_progress()
    print(f"Stage3 local ilustracoes salvo em: {output_metadata}")


def _shared_context_full_texts_from_pdf(
    stage1_payload: dict[str, Any],
) -> dict[str, str]:
    if PdfReader is None:
        return {}

    source_pdf_raw = stage1_payload.get("source_pdf")
    if not isinstance(source_pdf_raw, str) or not source_pdf_raw.strip():
        return {}

    source_pdf = Path(source_pdf_raw)
    if not source_pdf.exists():
        return {}

    stage1_data = stage1_payload.get("data")
    if not isinstance(stage1_data, dict):
        return {}

    shared_contexts = stage1_data.get("shared_contexts")
    if not isinstance(shared_contexts, list):
        return {}

    marker_pattern = re.compile(r"\{\s*0*(\d{1,3})\s*\}")

    def _normalized_text(value: str) -> str:
        return " ".join(value.lower().split())

    def _excerpt_anchors(excerpt: str) -> list[str]:
        parts = [part.strip() for part in excerpt.split("...") if part.strip()]
        anchors: list[str] = []
        for part in parts:
            cleaned = _normalized_text(part)
            if not cleaned:
                continue
            words = cleaned.split(" ")
            anchor_words = words[:8]
            anchor = " ".join(anchor_words).strip()
            if len(anchor) >= 20:
                anchors.append(anchor)
        if not anchors:
            cleaned = _normalized_text(excerpt)
            if len(cleaned) >= 20:
                anchors.append(" ".join(cleaned.split(" ")[:8]))
        return anchors

    def _candidate_matches_excerpt(candidate: str, excerpt: str) -> bool:
        candidate_norm = _normalized_text(candidate)
        excerpt_norm = _normalized_text(excerpt)
        if not candidate_norm or not excerpt_norm:
            return False

        anchors = _excerpt_anchors(excerpt)
        if not anchors:
            return False
        return all(anchor in candidate_norm for anchor in anchors)

    reader = PdfReader(str(source_pdf))
    total_pages = len(reader.pages)
    page_cache: dict[int, str] = {}

    extracted: dict[str, str] = {}
    for item in shared_contexts:
        if not isinstance(item, dict):
            continue

        context_id = item.get("context_id")
        if not isinstance(context_id, str) or not context_id.strip():
            continue

        question_ids = _extract_int_list(item.get("question_ids", []))
        page_numbers = _extract_int_list(item.get("page_numbers", []))
        text_excerpt = str(item.get("text_excerpt", "")).strip()
        if not question_ids or not page_numbers:
            continue

        anchor_qid = min(question_ids)
        context_text: str | None = None

        for page_number in page_numbers:
            if page_number <= 0 or page_number > total_pages:
                continue

            if page_number not in page_cache:
                page_text = reader.pages[page_number - 1].extract_text() or ""
                page_cache[page_number] = _clean_extracted_text(page_text)

            page_text = page_cache[page_number]
            if not page_text:
                continue

            matches = list(marker_pattern.finditer(page_text))
            if not matches:
                continue

            anchor_match: re.Match[str] | None = None
            for match in matches:
                if int(match.group(1)) == anchor_qid:
                    anchor_match = match
                    break

            if anchor_match is None:
                continue

            previous_matches = [
                match
                for match in matches
                if match.start() < anchor_match.start()
                and int(match.group(1)) not in question_ids
            ]
            start_index = previous_matches[-1].end() if previous_matches else 0
            candidate = page_text[start_index:anchor_match.start()]
            candidate = marker_pattern.sub("", candidate)
            candidate = _clean_extracted_text(candidate)
            if text_excerpt and not _candidate_matches_excerpt(candidate, text_excerpt):
                continue
            if candidate:
                context_text = candidate
                break

        if context_text:
            extracted[context_id.strip()] = context_text

    return extracted


def _inject_shared_context_full_text(
    stage1_payload: dict[str, Any],
) -> None:
    stage1_data = stage1_payload.get("data")
    if not isinstance(stage1_data, dict):
        return

    shared_contexts = stage1_data.get("shared_contexts")
    if not isinstance(shared_contexts, list):
        return

    full_texts = _shared_context_full_texts_from_pdf(stage1_payload)
    if not full_texts:
        return

    for item in shared_contexts:
        if not isinstance(item, dict):
            continue
        context_id = item.get("context_id")
        if not isinstance(context_id, str):
            continue
        full_text = full_texts.get(context_id.strip())
        if full_text:
            item["full_text"] = full_text


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _sync_viewer_template(output_dir: Path) -> Path | None:
    root = Path(__file__).resolve().parent
    destination = output_dir / "viewer_prova_estruturada.html"
    template_candidates = [
        root / "viewer_prova_estruturada.html",
        root / "artifacts" / "test14" / "viewer_prova_estruturada.html",
        root / "artifacts" / "latest" / "viewer_prova_estruturada.html",
        destination,
    ]

    template_path: Path | None = None
    for candidate in template_candidates:
        if candidate.exists():
            template_path = candidate
            break

    if template_path is None:
        searched_paths = ", ".join(str(path) for path in template_candidates)
        print(
            "Aviso: template do viewer nao encontrado nos caminhos esperados: "
            f"{searched_paths}. Arquivo HTML nao foi copiado."
        )
        return None

    if template_path.resolve() == destination.resolve():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Viewer HTML copiado para: {destination}")
    return destination


def _write_viewer_data_js(
    merged_output_path: Path,
    merged_payload: dict[str, Any],
    visual_alternatives: dict[str, Any] | None,
    stage2_path: Path | None,
    questions_text_path: Path | None,
) -> None:
    destination_questions_path = merged_output_path.parent / "questoes_texto_local.json"

    candidates: list[Path] = []
    if questions_text_path is not None:
        candidates.append(questions_text_path)
    candidates.append(destination_questions_path)
    if stage2_path is not None:
        candidates.append(stage2_path.parent / "questoes_texto_local.json")

    selected_questions_path: Path | None = None
    for candidate in candidates:
        if candidate.exists():
            selected_questions_path = candidate
            break

    questions_payload: dict[str, Any] | None = None
    if selected_questions_path is not None:
        questions_payload = _load_json(selected_questions_path)
        if selected_questions_path.resolve() != destination_questions_path.resolve():
            _save_json(destination_questions_path, questions_payload)
            print(
                "Texto local copiado para viewer em: "
                f"{destination_questions_path}"
            )

    viewer_data = {
        "structured": merged_payload,
        "questionsText": questions_payload,
        "visualAlternatives": visual_alternatives,
    }

    js_path = merged_output_path.parent / "viewer_data.js"
    js_content = "window.__VIEWER_DATA__ = " + json.dumps(
        viewer_data,
        ensure_ascii=False,
    ) + ";\n"
    js_path.write_text(js_content, encoding="utf-8")
    print(f"Viewer data JS salvo em: {js_path}")


def run_merge(
    stage1_path: Path,
    stage2_path: Path,
    stage3_path: Path,
    output: Path,
    visual_alternatives_path: Path | None,
    questions_text_path: Path | None,
) -> None:
    stage1 = _load_json(stage1_path)
    stage2 = _load_json(stage2_path)
    stage3 = _load_json(stage3_path)
    _inject_shared_context_full_text(stage1)

    resolved_visual_path = visual_alternatives_path
    if resolved_visual_path is None:
        candidate = stage3_path.parent / "alternativas_visuais_local.json"
        if candidate.exists():
            resolved_visual_path = candidate

    visual_alternatives: dict[str, Any] | None = None
    if resolved_visual_path is not None and resolved_visual_path.exists():
        visual_alternatives = _load_json(resolved_visual_path)

    resolved_questions_path = questions_text_path
    if resolved_questions_path is None:
        output_candidate = output.parent / "questoes_texto_local.json"
        if output_candidate.exists():
            resolved_questions_path = output_candidate
    if resolved_questions_path is None:
        stage2_candidate = stage2_path.parent / "questoes_texto_local.json"
        if stage2_candidate.exists():
            resolved_questions_path = stage2_candidate

    questions_text_local: dict[str, Any] | None = None
    if resolved_questions_path is not None and resolved_questions_path.exists():
        questions_text_local = _load_json(resolved_questions_path)

    shared_context_by_question: dict[int, list[dict[str, Any]]] = {}
    for item in stage1.get("data", {}).get("shared_contexts", []):
        if not isinstance(item, dict):
            continue
        for qid in item.get("question_ids", []):
            if isinstance(qid, int):
                shared_context_by_question.setdefault(qid, []).append(item)

    latex_by_question: dict[int, list[dict[str, Any]]] = {}
    for item in stage1.get("data", {}).get("latex_fragments", []):
        if not isinstance(item, dict):
            continue
        for qid in item.get("related_question_ids", []):
            if isinstance(qid, int):
                latex_by_question.setdefault(qid, []).append(item)

    visuals_by_question: dict[int, list[dict[str, Any]]] = {}
    for item in stage3.get("items", []):
        if isinstance(item, dict):
            qid = item.get("question_id")
            if isinstance(qid, int):
                visuals_by_question.setdefault(qid, []).append(item)

    visual_alts_by_question: dict[int, dict[str, Any]] = {}
    if isinstance(visual_alternatives, dict):
        for item in visual_alternatives.get("items", []):
            if not isinstance(item, dict):
                continue
            qid = item.get("question_id")
            if isinstance(qid, int):
                visual_alts_by_question[qid] = item

    all_question_ids = (
        set(shared_context_by_question.keys())
        | set(visuals_by_question.keys())
        | set(latex_by_question.keys())
        | set(visual_alts_by_question.keys())
    )
    structured_questions: list[dict[str, Any]] = []
    for qid in sorted(all_question_ids):
        structured_questions.append(
            {
                "question_id": qid,
                "shared_contexts": shared_context_by_question.get(qid, []),
                "latex_fragments": latex_by_question.get(qid, []),
                "illustration": (
                    visuals_by_question.get(qid, [])[0]
                    if visuals_by_question.get(qid)
                    else None
                ),
                "illustrations": visuals_by_question.get(qid, []),
                "visual_alternatives": visual_alts_by_question.get(qid),
            }
        )

    merged = {
        "created_at": _now_iso(),
        "sources": {
            "stage1": str(stage1_path),
            "stage2": str(stage2_path),
            "stage3": str(stage3_path),
            "questions_text_local": (
                str(resolved_questions_path) if resolved_questions_path else None
            ),
            "visual_alternatives_local": (
                str(resolved_visual_path) if resolved_visual_path else None
            ),
        },
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "questions_text_local": questions_text_local,
        "visual_alternatives_local": visual_alternatives,
        "structured_questions": structured_questions,
    }

    _save_json(output, merged)
    _write_viewer_data_js(
        merged_output_path=output,
        merged_payload=merged,
        visual_alternatives=visual_alternatives,
        stage2_path=stage2_path,
        questions_text_path=resolved_questions_path,
    )
    _sync_viewer_template(output.parent)
    print(f"Merge salvo em: {output}")


def _empty_stage3_payload(
    stage2_path: Path,
    question_images_dir: Path,
    output_dir: Path,
    model: str,
) -> dict[str, Any]:
    return {
        "stage": "stage3",
        "model": model,
        "created_at": _now_iso(),
        "question_images_dir": str(question_images_dir),
        "output_dir": str(output_dir),
        "source_stage2": str(stage2_path),
        "filtered_question_ids": _question_ids_from_stage2(_load_json(stage2_path)),
        "items": [],
    }


def _audit_generated_artifacts(merged_path: Path) -> None:
    try:
        merged = _load_json(merged_path)
    except Exception as exc:
        print(f"[pipeline] Aviso: nao foi possivel auditar artefatos ({exc})")
        return

    stage2_payload = merged.get("stage2")
    if isinstance(stage2_payload, dict):
        stage2_ids = set(_question_ids_from_stage2(stage2_payload))
    else:
        stage2_ids = set()

    questions_text_local = merged.get("questions_text_local")
    text_ids: set[int] = set()
    if isinstance(questions_text_local, dict):
        for item in questions_text_local.get("questions", []):
            if not isinstance(item, dict):
                continue
            qid = item.get("question_id")
            if isinstance(qid, int):
                text_ids.add(qid)

    missing_text_ids = sorted(stage2_ids - text_ids)
    if missing_text_ids:
        print(
            "[pipeline] Auditoria: aviso de cobertura de texto. "
            f"Sem texto local para questoes: {missing_text_ids}"
        )
    elif stage2_ids:
        print(
            "[pipeline] Auditoria: cobertura de texto OK para as questoes do Stage2."
        )

    stage3_payload = merged.get("stage3")
    if not isinstance(stage3_payload, dict):
        return

    question_images_dir = stage3_payload.get("question_images_dir")
    if not isinstance(question_images_dir, str) or not question_images_dir.strip():
        return

    base_norm = question_images_dir.replace("\\", "/").rstrip("/")
    if not base_norm:
        return

    external_inputs: list[str] = []
    for item in stage3_payload.get("items", []):
        if not isinstance(item, dict):
            continue
        input_image = item.get("input_image")
        if not isinstance(input_image, str) or not input_image.strip():
            continue
        input_norm = input_image.replace("\\", "/")
        if not input_norm.startswith(f"{base_norm}/"):
            external_inputs.append(input_image)

    if external_inputs:
        print(
            "[pipeline] Auditoria: possivel mistura de fontes de imagem no Stage3. "
            f"Entradas fora de question_images_dir: {external_inputs}"
        )
    else:
        print("[pipeline] Auditoria: origem de imagens do Stage3 OK.")


def run_pipeline(
    pdf_path: Path,
    artifacts_dir: Path,
    question_images_dir: Path,
    model: str,
    max_pages: int | None,
    pages_override: str | None,
    stage3_limit: int | None,
    skip_stage3: bool,
    local_stage3: bool,
    gemini_visual_alts: bool,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    stage1_path = artifacts_dir / "stage1_basico.json"
    stage2_path = artifacts_dir / "stage2_page_plan.json"
    stage3_output_dir = artifacts_dir / "ilustracoes"
    stage3_path = artifacts_dir / "stage3_ilustracoes.json"
    visual_alts_dir = artifacts_dir / "alternativas_visuais"
    visual_alts_path = artifacts_dir / "alternativas_visuais_local.json"
    questions_text_path = artifacts_dir / "questoes_texto_local.json"
    merged_path = artifacts_dir / "prova_estruturada.json"

    print("[pipeline] Stage1...")
    run_stage1(
        pdf_path=pdf_path,
        output=stage1_path,
        model=model,
        max_pages=max_pages,
    )

    print("[pipeline] Stage2...")
    run_stage2(
        pdf_path=pdf_path,
        stage1_path=stage1_path,
        output=stage2_path,
        model=model,
        pages_override=pages_override,
        dpi=220,
    )

    effective_question_images_dir = question_images_dir
    auto_question_images_dir = artifacts_dir / "input_questoes_stage2"
    print("[pipeline] Gerando imagens de questao a partir do PDF + Stage2...")
    try:
        generated_count = _build_question_images_from_stage2_segments(
            pdf_path=pdf_path,
            stage2_path=stage2_path,
            output_dir=auto_question_images_dir,
            dpi=220,
        )
        if generated_count > 0:
            effective_question_images_dir = auto_question_images_dir
            print(
                "[pipeline] Imagens locais de questao geradas: "
                f"{generated_count} (origem: {effective_question_images_dir})"
            )
        else:
            print(
                "[pipeline] Aviso: nenhuma imagem de questao foi gerada via Stage2. "
                f"Usando question-images-dir informado: {question_images_dir}"
            )
    except Exception as exc:
        print(
            "[pipeline] Aviso: falha ao gerar imagens de questao via Stage2 "
            f"({exc}). Usando question-images-dir informado: {question_images_dir}"
        )

    print("[pipeline] Texto local...")
    try:
        run_questions_text_local(
            pdf_path=pdf_path,
            stage2_path=stage2_path,
            output=questions_text_path,
            stage1_path=stage1_path,
        )
    except Exception as exc:
        print(
            "[pipeline] Aviso: falha na extracao local de texto "
            f"({exc}). O viewer pode mostrar placeholders."
        )

    if local_stage3:
        print("[pipeline] Stage3 local illustrations (sem API)...")
        run_stage3_local_illustrations(
            question_images_dir=effective_question_images_dir,
            output_dir=stage3_output_dir,
            output_metadata=stage3_path,
            resume=False,
            limit=stage3_limit,
            max_regions=3,
            stage2_path=stage2_path,
            only_stage2_questions=True,
            stage1_path=stage1_path,
        )
    elif skip_stage3:
        print("[pipeline] Stage3 pulado por --skip-stage3")
        _save_json(
            stage3_path,
            _empty_stage3_payload(
                stage2_path=stage2_path,
                    question_images_dir=effective_question_images_dir,
                output_dir=stage3_output_dir,
                model=model,
            ),
        )
    else:
        print("[pipeline] Stage3...")
        try:
            run_stage3(
                question_images_dir=effective_question_images_dir,
                output_dir=stage3_output_dir,
                output_metadata=stage3_path,
                model=model,
                stage2_path=stage2_path,
                only_stage2_questions=True,
                resume=False,
                limit=stage3_limit,
            )
        except Exception as exc:
            print(
                "[pipeline] Aviso: stage3 via Gemini falhou "
                f"({exc}). Tentando fallback local de ilustracoes."
            )
            try:
                run_stage3_local_illustrations(
                    question_images_dir=effective_question_images_dir,
                    output_dir=stage3_output_dir,
                    output_metadata=stage3_path,
                    resume=False,
                    limit=stage3_limit,
                    max_regions=3,
                    stage2_path=stage2_path,
                    only_stage2_questions=True,
                    stage1_path=stage1_path,
                )
            except Exception as local_exc:
                if not stage3_path.exists():
                    _save_json(
                        stage3_path,
                        _empty_stage3_payload(
                            stage2_path=stage2_path,
                            question_images_dir=effective_question_images_dir,
                            output_dir=stage3_output_dir,
                            model=model,
                        ),
                    )
                print(
                    "[pipeline] Aviso: fallback local de stage3 tambem falhou "
                    f"({local_exc}). Seguindo com merge parcial."
                )

    if gemini_visual_alts:
        print("[pipeline] Stage3 alternatives (Gemini, visual-only)...")
        try:
            run_stage3_alternatives(
                stage1_path=stage1_path,
                question_images_dir=effective_question_images_dir,
                output_dir=visual_alts_dir,
                output_metadata=visual_alts_path,
                model=model,
                stage2_path=stage2_path,
                only_stage2_questions=True,
                resume=False,
                limit=stage3_limit,
                per_letter_refine=True,
            )
        except Exception as exc:
            print(
                "[pipeline] Aviso: stage3 alternatives via Gemini falhou "
                f"({exc}). Usando fallback local."
            )
            run_stage3_local_alternatives(
                stage1_path=stage1_path,
                question_images_dir=effective_question_images_dir,
                output_dir=visual_alts_dir,
                output_metadata=visual_alts_path,
            )
    else:
        print("[pipeline] Stage3 local alternatives...")
        run_stage3_local_alternatives(
            stage1_path=stage1_path,
            question_images_dir=effective_question_images_dir,
            output_dir=visual_alts_dir,
            output_metadata=visual_alts_path,
        )

    print("[pipeline] Merge...")
    run_merge(
        stage1_path=stage1_path,
        stage2_path=stage2_path,
        stage3_path=stage3_path,
        output=merged_path,
        visual_alternatives_path=visual_alts_path,
        questions_text_path=questions_text_path,
    )

    _audit_generated_artifacts(merged_path)

    print("[pipeline] Finalizado")
    print(f"[pipeline] JSON final: {merged_path}")
    print(f"[pipeline] Viewer data: {artifacts_dir / 'viewer_data.js'}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prototipo em 3 prompts com Gemini para estruturar prova"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage1 = subparsers.add_parser("stage1", help="Prompt 1: leitura global do PDF")
    stage1.add_argument("pdf_path", help="Caminho do PDF da prova")
    stage1.add_argument(
        "--output",
        default="artifacts/stage1_basico.json",
        help="JSON de saida do stage1",
    )
    stage1.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modelo Gemini (default: {DEFAULT_MODEL})",
    )
    stage1.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limita stage1 para as primeiras N paginas do PDF",
    )

    stage2 = subparsers.add_parser(
        "stage2",
        help="Prompt 2: decidir split por pagina (2 partes ou pagina inteira)",
    )
    stage2.add_argument("pdf_path", help="Caminho do PDF da prova")
    stage2.add_argument(
        "--stage1",
        default="artifacts/stage1_basico.json",
        help="JSON gerado no stage1",
    )
    stage2.add_argument(
        "--output",
        default="artifacts/stage2_page_plan.json",
        help="JSON de saida do stage2",
    )
    stage2.add_argument(
        "--pages",
        default=None,
        help="Override manual de paginas (ex: 7,8,10-12)",
    )
    stage2.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="DPI para render da pagina (default: 220)",
    )
    stage2.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modelo Gemini (default: {DEFAULT_MODEL})",
    )

    stage3 = subparsers.add_parser(
        "stage3",
        help="Prompt 3: crop da ilustracao por imagem de questao",
    )
    stage3.add_argument(
        "--question-images-dir",
        default="banco_imagens_fuvest_final",
        help="Diretorio com imagens questao_XX.png",
    )
    stage3.add_argument(
        "--output-dir",
        default="banco_imagens_fuvest_final/ilustracoes",
        help="Diretorio para salvar os crops",
    )
    stage3.add_argument(
        "--output-metadata",
        default="artifacts/stage3_ilustracoes.json",
        help="JSON de metadados do stage3",
    )
    stage3.add_argument(
        "--stage2",
        default="artifacts/stage2_page_plan.json",
        help="JSON do stage2 (opcional para hints)",
    )
    stage3.add_argument(
        "--no-stage2",
        action="store_true",
        help="Ignora o arquivo do stage2",
    )
    stage3.add_argument(
        "--all-questions",
        action="store_true",
        help="Processa todas as imagens de questao, mesmo com stage2 disponivel",
    )
    stage3.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignora metadados existentes do stage3 e recomeca do zero",
    )
    stage3.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita quantidade de imagens para teste",
    )
    stage3.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modelo Gemini (default: {DEFAULT_MODEL})",
    )

    stage3_local_illustrations = subparsers.add_parser(
        "stage3-local-illustrations",
        help="Extracao local de ilustracoes sem API (suporta multiplas por questao)",
    )
    stage3_local_illustrations.add_argument(
        "--question-images-dir",
        default="banco_imagens_fuvest_final",
        help="Diretorio com imagens questao_XX.png",
    )
    stage3_local_illustrations.add_argument(
        "--output-dir",
        default="artifacts/ilustracoes",
        help="Diretorio para salvar os crops locais de ilustracao",
    )
    stage3_local_illustrations.add_argument(
        "--output-metadata",
        default="artifacts/stage3_ilustracoes.json",
        help="JSON de metadados da extracao local de ilustracoes",
    )
    stage3_local_illustrations.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignora metadados existentes e recomeca do zero",
    )
    stage3_local_illustrations.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita quantidade de imagens para teste",
    )
    stage3_local_illustrations.add_argument(
        "--max-regions",
        type=int,
        default=3,
        help="Numero maximo de ilustracoes por questao",
    )
    stage3_local_illustrations.add_argument(
        "--stage2",
        default="artifacts/stage2_page_plan.json",
        help="JSON do stage2 (opcional para filtrar questoes)",
    )
    stage3_local_illustrations.add_argument(
        "--no-stage2",
        action="store_true",
        help="Ignora o arquivo do stage2",
    )
    stage3_local_illustrations.add_argument(
        "--all-questions",
        action="store_true",
        help="Processa todas as imagens de questao, mesmo com stage2 disponivel",
    )
    stage3_local_illustrations.add_argument(
        "--stage1",
        default="artifacts/stage1_basico.json",
        help=(
            "JSON do stage1 (opcional, melhora filtro local para questoes "
            "com alternativas visuais)"
        ),
    )

    stage3_local_alts = subparsers.add_parser(
        "stage3-local-alternatives",
        help="Extracao local de alternativas visuais (A-E) sem API",
    )
    stage3_local_alts.add_argument(
        "--stage1",
        default="artifacts/stage1_basico.json",
        help="JSON do stage1 com questoes de alternativas visuais",
    )
    stage3_local_alts.add_argument(
        "--question-images-dir",
        default="banco_imagens_fuvest_final",
        help="Diretorio com imagens questao_XX.png",
    )
    stage3_local_alts.add_argument(
        "--output-dir",
        default="artifacts/alternativas_visuais",
        help="Diretorio para salvar crops das alternativas",
    )
    stage3_local_alts.add_argument(
        "--output-metadata",
        default="artifacts/alternativas_visuais_local.json",
        help="JSON de metadados de alternativas visuais",
    )

    stage3_alternatives = subparsers.add_parser(
        "stage3-alternatives",
        help="Extracao de alternativas visuais (A-E) via Gemini, sem tocar alternativas textuais",
    )
    stage3_alternatives.add_argument(
        "--stage1",
        default="artifacts/stage1_basico.json",
        help="JSON do stage1 com questoes de alternativas visuais",
    )
    stage3_alternatives.add_argument(
        "--question-images-dir",
        default="banco_imagens_fuvest_final",
        help="Diretorio com imagens questao_XX.png",
    )
    stage3_alternatives.add_argument(
        "--output-dir",
        default="artifacts/alternativas_visuais",
        help="Diretorio para salvar crops das alternativas visuais",
    )
    stage3_alternatives.add_argument(
        "--output-metadata",
        default="artifacts/alternativas_visuais_local.json",
        help="JSON de metadados de alternativas visuais",
    )
    stage3_alternatives.add_argument(
        "--stage2",
        default="artifacts/stage2_page_plan.json",
        help="JSON do stage2 (opcional para filtrar questoes)",
    )
    stage3_alternatives.add_argument(
        "--no-stage2",
        action="store_true",
        help="Ignora o arquivo do stage2",
    )
    stage3_alternatives.add_argument(
        "--all-questions",
        action="store_true",
        help="Processa todas as questoes visuais do stage1, mesmo com stage2 disponivel",
    )
    stage3_alternatives.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignora metadados existentes e recomeca do zero",
    )
    stage3_alternatives.add_argument(
        "--no-per-letter-refine",
        action="store_true",
        help="Desativa refinamento por alternativa (um prompt adicional por letra)",
    )
    stage3_alternatives.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita quantidade de questoes para teste",
    )
    stage3_alternatives.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modelo Gemini (default: {DEFAULT_MODEL})",
    )

    merge = subparsers.add_parser(
        "merge",
        help="Junta metadados dos 3 stages em um JSON final",
    )
    merge.add_argument("--stage1", default="artifacts/stage1_basico.json")
    merge.add_argument("--stage2", default="artifacts/stage2_page_plan.json")
    merge.add_argument("--stage3", default="artifacts/stage3_ilustracoes.json")
    merge.add_argument(
        "--output",
        default="artifacts/prova_estruturada.json",
        help="JSON final consolidado",
    )
    merge.add_argument(
        "--visual-alternatives",
        default=None,
        help=(
            "JSON de alternativas visuais locais (opcional). "
            "Se omitido, tenta automaticamente stage3_dir/alternativas_visuais_local.json"
        ),
    )
    merge.add_argument(
        "--questions-text",
        default=None,
        help=(
            "JSON de texto local das questoes (opcional). "
            "Se omitido, tenta output_dir/questoes_texto_local.json e stage2_dir/questoes_texto_local.json"
        ),
    )

    run_all = subparsers.add_parser(
        "run",
        help="Executa o pipeline completo com defaults (stage1->stage2->stage3->alts->merge)",
    )
    run_all.add_argument("pdf_path", help="Caminho do PDF da prova")
    run_all.add_argument(
        "--artifacts-dir",
        default="artifacts/latest",
        help="Diretorio de saida consolidado (default: artifacts/latest)",
    )
    run_all.add_argument(
        "--question-images-dir",
        default="banco_imagens_fuvest_final",
        help="Diretorio com imagens questao_XX.png",
    )
    run_all.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limita stage1 para as primeiras N paginas do PDF",
    )
    run_all.add_argument(
        "--pages",
        default=None,
        help="Override manual das paginas do stage2 (ex: 1-7)",
    )
    run_all.add_argument(
        "--stage3-limit",
        type=int,
        default=None,
        help="Limita quantidade de questoes no stage3 (teste rapido)",
    )
    run_all.add_argument(
        "--skip-stage3",
        action="store_true",
        help="Pula stage3 (API) e segue com merge usando apenas dados locais",
    )
    run_all.add_argument(
        "--local-stage3",
        action="store_true",
        help="Usa detector local de ilustracoes no lugar do stage3 via API",
    )
    run_all.add_argument(
        "--gemini-visual-alts",
        action="store_true",
        default=True,
        help="Usa Gemini para alternativas visuais (default; mantido por compatibilidade)",
    )
    run_all.add_argument(
        "--local-visual-alts",
        action="store_true",
        help="Usa detector local para alternativas visuais em vez do Gemini",
    )
    run_all.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modelo Gemini (default: {DEFAULT_MODEL})",
    )

    return parser


def main() -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "stage1":
            run_stage1(
                pdf_path=Path(args.pdf_path).expanduser().resolve(),
                output=Path(args.output).expanduser().resolve(),
                model=args.model,
                max_pages=args.max_pages,
            )
            return 0

        if args.command == "stage2":
            run_stage2(
                pdf_path=Path(args.pdf_path).expanduser().resolve(),
                stage1_path=Path(args.stage1).expanduser().resolve(),
                output=Path(args.output).expanduser().resolve(),
                model=args.model,
                pages_override=args.pages,
                dpi=args.dpi,
            )
            return 0

        if args.command == "stage3":
            stage2_path = None if args.no_stage2 else Path(args.stage2).expanduser().resolve()
            run_stage3(
                question_images_dir=Path(args.question_images_dir).expanduser().resolve(),
                output_dir=Path(args.output_dir).expanduser().resolve(),
                output_metadata=Path(args.output_metadata).expanduser().resolve(),
                model=args.model,
                stage2_path=stage2_path,
                only_stage2_questions=not args.all_questions,
                resume=not args.no_resume,
                limit=args.limit,
            )
            return 0

        if args.command == "stage3-local-illustrations":
            stage2_path = None if args.no_stage2 else Path(args.stage2).expanduser().resolve()
            stage1_path = Path(args.stage1).expanduser().resolve()
            run_stage3_local_illustrations(
                question_images_dir=Path(args.question_images_dir).expanduser().resolve(),
                output_dir=Path(args.output_dir).expanduser().resolve(),
                output_metadata=Path(args.output_metadata).expanduser().resolve(),
                resume=not args.no_resume,
                limit=args.limit,
                max_regions=args.max_regions,
                stage2_path=stage2_path,
                only_stage2_questions=not args.all_questions,
                stage1_path=(stage1_path if stage1_path.exists() else None),
            )
            return 0

        if args.command == "stage3-local-alternatives":
            run_stage3_local_alternatives(
                stage1_path=Path(args.stage1).expanduser().resolve(),
                question_images_dir=Path(args.question_images_dir).expanduser().resolve(),
                output_dir=Path(args.output_dir).expanduser().resolve(),
                output_metadata=Path(args.output_metadata).expanduser().resolve(),
            )
            return 0

        if args.command == "stage3-alternatives":
            stage2_path = None if args.no_stage2 else Path(args.stage2).expanduser().resolve()
            run_stage3_alternatives(
                stage1_path=Path(args.stage1).expanduser().resolve(),
                question_images_dir=Path(args.question_images_dir).expanduser().resolve(),
                output_dir=Path(args.output_dir).expanduser().resolve(),
                output_metadata=Path(args.output_metadata).expanduser().resolve(),
                model=args.model,
                stage2_path=stage2_path,
                only_stage2_questions=not args.all_questions,
                resume=not args.no_resume,
                limit=args.limit,
                per_letter_refine=not args.no_per_letter_refine,
            )
            return 0

        if args.command == "merge":
            run_merge(
                stage1_path=Path(args.stage1).expanduser().resolve(),
                stage2_path=Path(args.stage2).expanduser().resolve(),
                stage3_path=Path(args.stage3).expanduser().resolve(),
                output=Path(args.output).expanduser().resolve(),
                visual_alternatives_path=(
                    Path(args.visual_alternatives).expanduser().resolve()
                    if args.visual_alternatives
                    else None
                ),
                questions_text_path=(
                    Path(args.questions_text).expanduser().resolve()
                    if args.questions_text
                    else None
                ),
            )
            return 0

        if args.command == "run":
            run_pipeline(
                pdf_path=Path(args.pdf_path).expanduser().resolve(),
                artifacts_dir=Path(args.artifacts_dir).expanduser().resolve(),
                question_images_dir=Path(args.question_images_dir).expanduser().resolve(),
                model=args.model,
                max_pages=args.max_pages,
                pages_override=args.pages,
                stage3_limit=args.stage3_limit,
                skip_stage3=args.skip_stage3,
                local_stage3=args.local_stage3,
                gemini_visual_alts=(
                    args.gemini_visual_alts and not args.local_visual_alts
                ),
            )
            return 0

        raise RuntimeError(f"Comando nao suportado: {args.command}")
    except Exception as exc:
        print(f"Erro: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
