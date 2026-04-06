#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image


PROMPT = """
Voce recebera uma imagem de uma questao de vestibular.

Objetivo:
- Encontrar somente a area da ilustracao principal (grafico, tabela, desenho, foto, mapa etc).
- Incluir a linha de fonte/legenda quando ela estiver visualmente colada na ilustracao.
- Excluir completamente enunciado, texto da questao e alternativas (A-E).

Retorne APENAS um JSON valido neste formato:
{"bbox": [ymin, xmin, ymax, xmax]}

Regras do bbox:
- Coordenadas normalizadas entre 0 e 1000.
- Use inteiros.
- Garanta ymin < ymax e xmin < xmax.
- Gere uma unica caixa que cubra toda a ilustracao relevante.
""".strip()


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned)
    cleaned = re.sub(r"\\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Resposta do Gemini nao contem JSON valido: {text}")
        return json.loads(match.group(0))


def _validate_bbox(payload: dict[str, Any]) -> tuple[int, int, int, int]:
    if "bbox" not in payload:
        raise ValueError(f"JSON sem chave 'bbox': {payload}")

    bbox = payload["bbox"]
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


def _to_pixels(
    bbox: tuple[int, int, int, int], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = bbox

    left = int((xmin / 1000) * image_width)
    top = int((ymin / 1000) * image_height)
    right = int((xmax / 1000) * image_width)
    bottom = int((ymax / 1000) * image_height)

    left = max(0, min(left, image_width - 1))
    right = max(left + 1, min(right, image_width))
    top = max(0, min(top, image_height - 1))
    bottom = max(top + 1, min(bottom, image_height))

    return left, top, right, bottom


def crop_illustration_with_gemini(
    input_path: Path, output_path: Path, model_name: str
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY nao encontrada no ambiente/.env")

    client = genai.Client(api_key=api_key)

    with Image.open(input_path) as image:
        width, height = image.size

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[PROMPT, image],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
        except genai_errors.ClientError as exc:
            message = str(exc)
            if "404 NOT_FOUND" in message:
                raise RuntimeError(
                    "Modelo Gemini nao encontrado para generateContent. "
                    "Use, por exemplo, --model gemini-2.5-flash ou --model gemini-2.0-flash."
                ) from exc
            if "429 RESOURCE_EXHAUSTED" in message:
                raise RuntimeError(
                    "Cota da Gemini excedida para esta chave/projeto. "
                    "Verifique faturamento/limites e tente novamente depois."
                ) from exc
            raise

        payload = _extract_json(response.text)
        bbox_norm = _validate_bbox(payload)
        bbox_px = _to_pixels(bbox_norm, width, height)

        cropped = image.crop(bbox_px)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)

    return bbox_norm, bbox_px


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Usa Gemini para recortar apenas a ilustracao da questao 14"
    )
    parser.add_argument(
        "--input",
        default="banco_imagens_fuvest_final/questao_14.png",
        help="Imagem da questao (default: banco_imagens_fuvest_final/questao_14.png)",
    )
    parser.add_argument(
        "--output",
        default="banco_imagens_fuvest_final/questao_14_ilustracao.png",
        help="Caminho do crop final (default: banco_imagens_fuvest_final/questao_14_ilustracao.png)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Modelo Gemini (default: gemini-2.5-flash)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Arquivo de entrada nao encontrado: {input_path}")

    try:
        bbox_norm, bbox_px = crop_illustration_with_gemini(
            input_path=input_path,
            output_path=output_path,
            model_name=args.model,
        )
    except Exception as exc:
        raise SystemExit(f"Erro: {exc}") from exc

    print("Crop concluido com sucesso.")
    print(f"Entrada: {input_path}")
    print(f"Saida: {output_path}")
    print(f"BBox normalizado (0-1000): {bbox_norm}")
    print(f"BBox em pixels: {bbox_px}")


if __name__ == "__main__":
    main()