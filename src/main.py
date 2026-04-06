from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agents import ExtracaoQuestoesPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extracao de questoes de PDF com arquitetura multi-agentes"
    )
    parser.add_argument("pdf_path", help="Caminho para o arquivo .pdf")
    parser.add_argument(
        "--output",
        help="Caminho do arquivo JSON de saida (default: mesmo nome do PDF)",
        default=None,
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "Arquivo de checkpoint para salvar/retomar etapas "
            "(default: <output>.checkpoint.json)"
        ),
        default=None,
    )
    parser.add_argument(
        "--no-resume",
        help="Ignora checkpoint existente e recomeca do zero.",
        action="store_true",
    )
    parser.add_argument(
        "--no-checkpoint",
        help="Desativa salvamento de checkpoint durante a execucao.",
        action="store_true",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(args.pdf_path).expanduser().resolve().with_suffix(".json")
    )

    checkpoint_path: Path | None
    if args.no_checkpoint:
        checkpoint_path = None
    elif args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    else:
        checkpoint_path = output_path.with_suffix(".checkpoint.json")

    pipeline = ExtracaoQuestoesPipeline()
    questoes = await pipeline.run(
        args.pdf_path,
        checkpoint_path=checkpoint_path,
        resume=not args.no_resume,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [questao.model_dump(mode="json") for questao in questoes]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Arquivo JSON exportado em: {output_path}")
    print(f"Total de questoes extraidas: {len(payload)}")
    if checkpoint_path is not None:
        print(f"Checkpoint atualizado em: {checkpoint_path}")
    return 0


def main() -> int:
    load_dotenv()

    parser = _build_parser()
    args = parser.parse_args()

    try:
        return asyncio.run(_run(args))
    except Exception as error:
        print(f"Falha na execucao: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
