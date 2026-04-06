from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task
from pydantic import ValidationError

from src.config import AppConfig
from src.ingestion import pdf_to_markdown
from src.models import Questao
from src.utils import extract_json_payload, run_with_retry


class ExtracaoQuestoesPipeline:
    """Pipeline ponta-a-ponta: PDF -> Markdown -> CrewAI -> lista de Questoes."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig.from_env()

    async def run(
        self,
        pdf_path: str | Path,
        checkpoint_path: str | Path | None = None,
        resume: bool = True,
    ) -> list[Questao]:
        resolved_pdf_path = Path(pdf_path).expanduser().resolve()
        resolved_checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path is not None
            else None
        )

        checkpoint_state = self._initialize_checkpoint_state(
            checkpoint_path=resolved_checkpoint_path,
            pdf_path=resolved_pdf_path,
            resume=resume,
        )

        markdown = checkpoint_state.get("markdown")
        if not isinstance(markdown, str) or not markdown.strip():
            print("Etapa 1/5: convertendo PDF para markdown...")
            markdown = await asyncio.to_thread(
                pdf_to_markdown,
                resolved_pdf_path,
                self.config,
            )
            checkpoint_state["markdown"] = markdown
            self._save_checkpoint_state(
                checkpoint_path=resolved_checkpoint_path,
                state=checkpoint_state,
                pdf_path=resolved_pdf_path,
            )
        else:
            print("Checkpoint: markdown reutilizado.")

        agents = self._build_agents(review_model=self.config.review_model)

        map_payload = checkpoint_state.get("map_payload")
        if not isinstance(map_payload, dict):
            print("Etapa 2/5: executando Agente Mapeador...")
            map_payload = await asyncio.to_thread(
                self._run_mapper_stage,
                agents["mapper"],
                markdown,
            )
            checkpoint_state["map_payload"] = map_payload
            self._save_checkpoint_state(
                checkpoint_path=resolved_checkpoint_path,
                state=checkpoint_state,
                pdf_path=resolved_pdf_path,
            )
        else:
            print("Checkpoint: etapa Mapeador reutilizada.")

        extract_payload = checkpoint_state.get("extract_payload")
        if not isinstance(extract_payload, dict):
            print("Etapa 3/5: executando Agente Extrator...")
            extract_payload = await asyncio.to_thread(
                self._run_extractor_stage,
                agents["extractor"],
                markdown,
                map_payload,
            )
            checkpoint_state["extract_payload"] = extract_payload
            self._save_checkpoint_state(
                checkpoint_path=resolved_checkpoint_path,
                state=checkpoint_state,
                pdf_path=resolved_pdf_path,
            )
        else:
            print("Checkpoint: etapa Extrator reutilizada.")

        vision_payload = checkpoint_state.get("vision_payload")
        if not isinstance(vision_payload, dict):
            print("Etapa 4/5: executando Agente Visionario...")
            vision_payload = await asyncio.to_thread(
                self._run_visionary_stage,
                agents["visionary"],
                extract_payload,
            )
            checkpoint_state["vision_payload"] = vision_payload
            self._save_checkpoint_state(
                checkpoint_path=resolved_checkpoint_path,
                state=checkpoint_state,
                pdf_path=resolved_pdf_path,
            )
        else:
            print("Checkpoint: etapa Visionario reutilizada.")

        review_payload = checkpoint_state.get("review_payload")
        if not isinstance(review_payload, dict):
            print("Etapa 5/5: executando Agente Revisor...")
            review_payload = await asyncio.to_thread(
                self._run_reviewer_stage_with_fallback,
                markdown,
                vision_payload,
            )
            checkpoint_state["review_payload"] = review_payload
            self._save_checkpoint_state(
                checkpoint_path=resolved_checkpoint_path,
                state=checkpoint_state,
                pdf_path=resolved_pdf_path,
            )
        else:
            print("Checkpoint: etapa Revisor reutilizada.")

        validated_questions = self._parse_and_validate_output(review_payload)
        checkpoint_state["final_questions"] = [
            question.model_dump(mode="json") for question in validated_questions
        ]
        self._save_checkpoint_state(
            checkpoint_path=resolved_checkpoint_path,
            state=checkpoint_state,
            pdf_path=resolved_pdf_path,
        )
        return validated_questions

    def _initialize_checkpoint_state(
        self,
        checkpoint_path: Path | None,
        pdf_path: Path,
        resume: bool,
    ) -> dict[str, Any]:
        if checkpoint_path is None:
            return {}

        if not checkpoint_path.exists() or not resume:
            if checkpoint_path.exists() and not resume:
                print("Checkpoint ignorado por --no-resume; iniciando do zero.")
            return {}

        try:
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Checkpoint corrompido: {checkpoint_path}. "
                "Use --no-resume para reconstruir."
            ) from error

        if not isinstance(loaded, dict):
            raise ValueError(
                f"Checkpoint invalido: {checkpoint_path}. "
                "Use --no-resume para reconstruir."
            )

        stored_pdf_path = loaded.get("pdf_path")
        if isinstance(stored_pdf_path, str) and stored_pdf_path.strip():
            resolved_stored_pdf = Path(stored_pdf_path).expanduser().resolve()
            if resolved_stored_pdf != pdf_path:
                raise ValueError(
                    "Checkpoint pertence a outro PDF. "
                    "Use --checkpoint com outro arquivo ou --no-resume."
                )

        print(f"Checkpoint carregado: {checkpoint_path}")
        return loaded

    @staticmethod
    def _save_checkpoint_state(
        checkpoint_path: Path | None,
        state: dict[str, Any],
        pdf_path: Path,
    ) -> None:
        if checkpoint_path is None:
            return

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state["pdf_path"] = str(pdf_path)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()

        temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(checkpoint_path)

    def _run_mapper_stage(self, agent: Agent, markdown: str) -> dict[str, Any]:
        raw_output = self._run_single_task_crew(
            agent=agent,
            description=(
                "Analise o markdown da prova e mapeie a hierarquia completa.\n"
                "Regras:\n"
                "1) Identifique todas as questoes.\n"
                "2) Associe textos de apoio ao id da questao correta.\n"
                "3) Aponte referencias a imagens e tabelas por questao.\n"
                "4) Nao invente conteudo.\n"
                "Entrada markdown:\n"
                "{markdown_prova}\n"
                "Saida obrigatoria (somente JSON): objeto com chave mapa_questoes.\n"
                "Cada item de mapa_questoes deve conter: id, texto_apoio_vinculado, "
                "referencias_imagem e referencias_tabela.\n"
            ),
            expected_output="JSON valido com mapa_questoes.",
            inputs={"markdown_prova": markdown},
        )

        payload = extract_json_payload(raw_output)
        return self._normalize_map_payload(payload)

    def _run_extractor_stage(
        self,
        agent: Agent,
        markdown: str,
        map_payload: dict[str, Any],
    ) -> dict[str, Any]:
        raw_output = self._run_single_task_crew(
            agent=agent,
            description=(
                "Com base no markdown original e no mapa de questoes, extraia cada "
                "questao completa.\n"
                "Regras:\n"
                "1) Gere enunciado sem cortes.\n"
                "2) Gere alternativas A, B, C, D e E obrigatoriamente.\n"
                "3) Se houver equacoes, formate em LaTeX.\n"
                "4) Mantenha texto_apoio_vinculado do mapeamento.\n"
                "5) Nao invente metadados ausentes (use null).\n"
                "Entrada markdown:\n"
                "{markdown_prova}\n"
                "Mapa de questoes (JSON):\n"
                "{mapa_questoes_json}\n"
                "Saida obrigatoria (somente JSON): objeto com chave questoes.\n"
                "Cada item de questoes deve conter: id, enunciado, alternativas com "
                "chaves A-E, texto_apoio_vinculado, tags_assunto e metadados_banca.\n"
            ),
            expected_output="JSON valido com questoes extraidas.",
            inputs={
                "markdown_prova": markdown,
                "mapa_questoes_json": json.dumps(
                    map_payload,
                    ensure_ascii=False,
                ),
            },
        )

        payload = extract_json_payload(raw_output)
        return self._normalize_questions_payload(payload)

    def _run_visionary_stage(
        self,
        agent: Agent,
        extract_payload: dict[str, Any],
    ) -> dict[str, Any]:
        raw_output = self._run_single_task_crew(
            agent=agent,
            description=(
                "Revise o JSON de questoes e enriqueca conteudo visual.\n"
                "Regras:\n"
                "1) Descreva imagens referenciadas de forma objetiva.\n"
                "2) Converta tabelas complexas para markdown ou html limpo.\n"
                "3) Nao altere significado do enunciado/alternativas.\n"
                "4) Preserve o schema final das questoes.\n"
                "JSON de questoes extraidas:\n"
                "{questoes_extraidas_json}\n"
                "Saida obrigatoria: somente JSON no mesmo formato de questoes.\n"
            ),
            expected_output="JSON valido com questoes enriquecidas visualmente.",
            inputs={
                "questoes_extraidas_json": json.dumps(
                    extract_payload,
                    ensure_ascii=False,
                )
            },
        )

        payload = extract_json_payload(raw_output)
        return self._normalize_questions_payload(payload)

    def _run_reviewer_stage_with_fallback(
        self,
        markdown: str,
        vision_payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return self._run_reviewer_stage(
                review_model=self.config.review_model,
                markdown=markdown,
                vision_payload=vision_payload,
            )
        except Exception as primary_error:
            fallback_model = self.config.review_fallback_model
            if not fallback_model or fallback_model == self.config.review_model:
                raise

            print(
                "Falha no modelo de revisao primario; tentando fallback: "
                f"{fallback_model}"
            )

            try:
                return self._run_reviewer_stage(
                    review_model=fallback_model,
                    markdown=markdown,
                    vision_payload=vision_payload,
                )
            except Exception as fallback_error:
                raise RuntimeError(
                    "Falha na revisao com modelo primario e fallback. "
                    f"Primario ({self.config.review_model}): {primary_error}. "
                    f"Fallback ({fallback_model}): {fallback_error}."
                ) from fallback_error

    def _run_reviewer_stage(
        self,
        review_model: str,
        markdown: str,
        vision_payload: dict[str, Any],
    ) -> dict[str, Any]:
        reviewer = self._build_reviewer_agent(review_model=review_model)

        raw_output = self._run_single_task_crew(
            agent=reviewer,
            description=(
                "Audite o JSON final comparando com markdown original.\n"
                "Checklist:\n"
                "1) Sem alucinacao.\n"
                "2) Sem perda de texto relevante.\n"
                "3) Schema final estritamente compativel com o modelo Questao.\n"
                "4) Caso algo esteja duvidoso, prefira texto literal do documento.\n"
                "Markdown original:\n"
                "{markdown_prova}\n"
                "JSON revisado visualmente:\n"
                "{questoes_enriquecidas_json}\n"
                "Retorne SOMENTE o JSON final, sem explicacoes.\n"
            ),
            expected_output="JSON final auditado no formato de lista de questoes.",
            inputs={
                "markdown_prova": markdown,
                "questoes_enriquecidas_json": json.dumps(
                    vision_payload,
                    ensure_ascii=False,
                ),
            },
        )

        payload = extract_json_payload(raw_output)
        return self._normalize_questions_payload(payload)

    def _run_single_task_crew(
        self,
        agent: Agent,
        description: str,
        expected_output: str,
        inputs: dict[str, Any],
    ) -> str:
        task = Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

        output = run_with_retry(
            crew.kickoff,
            inputs=inputs,
            max_retries=self.config.max_retries,
            base_delay_seconds=self.config.base_retry_seconds,
            max_delay_seconds=self.config.max_retry_seconds,
        )

        if hasattr(output, "raw"):
            return str(output.raw)

        return str(output)

    def _build_agents(self, review_model: str) -> dict[str, Agent]:
        mapper = Agent(
            role="Agente Mapeador",
            goal=(
                "Mapear a hierarquia da prova e identificar quais textos de apoio "
                "pertencem a cada questao."
            ),
            backstory=(
                "Especialista em analise estrutural de provas com foco em contexto, "
                "blocos e relacionamento entre secoes."
            ),
            llm=self.config.extraction_model,
            allow_delegation=False,
            max_iter=1,
            verbose=False,
        )

        extractor = Agent(
            role="Agente Extrator",
            goal=(
                "Extrair questoes completas em blocos consistentes com enunciado, "
                "alternativas A-E e indicacoes de imagens/tabelas."
            ),
            backstory=(
                "Especialista em parser de avaliacoes, focado em nao truncar texto "
                "e em padronizar formato de saida."
            ),
            llm=self.config.extraction_model,
            allow_delegation=False,
            max_iter=1,
            verbose=False,
        )

        visionary = Agent(
            role="Agente Visionario",
            goal=(
                "Descrever imagens e normalizar tabelas complexas em markdown/html "
                "limpo para preservar conteudo sem ambiguidades."
            ),
            backstory=(
                "Especialista em interpretacao visual e limpeza de estrutura tabular "
                "em provas digitalizadas."
            ),
            llm=self.config.extraction_model,
            allow_delegation=False,
            max_iter=1,
            verbose=False,
        )

        return {
            "mapper": mapper,
            "extractor": extractor,
            "visionary": visionary,
        }

    @staticmethod
    def _build_reviewer_agent(review_model: str) -> Agent:
        return Agent(
            role="Agente Revisor",
            goal=(
                "Auditar o JSON final contra o markdown original e remover qualquer "
                "alucinacao, omissao ou texto inventado."
            ),
            backstory=(
                "Auditor rigoroso de qualidade de dados, com foco em fidelidade ao "
                "documento original."
            ),
            llm=review_model,
            allow_delegation=False,
            max_iter=1,
            verbose=False,
        )

    def _parse_and_validate_output(self, raw_output: Any) -> list[Questao]:
        payload = (
            extract_json_payload(raw_output)
            if isinstance(raw_output, str)
            else raw_output
        )
        items = self._extract_question_items(payload)

        validated_questions: list[Questao] = []
        invalid_items: list[tuple[int, ValidationError]] = []

        for idx, item in enumerate(items, start=1):
            try:
                normalized_item = self._normalize_item(item)
                validated_questions.append(Questao.model_validate(normalized_item))
            except ValidationError as error:
                invalid_items.append((idx, error))

        if not validated_questions:
            if invalid_items:
                first_idx, first_error = invalid_items[0]
                raise ValueError(
                    f"Nenhuma questao valida foi produzida. "
                    f"Primeiro erro na posicao {first_idx}: {first_error}"
                ) from first_error
            raise ValueError("Nenhuma questao valida foi produzida pela pipeline")

        if invalid_items:
            print(
                "Aviso: "
                f"{len(invalid_items)} questao(oes) invalida(s) foram descartadas "
                "na validacao final."
            )

        return validated_questions

    @staticmethod
    def _normalize_map_payload(payload: Any) -> dict[str, Any]:
        if isinstance(payload, list):
            return {"mapa_questoes": payload}

        if isinstance(payload, dict):
            for key in ("mapa_questoes", "mapaQuestoes", "question_map", "map"):
                value = payload.get(key)
                if isinstance(value, list):
                    return {"mapa_questoes": value}

        raise ValueError(
            "JSON de mapeamento invalido. Esperado objeto com chave mapa_questoes."
        )

    @staticmethod
    def _normalize_questions_payload(payload: Any) -> dict[str, Any]:
        if isinstance(payload, list):
            return {"questoes": payload}

        if isinstance(payload, dict):
            for key in ("questoes", "questions", "itens", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return {"questoes": value}

        raise ValueError(
            "JSON de questoes invalido. Esperado lista ou objeto com chave questoes."
        )

    @staticmethod
    def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(item)

        if isinstance(normalized.get("texto_apoio_vinculado"), str):
            normalized["texto_apoio_vinculado"] = [
                normalized["texto_apoio_vinculado"]
            ]

        if normalized.get("tags_assunto") is None:
            normalized["tags_assunto"] = []

        if normalized.get("metadados_banca") is None:
            normalized["metadados_banca"] = {}

        return normalized

    @staticmethod
    def _extract_question_items(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict):
            for key in ("questoes", "questions", "itens", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value

        raise ValueError(
            "JSON final invalido. Esperado lista de questoes ou objeto com chave questoes."
        )
