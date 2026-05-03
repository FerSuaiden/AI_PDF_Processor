# MVP - Extracao de Questoes em PDF (Multi-Agentes)

Este projeto implementa um MVP em Python para extrair questoes de provas em PDF com uma arquitetura de multi-agentes usando CrewAI.

## Agentes implementados

1. Agente Mapeador
   - Mapeia a hierarquia e vincula textos de apoio as questoes.
2. Agente Extrator
   - Extrai enunciado, alternativas A-E, referencias visuais e equacoes em LaTeX.
3. Agente Visionario
   - Enriquece descricoes de imagens e normaliza tabelas complexas.
4. Agente Revisor
   - Audita o JSON final para evitar alucinacoes e cortes de texto.

## Estrutura

- src/models/questao.py: contrato de saida com Pydantic.
- src/ingestion/pdf_to_markdown.py: conversao PDF -> Markdown via Docling ou LlamaParse.
- src/agents/crew_pipeline.py: orquestracao dos agentes com CrewAI.
- src/main.py: script principal para execucao por CLI.

## Requisitos

- Python 3.11+
- Dependencias em requirements.txt
- Variaveis de ambiente configuradas (use .env.example como base)

## Instalacao

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuracao

1. Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

2. Preencha as chaves de API conforme os modelos/providers escolhidos.

Observacao: o pipeline usa REVIEW_MODEL como primario e REVIEW_FALLBACK_MODEL como contingencia para o Agente Revisor.
Para PDF textual (nao escaneado), mantenha DOCLING_DO_OCR=false para execucao mais rapida.

## Execucao

```bash
python -m src.main /caminho/arquivo.pdf --output /caminho/saida.json
```

Se --output for omitido, o JSON sera salvo com o mesmo nome do PDF.

O pipeline agora salva checkpoint por etapa (markdown, mapeamento, extracao, visao e revisao).
Se a API gratuita estourar cota, basta executar novamente o mesmo comando quando a cota recarregar e a execucao retomara da ultima etapa concluida.

Opcoes uteis:

```bash
# Definir checkpoint customizado
python -m src.main prova.pdf --output saida.json --checkpoint estado.json

# Ignorar checkpoint e recomecar do zero
python -m src.main prova.pdf --output saida.json --no-resume

# Desativar checkpoint
python -m src.main prova.pdf --output saida.json --no-checkpoint
```

## Testes

Execute os testes de contrato do schema e utilitarios:

```bash
pytest -q
```

## Pipeline simplificado (1 comando)

Para evitar rodar varios subcomandos com muitos parametros, use:

```bash
python gemini_prova_3_prompts.py run fuvest2026-fase1-prova-V1.pdf
```

Esse comando executa stage1, stage2, stage3, alternativas visuais locais e merge com defaults,
salvando em `artifacts/latest/`.

Arquivos finais importantes em `artifacts/latest/`:

- `prova_estruturada.json`
- `viewer_data.js`
- `viewer_prova_estruturada.html`

Abra `viewer_prova_estruturada.html`: o carregamento e automatico (sem selecionar arquivos).

Opcoes uteis do comando `run`:

```bash
# Limitar leitura inicial para teste
python gemini_prova_3_prompts.py run fuvest2026-fase1-prova-V1.pdf --max-pages 7

# Pular stage3 da API (quando cota estiver estourada)
python gemini_prova_3_prompts.py run fuvest2026-fase1-prova-V1.pdf --skip-stage3

# Usar detector local de ilustracoes (sem gastar creditos Gemini)
python gemini_prova_3_prompts.py run fuvest2026-fase1-prova-V1.pdf --local-stage3
```

## Stage3 local (sem API)

Para extrair ilustracoes sem Gemini, inclusive com suporte a mais de uma ilustracao por questao:

```bash
python gemini_prova_3_prompts.py stage3-local-illustrations \
   --question-images-dir banco_imagens_fuvest_final \
   --output-dir artifacts/ilustracoes \
   --output-metadata artifacts/stage3_ilustracoes.json
```

Observacoes:

- O script considera apenas arquivos no formato `questao_XX.png` como entrada base.
- Arquivos como `questao_14_ilustracao.png` nao entram como entrada do stage3.

## Scripts em not used

Os scripts em `not used/` sao experimentais e nao fazem parte do fluxo principal atual.
Isso inclui `not used/aaaa.py` e `not used/bbbbb.py`.
