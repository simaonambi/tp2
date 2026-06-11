# Retail Vision Intelligence System

Sistema de inspecção contínua de prateleiras de retalho com LLM multimodal, motor de regras em linguagem natural e memória semântica RAG.

> **Trabalho Prático #2** — Interacção com Modelos de Grande Escala (LIACD)
> Universidade da Beira Interior · Ano Lectivo 2025/2026
> **Autor:** Simão Nambi · N.º 53558

> **Para avaliação:** consultar [`DEFENSE.md`](DEFENSE.md) com setup rápido, demonstração em 3 comandos e formato dos ficheiros de teste.

## Visão Geral

O sistema recebe imagens de prateleiras de supermercado e produz inteligência operacional accionável: detecta problemas (prateleiras vazias, produtos mal posicionados, embalagens danificadas), permite ao gestor definir regras em linguagem natural que são convertidas automaticamente em configurações executáveis, e mantém memória semântica do histórico de inspecções para recuperação contextualizada.

### Cinco componentes obrigatórios

| Módulo | Função |
|---|---|
| `src/shelf_inspector.py` | Análise visual com Google Gemini (3 estratégias de prompting) |
| `src/rule_engine.py` | Conversão de regras NL → JSON com detecção de ambiguidades |
| `src/rag_memory.py` | Memória semântica com ChromaDB (2 estratégias de chunking) |
| `src/report_generator.py` | Relatórios automáticos em Markdown com 6 secções |
| `src/interface.py` | CLI conversacional |
| `src/interface_menu.py` | Alternativa amigável com menu numerado |

## Instalação

```bash
# 1. Ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Dependências
pip install -r requirements.txt

# 3. Configurar chave Gemini
cp .env.example .env
# editar .env e colar GEMINI_API_KEY=...
```

Obter chave gratuita em https://aistudio.google.com → "Get API Key".

## Demonstração rápida

```bash
# Interface por menu numerado (recomendado para demonstração)
python3 src/interface_menu.py

# Interface conversacional (comandos por linha)
python3 src/interface.py

# Inspecção directa de uma imagem
python3 src/shelf_inspector.py \
    --image data/images/_pool/IMG_6584.heic \
    --zone Z_S1 \
    --strategy chain_of_thought

# Comparação de estratégias de chunking (não consome quota)
python3 src/rag_memory.py compare --queries data/rag_queries.json --k 1

# Avaliação completa
python3 evaluate.py --images-dir test_images/ --output evaluation_report.json
```

## Estrutura do projecto

```
tp2/
├── README.md                  este ficheiro
├── DEFENSE.md                 guia para o docente
├── requirements.txt
├── .env.example
├── .gitignore
├── evaluate.py                harness com 11 métricas + LLM-as-judge
├── data/
│   ├── images/                dataset (751 imagens)
│   ├── inspections/           records gerados em runtime
│   ├── rules/                 regras persistidas
│   ├── ground_truth.json      18 anotações manuais
│   ├── rag_queries.json       8 queries com relevant_ids
│   └── rule_cases.json        6 casos de teste
├── src/
│   ├── shelf_inspector.py
│   ├── rule_engine.py
│   ├── rag_memory.py
│   ├── report_generator.py
│   ├── interface.py
│   └── interface_menu.py
├── prompts/                   8 prompts versionados em .txt
│   ├── shelf_zero_shot.txt
│   ├── shelf_chain_of_thought.txt
│   ├── shelf_few_shot.txt
│   ├── rule_engine.txt
│   ├── rag_answer.txt
│   ├── rag_summary.txt
│   ├── judge_faithfulness.txt
│   └── judge_relevance.txt
├── scripts/
│   ├── build_dataset.py       gerador de dataset com suporte HEIC
│   └── annotate.py            anotação interactiva
├── cache/                     cache MD5 (runtime, no .gitignore)
└── vectorstore/               ChromaDB persistente (runtime, no .gitignore)
```

## Funcionalidades

- **Três estratégias de prompting** para análise visual: zero-shot, chain-of-thought, few-shot textual
- **Suporte HEIC nativo** (`pillow-heif`) para fotos de iPhone
- **Cache MD5** automático por hash da imagem + estratégia
- **Backoff exponencial** em respostas HTTP 429 (rate-limit)
- **Fallback gracioso** quando a quota diária esgota
- **Schema JSON** validado com 6 tipos de issue e 3 níveis de severidade
- **Detecção de ambiguidades** em regras escritas em português
- **Duas estratégias de chunking** comparáveis para o RAG (`hybrid` vs `per_issue`)
- **LLM-as-judge** para avaliação qualitativa (faithfulness + answer relevance)
- **Comando `compare`** para agregar métricas entre zonas (`compare Z_S1 Z_S3 --period "last 7 days"`)
- **Filtro `--period`** em relatórios (`report --period "last 14 days"`)
- **Interface CLI conversacional** com comandos `inspect`, `add rule`, `history`, `report`, etc.
- **Interface menu numerada** para demonstração em defesa oral

## Stack técnico

- **LLM multimodal:** Google Gemini 2.5 Flash Lite (API gratuita, 15 req/min)
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (local, suporta PT)
- **Vector store:** ChromaDB persistente em disco (zero-config)
- **Python 3.9+** com `python-dotenv`, `pillow-heif`, `chromadb`, `google-generativeai`

## Dataset

- **751 imagens** distribuídas em 5 categorias (mínimo 500 do enunciado)
- **33 fotografias próprias** tiradas em Continente e Auchan (Covilhã, formato HEIC)
- **2 imagens da Internet** (Unsplash, licença gratuita)
- **716 variantes sintéticas** geradas via transformações Pillow
- **18 anotações manuais** em `data/ground_truth.json` (mínimo 15 do enunciado)

## Resultados (corrida final)

### RAG — comparação de chunking
| Estratégia | Recall@1 (3 insp.) | Recall@1 (18 insp.) |
|---|---|---|
| hybrid | 62,5% | 25,0% |
| per_issue | **75,0%** | **50,0%** |

A vantagem do `per_issue` cresce com o tamanho do índice (12,5 → 25 pontos percentuais).

### Análise visual
- JSON Parse Rate: **93,3%**
- Issue Detection Rate (literal): 0% — devido a não-conformidade ao schema (caso de falha documentado)
- Issue Detection Rate (com normalização de sinónimos): **~43%**

## Limitações

- Quota gratuita Gemini limitou avaliação intensiva das 3 estratégias
- O modelo ocasionalmente retorna tipos fora do schema canónico (caso de falha documentado no relatório)
- Variantes sintéticas podem inflar optimisticamente algumas métricas
- Integração com TP1 (afluência → contexto de stockout) não foi implementada

## Documentos

- [`DEFENSE.md`](DEFENSE.md) — guia para avaliação pelo docente
- `tp2_relatório.pdf` — relatório técnico IEEE de 9 páginas

## Licença

Projecto académico. As fotografias próprias do dataset estão licenciadas sob Creative Commons BY-NC 4.0 para fins académicos. As imagens Unsplash seguem a licença Unsplash. O código é entrega académica individual.
