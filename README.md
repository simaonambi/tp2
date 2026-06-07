# TP2 — Retail Vision Intelligence System

LIACD · Universidade da Beira Interior · 2025/2026
Simão · Nº 53558

> **Para avaliação/defesa:** ver [`DEFENSE.md`](DEFENSE.md) com setup rápido,
> demonstração em 3 comandos e formato dos ficheiros de teste.

## Visão Geral

Sistema de inspecção contínua de prateleiras com:
- análise visual por Google Gemini (3 estratégias de prompting: zero-shot,
  chain-of-thought, few-shot)
- motor de regras em linguagem natural com detecção de ambiguidades
- memória semântica (RAG) com ChromaDB e 2 estratégias de chunking
- gerador de relatórios e interface conversacional CLI

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# editar .env e colar a chave Gemini
```

Obter chave gratuita em https://aistudio.google.com → "Get API Key".

## Estrutura

```
tp2/
├── README.md
├── requirements.txt
├── .env.example
├── evaluate.py
├── data/
│   ├── images/                imagens de prateleiras
│   ├── inspections/           registos JSON gerados
│   └── rules/                  regras persistidas
├── src/
│   ├── shelf_inspector.py
│   ├── rule_engine.py
│   ├── rag_memory.py
│   ├── report_generator.py
│   └── interface.py
├── prompts/                    prompts versionados
├── vectorstore/                ChromaDB persistente
└── cache/                      cache MD5 de inspecções
```

## Utilização

### Inspecção de uma imagem

```bash
python3 src/shelf_inspector.py --image data/images/foto1.jpg --zone Z_S3 --strategy chain_of_thought
```

### Interface conversacional

```bash
python3 src/interface.py
```

Comandos disponíveis dentro da CLI:

```
inspect Z_S3 --image data/images/foto.jpg
inspect-dir data/images/ --zone Z_S1
add rule "Avisa-me quando a prateleira inferior estiver mais de 30% vazia"
list rules
delete rule RULE_ABC123
test rule data/inspections/INS_xxx.json
history "Quando foi a ultima vez que Z_S1 esteve vazia?"
search "problemas de planograma"
reindex
report --session 2026-05-13
help
exit
```

### Avaliação

```bash
python3 evaluate.py --images-dir data/images/ --output evaluation_report.json
```

Requer três ficheiros em `data/`:
- `ground_truth.json` — issues anotados para cada imagem
- `rag_queries.json` — perguntas com IDs relevantes esperados
- `rule_cases.json` — regras com inspecções sintéticas para testar matching

## Modelo Utilizado

`gemini-1.5-flash` via API gratuita do Google AI Studio (1500 req/dia, 15 req/min).
Embeddings com `paraphrase-multilingual-MiniLM-L12-v2` (local, suporta português).
Vector store: ChromaDB persistente em disco.

## Reprodutibilidade

Todas as chamadas LLM usam `temperature=0`. O Gemini não expõe seed, pelo que
outputs podem variar ligeiramente entre execuções. Resultados são cacheados por
hash MD5 da imagem + estratégia, de forma a evitar consumo de quota e garantir
reprodutibilidade dentro da mesma sessão.

## Notas de Licença

Imagens utilizadas no desenvolvimento provêm de datasets públicos
(SKU-110K, Grocery Store Dataset, Open Images). A origem e licença de cada
conjunto está documentada no relatório técnico.
