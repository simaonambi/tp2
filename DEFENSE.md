# Guia de Avaliação e Defesa — TP2

Documento dirigido ao docente para facilitar a avaliação do sistema e a defesa oral.

## 1. Setup Rápido

```bash
# 1. Criar ambiente virtual e instalar dependências
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configurar chave Gemini do avaliador
cp .env.example .env
# editar .env e colocar GEMINI_API_KEY=...
# (a chave do aluno NÃO está incluída — use a sua chave gratuita do
#  Google AI Studio em https://aistudio.google.com)
```

## 2. Demonstração Rápida (3 comandos)

Para verificar que o sistema está funcional:

```bash
# A. Inspecionar uma imagem qualquer
python3 src/shelf_inspector.py --image data/images/_pool/IMG_6584.heic \
    --zone Z_S1 --strategy chain_of_thought

# B. Comparar estratégias de chunking do RAG (não consome quota API)
python3 src/rag_memory.py compare --queries data/rag_queries.json --k 1

# C. Abrir a interface conversacional
python3 src/interface.py
# Comandos sugeridos dentro da CLI:
#   inspect Z_S1 --image data/images/_pool/IMG_6584.heic
#   add rule "Avisa quando a prateleira inferior estiver mais de 30% vazia"
#   list rules
#   history "Que problemas houve em Z_S1?"
#   report
#   exit
```

## 3. Avaliação Automatizada com as Suas Imagens

### Opção A — Avaliar apenas a análise visual (mais simples)

Coloque as 10 imagens de teste numa pasta e corra:

```bash
python3 evaluate.py --images-dir <sua_pasta_teste>/ \
    --output evaluation_report.json --strategies chain_of_thought
```

Sem `ground_truth.json`, o harness corre na mesma e mede a **taxa de
parseamento JSON**. Os ficheiros de inspecção são gravados em
`data/inspections/INS_*.json` para inspeção manual.

### Opção B — Avaliação completa (com o seu ground truth)

Crie um ficheiro `gt_avaliacao.json` no formato:

```json
[
  {
    "image": "imagem_01.jpg",
    "zone": "Z_S1",
    "issues": [
      {"type": "empty_shelf", "severity": "high"},
      {"type": "misaligned", "severity": "low"}
    ]
  },
  {
    "image": "imagem_02.jpg",
    "zone": "Z_S3",
    "issues": []
  }
]
```

**Tipos válidos:** `empty_shelf`, `wrong_product`, `damaged`, `misaligned`,
`label_missing`, `other`
**Severidades válidas:** `low`, `medium`, `high`

Depois execute:

```bash
python3 evaluate.py --images-dir <sua_pasta_teste>/ \
    --ground-truth gt_avaliacao.json \
    --output evaluation_report.json \
    --strategies chain_of_thought
```

### Opção C — Avaliar as 3 estratégias

Remova `--strategies chain_of_thought` para correr também `zero_shot` e
`few_shot`. **Atenção:** triplica o número de chamadas à API.

## 4. Limites de Quota da API Gemini

A versão usada (`gemini-2.5-flash-lite`) tem, no tier gratuito da conta
desenvolvedora:

- 15 req/min (rate limit por minuto)
- 1000 req/dia (mas contas novas podem ter apenas 20/dia)

O sistema implementa **cache automático** por hash MD5 das imagens, pelo
que execuções repetidas com as mesmas imagens não consomem quota
adicional. A cache fica em `cache/`.

Se o limite for atingido durante a avaliação, o sistema regista a falha
graciosamente e continua para os passos seguintes — o `evaluation_report.json`
é escrito mesmo com resultados parciais.

## 5. Estrutura dos Ficheiros JSON

### `ground_truth.json` — anotação das imagens
```json
[{"image": "nome.jpg", "zone": "Z_S1", "issues": [...]}]
```

### `rag_queries.json` — perguntas para o RAG
```json
[{"query": "Que zonas com problemas?", "relevant_ids": ["INS_..."]}]
```

### `rule_cases.json` — regras com casos de teste
```json
[{
  "text": "Avisa-me quando a prateleira estiver vazia",
  "should_be_ambiguous": true,
  "test_inspection": {...inspeção sintética...},
  "should_match": true
}]
```

Os ficheiros existentes em `data/` são exemplos do aluno e podem ser
substituídos pelos seus.

## 6. Notas para a Defesa Oral

**Decisões técnicas principais que estou preparado para defender:**

- **Modelo**: `gemini-2.5-flash-lite` (3 estratégias de prompting implementadas)
- **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (suporta PT)
- **Vector store**: ChromaDB persistente em disco
- **Chunking**: comparação entre `hybrid` (1 chunk por inspeção) e
  `per_issue` (cada problema indexado separadamente) — secção 5 do relatório
- **Cache**: MD5 da imagem + estratégia para evitar consumo desnecessário de quota
- **HEIC**: suporte via `pillow-heif` para fotografias de iPhone

**Limitações honestas documentadas no relatório:**

- Variantes sintéticas no dataset podem inflar artificialmente as métricas
- Distribuição de tipos de problema enviesada (mais `empty_shelf` que `damaged`)
- Quota gratuita Gemini limitou o número de chamadas durante avaliação
- O Gemini ocasionalmente retorna tipos fora do schema (documentado como caso de falha)

## 7. Resolução de Problemas

| Sintoma | Causa | Solução |
|---|---|---|
| `cannot identify image file ... heic` | falta `pillow-heif` | `pip install pillow-heif` |
| `429 Quota exceeded` | quota Gemini esgotada | aguardar reset diário ou usar cache |
| `Failed to send telemetry event` | bug do ChromaDB (não crítico) | ignorar, sistema funciona |
| `module not found` | ambiente virtual não ativo | `source .venv/bin/activate` |
