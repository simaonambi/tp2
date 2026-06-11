"""
rag_memory.py
Indexacao e recuperacao semantica de inspecoes usando ChromaDB e
sentence-transformers (modelo multilingue). Suporta queries em linguagem
natural traduzidas em pesquisa vetorial + filtros de metadata.
"""

import os
# Silenciar telemetria do ChromaDB antes de importar o modulo
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]      = "False"

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY         = os.getenv("GEMINI_API_KEY")
MODEL_NAME      = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")
INSPECTIONS_DIR = Path(os.getenv("INSPECTIONS_DIR", "data/inspections"))
EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNKING        = os.getenv("CHUNKING_STRATEGY", "hybrid")   # "hybrid" ou "per_issue"

if API_KEY:
    genai.configure(api_key=API_KEY)


def collection_name_for(strategy: str = None) -> str:
    """Devolve o nome de coleccao por estrategia (isolam-se em coleccoes separadas)."""
    s = strategy or CHUNKING
    return "inspections_hybrid" if s == "hybrid" else "inspections_per_issue"


def get_client():
    """ChromaDB com persistencia em disco e telemetria desligada."""
    return chromadb.PersistentClient(
        path=VECTORSTORE_DIR,
        settings=Settings(anonymized_telemetry=False)
    )


def get_embedder():
    """Funcao de embedding baseada em sentence-transformers."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )


def get_collection(name: str = None):
    """Devolve a coleccao da estrategia activa (ou de uma estrategia especifica)."""
    if name is None:
        name = collection_name_for()
    return get_client().get_or_create_collection(
        name=name,
        embedding_function=get_embedder()
    )


def build_summary(record: dict) -> str:
    """Constroi um summary textual rico para indexacao.
    Privilegia termos uteis para retrieval futuro (zona, dia, fill rate, tipos)."""
    zone   = record.get("zone_id", "zona desconhecida")
    status = record.get("overall_status", "?")
    fill   = record.get("shelf_fill_rate", 0.0)
    ts     = record.get("timestamp", "")

    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        day_name = ["segunda","terca","quarta","quinta","sexta","sabado","domingo"][dt.weekday()]
        hour     = dt.hour
        date_str = dt.strftime("%Y-%m-%d")
    except Exception:
        day_name, hour, date_str = "dia desconhecido", -1, ts[:10] if ts else "data desconhecida"

    issues = record.get("issues", [])
    if issues:
        issue_parts = []
        for iss in issues:
            t = iss.get("type", "?")
            sev = iss.get("severity", "?")
            loc = iss.get("location", "?")
            desc = iss.get("description", "")
            # Converter campos potencialmente nao-string em strings seguras
            t = str(t) if not isinstance(t, str) else t
            sev = str(sev) if not isinstance(sev, str) else sev
            loc = str(loc) if not isinstance(loc, str) else loc
            desc = str(desc) if not isinstance(desc, str) else desc
            issue_parts.append(f"{t} ({sev}) em {loc}: {desc[:120]}")
        issues_text = " Problemas detectados: " + "; ".join(issue_parts) + "."
    else:
        issues_text = " Sem problemas detectados."

    products = record.get("products_detected", [])
    # Robustecer: ignorar entradas nao-string ou converte-las
    safe_products = [p if isinstance(p, str) else str(p) for p in products if p]
    products_text = f" Produtos visiveis: {', '.join(safe_products)}." if safe_products else ""

    return (
        f"Inspeccao da zona {zone} em {date_str} ({day_name} as {hour}h). "
        f"Estado geral: {status}. Taxa de preenchimento: {fill:.0%}."
        f"{issues_text}{products_text}"
    )


def build_metadata(record: dict) -> dict:
    """Metadata estruturada para filtragem pre-retrieval."""
    ts = record.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        weekday = dt.weekday()
        hour    = dt.hour
        date    = dt.strftime("%Y-%m-%d")
    except Exception:
        weekday, hour, date = -1, -1, ""

    # Garantir que so apanhamos types que sao strings nao vazias.
    # Em casos de falha de schema, o Gemini pode devolver type como dict ou lista;
    # convertemos tudo para string segura antes do join.
    issue_types = []
    for iss in record.get("issues", []):
        t = iss.get("type", "")
        if isinstance(t, str) and t:
            issue_types.append(t)
        elif t:
            issue_types.append(str(t))
    issue_types = list(set(issue_types))

    return {
        "inspection_id":  record.get("inspection_id", ""),
        "zone_id":        record.get("zone_id", ""),
        "overall_status": record.get("overall_status", ""),
        "shelf_fill_rate": float(record.get("shelf_fill_rate", 0.0)),
        "issue_count":    len(record.get("issues", [])),
        "issue_types":    ",".join(issue_types),
        "timestamp":      ts,
        "date":           date,
        "weekday":        weekday,
        "hour":           hour
    }


def build_issue_chunk(record: dict, issue: dict, issue_idx: int) -> tuple:
    """Constroi chunk semantico para UM issue. Retorna (id, texto, metadata).
    Usado pela estrategia 'per_issue'."""
    zone   = record.get("zone_id", "?")
    ts     = record.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        day_name = ["segunda","terca","quarta","quinta","sexta","sabado","domingo"][dt.weekday()]
        date_str = dt.strftime("%Y-%m-%d")
        weekday = dt.weekday()
        hour = dt.hour
    except Exception:
        day_name, date_str, weekday, hour = "?", ts[:10] if ts else "", -1, -1

    text = (
        f"Problema de tipo {issue.get('type','?')} com severidade {issue.get('severity','?')} "
        f"detectado na zona {zone} em {date_str} ({day_name}). "
        f"Localizacao: {issue.get('location','nao especificado')}. "
        f"Descricao: {issue.get('description','')[:160]}"
    )
    chunk_id = f"{record.get('inspection_id')}_iss{issue_idx:02d}"
    metadata = {
        "parent_inspection_id": record.get("inspection_id", ""),
        "zone_id":      zone,
        "issue_type":   issue.get("type", ""),
        "severity":     issue.get("severity", ""),
        "timestamp":    ts,
        "date":         date_str,
        "weekday":      weekday,
        "hour":         hour,
    }
    return chunk_id, text, metadata


def index_inspection(record: dict, strategy: str = None) -> bool:
    """Indexa uma inspecao. Idempotente por inspection_id.
    Suporta duas estrategias de chunking:
      - 'hybrid' (default): summary unico com metadata estruturada
      - 'per_issue': cada issue indexado separadamente
    """
    s = strategy or CHUNKING
    iid = record.get("inspection_id")
    if not iid:
        return False

    if s == "per_issue":
        col = get_collection(collection_name_for("per_issue"))
        # apagar chunks antigos desta inspecao
        existing = col.get(where={"parent_inspection_id": iid})
        if existing.get("ids"):
            col.delete(ids=existing["ids"])

        issues = record.get("issues", [])
        if not issues:
            # sem issues - guarda chunk de cobertura para inspeccoes 'ok'
            summary = build_summary(record)
            metadata = build_metadata(record)
            metadata["parent_inspection_id"] = iid
            col.add(documents=[summary], metadatas=[metadata], ids=[f"{iid}_ok"])
            return True

        ids, docs, metas = [], [], []
        for i, iss in enumerate(issues):
            cid, text, meta = build_issue_chunk(record, iss, i)
            ids.append(cid)
            docs.append(text)
            metas.append(meta)
        col.add(documents=docs, metadatas=metas, ids=ids)
        return True

    # estrategia 'hybrid' (default)
    col = get_collection(collection_name_for("hybrid"))
    existing = col.get(ids=[iid])
    if existing.get("ids"):
        col.delete(ids=[iid])

    summary  = build_summary(record)
    metadata = build_metadata(record)
    col.add(documents=[summary], metadatas=[metadata], ids=[iid])
    return True


def index_all_from_disk(strategy: str = None) -> int:
    """Re-indexa todas as inspeccoes existentes em disco com a estrategia indicada."""
    count = 0
    for f in sorted(INSPECTIONS_DIR.glob("INS_*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            if index_inspection(rec, strategy=strategy):
                count += 1
        except (json.JSONDecodeError, KeyError):
            continue
    return count


def search(query: str, k: int = 3, where: dict = None, strategy: str = None) -> list:
    """Pesquisa semantica na coleccao da estrategia indicada (ou default).
    Para 'per_issue' os resultados sao agrupados por parent_inspection_id
    para devolver no maximo k inspeccoes distintas."""
    s = strategy or CHUNKING
    col = get_collection(collection_name_for(s))

    # pedir mais resultados para per_issue (varios chunks por inspeccao)
    n_results = k * 4 if s == "per_issue" else k
    kwargs = {"query_texts": [query], "n_results": n_results}
    if where:
        kwargs["where"] = where

    res = col.query(**kwargs)
    results = []
    if not res.get("ids") or not res["ids"][0]:
        return results

    if s == "per_issue":
        # deduplicar por parent_inspection_id, manter a primeira ocorrencia (mais relevante)
        seen = set()
        for i, cid in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i] if res.get("metadatas") else {}
            parent = meta.get("parent_inspection_id", cid)
            if parent in seen:
                continue
            seen.add(parent)
            results.append({
                "inspection_id": parent,
                "chunk_id":      cid,
                "summary":       res["documents"][0][i],
                "metadata":      meta,
                "distance":      res["distances"][0][i] if res.get("distances") else None
            })
            if len(results) >= k:
                break
        return results

    # hybrid
    for i, iid in enumerate(res["ids"][0]):
        results.append({
            "inspection_id": iid,
            "summary":       res["documents"][0][i],
            "metadata":      res["metadatas"][0][i] if res.get("metadatas") else {},
            "distance":      res["distances"][0][i] if res.get("distances") else None
        })
    return results


def compare_chunking_strategies(queries: list, k: int = 3) -> dict:
    """Compara as duas estrategias de chunking sobre um conjunto de queries.
    Retorna metricas Recall@k por estrategia.
    Cada query e' um dict com 'query' e 'relevant_ids' (lista de inspection_ids)."""
    results = {}
    for strategy in ("hybrid", "per_issue"):
        n_indexed = index_all_from_disk(strategy=strategy)
        # contar chunks reais na coleccao (per_issue tem multiplos por inspeccao)
        col = get_collection(collection_name_for(strategy))
        n_chunks = col.count()

        recalls = []
        for q in queries:
            relevant = set(q.get("relevant_ids", []))
            if not relevant:
                continue
            hits = search(q["query"], k=k, strategy=strategy)
            hit_ids = {h["inspection_id"] for h in hits}
            recalls.append(1.0 if (relevant & hit_ids) else 0.0)

        results[strategy] = {
            "inspections":      n_indexed,
            "chunks_total":     n_chunks,
            "queries":          len(recalls),
            "recall_at_k_pct":  round(sum(recalls) / len(recalls) * 100, 1) if recalls else None
        }
    return results


RAG_ANSWER_PROMPT = """Es um analista de operacoes de retalho. Responde a pergunta do gestor com base apenas no contexto fornecido.

REGRAS:
- Usa apenas factos presentes no contexto.
- Cita as inspeccoes relevantes pelo seu inspection_id e data.
- Se o contexto nao for suficiente, diz isso claramente.
- Responde em portugues europeu, conciso, sem repeticoes.

PERGUNTA DO GESTOR:
{query}

CONTEXTO (inspecoes mais relevantes):
{context}

Responde em prosa curta com referencias explicitas."""


def answer_with_rag(query: str, k: int = 3) -> dict:
    """Responde a uma pergunta usando RAG sobre as inspecoes indexadas."""
    hits = search(query, k=k)
    if not hits:
        return {
            "query":   query,
            "answer":  "Nao ha inspeccoes indexadas relevantes para responder.",
            "sources": []
        }

    context = "\n\n".join(
        f"[{h['inspection_id']} | {h['metadata'].get('date','')} | zona {h['metadata'].get('zone_id','')}]\n"
        f"{h['summary']}"
        for h in hits
    )

    prompt = RAG_ANSWER_PROMPT.format(query=query, context=context)

    if not API_KEY:
        return {
            "query":   query,
            "answer":  "Sem API key configurada. Resultados retrieval abaixo:",
            "sources": hits
        }

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2, "max_output_tokens": 800}
    )

    return {
        "query":   query,
        "answer":  response.text or "",
        "sources": hits
    }


def main():
    ap = argparse.ArgumentParser(description="Memoria RAG das inspecoes")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_re = sub.add_parser("reindex", help="Re-indexar todas as inspecoes em disco")
    p_re.add_argument("--strategy", choices=["hybrid", "per_issue"], default=None)

    p_q = sub.add_parser("query", help="Pesquisa semantica")
    p_q.add_argument("text")
    p_q.add_argument("--k", type=int, default=3)
    p_q.add_argument("--strategy", choices=["hybrid", "per_issue"], default=None)

    p_a = sub.add_parser("ask", help="Pergunta em linguagem natural (RAG completo)")
    p_a.add_argument("text")
    p_a.add_argument("--k", type=int, default=3)

    p_c = sub.add_parser("compare", help="Compara as duas estrategias de chunking")
    p_c.add_argument("--queries", default="data/rag_queries.json",
                      help="Ficheiro JSON com queries e relevant_ids")
    p_c.add_argument("--k", type=int, default=3)

    args = ap.parse_args()

    if args.cmd == "reindex":
        n = index_all_from_disk(strategy=args.strategy)
        s = args.strategy or CHUNKING
        print(f"{n} inspecoes (re)indexadas com estrategia '{s}'.")

    elif args.cmd == "query":
        for h in search(args.text, k=args.k, strategy=args.strategy):
            print(f"[{h['inspection_id']}] dist={h['distance']:.3f}")
            print(f"  {h['summary'][:200]}")

    elif args.cmd == "ask":
        res = answer_with_rag(args.text, k=args.k)
        print(f"Resposta:\n{res['answer']}\n")
        print(f"Fontes ({len(res['sources'])}):")
        for s in res["sources"]:
            print(f"  - {s['inspection_id']} ({s['metadata'].get('date','')})")

    elif args.cmd == "compare":
        qpath = Path(args.queries)
        if not qpath.exists():
            print(f"Ficheiro de queries nao encontrado: {qpath}")
            return
        queries = json.loads(qpath.read_text(encoding="utf-8"))
        print(f"\nA comparar estrategias sobre {len(queries)} queries...")
        results = compare_chunking_strategies(queries, k=args.k)

        print("\n" + "=" * 65)
        print(f"  COMPARACAO DE ESTRATEGIAS DE CHUNKING (k={args.k})")
        print("=" * 65)
        for strat, m in results.items():
            recall = m.get("recall_at_k_pct")
            recall_str = f"{recall}%" if recall is not None else "N/A (sem ground truth)"
            print(f"  {strat:<12}  inspeccoes={m['inspections']:>2}  "
                   f"chunks={m['chunks_total']:>3}  "
                   f"queries={m['queries']:>2}  Recall@{args.k}={recall_str}")
        print("=" * 65)


if __name__ == "__main__":
    main()