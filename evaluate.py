"""
evaluate.py
Harness de avaliacao do sistema. Mede:
- analise visual: detection rate, false positive rate, severity accuracy,
                  JSON parse rate, hallucination rate
- RAG: Recall@3, faithfulness, answer relevance (LLM-as-judge)
- Rule engine: parse rate, correctness, ambiguity detection
"""

import argparse
import json
import os
import sys
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))

from shelf_inspector import inspect_image
from rag_memory      import index_inspection, search, answer_with_rag
from rule_engine     import add_rule, rule_matches

load_dotenv()
API_KEY    = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
if API_KEY:
    genai.configure(api_key=API_KEY)


# ───── Analise visual ──────────────────────────────────────────────────────
def eval_visual(images_dir: Path, ground_truth_file: Path,
                 strategy: str = "chain_of_thought") -> dict:
    """Avalia analise visual contra ground truth.

    Comportamento robusto:
    - Se ground_truth_file existir: usa-o e calcula todas as metricas.
    - Se nao existir: corre as imagens da pasta na mesma e mede so JSON parse rate
      (util para o professor que pode nao ter passado o ground-truth).

    Schema do ground_truth.json:
    [
      {"image":"img1.jpg","zone":"Z_S1",
       "issues":[{"type":"empty_shelf","severity":"high"}, ...]},
      ...
    ]
    """
    has_gt = ground_truth_file.exists()

    if has_gt:
        gt = json.loads(ground_truth_file.read_text(encoding="utf-8"))
    else:
        # Sem ground truth: corre todas as imagens da pasta como "ok" implicito
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
        images = sorted([
            f.name for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ])
        gt = [{"image": name, "zone": "Z_UNKNOWN", "issues": []} for name in images]
        print(f"    AVISO: ground truth nao encontrado ({ground_truth_file})")
        print(f"    A correr inspeccao em {len(gt)} imagens, so com JSON parse rate.")

    if not gt:
        return {"error": "Sem imagens para avaliar."}

    tp, fp, fn = 0, 0, 0
    sev_correct, sev_total = 0, 0
    parse_ok, parse_total = 0, 0
    halluc_ok, halluc_total = 0, 0

    per_image = []

    for entry in gt:
        img_path = images_dir / entry["image"]
        if not img_path.exists():
            # Imagem do ground truth nao esta na pasta indicada - ignora a entrada,
            # nao penaliza as percentagens.
            per_image.append({"image": entry["image"], "skipped": True,
                              "reason": "imagem nao esta em --images-dir"})
            continue

        parse_total += 1
        try:
            rec = inspect_image(img_path, strategy=strategy,
                                 zone_id=entry.get("zone", "Z_UNKNOWN"))
        except Exception as e:
            per_image.append({"image": entry["image"], "error": str(e)[:120]})
            # falha de processamento conta como JSON parse falhado mas nao como
            # imagem avaliada
            continue

        if rec.get("json_parse_ok"):
            parse_ok += 1

        predicted = rec.get("issues", [])
        gt_issues = entry.get("issues", [])

        pred_types = [i.get("type") for i in predicted]
        gt_types   = [i.get("type") for i in gt_issues]

        for t in gt_types:
            if t in pred_types:
                tp += 1
            else:
                fn += 1
        for t in pred_types:
            if t not in gt_types:
                fp += 1

        gt_by_type = {i.get("type"): i.get("severity") for i in gt_issues}
        for p in predicted:
            if p.get("type") in gt_by_type:
                sev_total += 1
                if p.get("severity") == gt_by_type[p.get("type")]:
                    sev_correct += 1

        for p in predicted:
            halluc_total += 1
            if p.get("type") in gt_types:
                halluc_ok += 1

        per_image.append({
            "image":     entry["image"],
            "predicted": pred_types,
            "expected":  gt_types,
            "tp_local":  len(set(pred_types) & set(gt_types)),
            "fp_local":  len(set(pred_types) - set(gt_types)),
            "fn_local":  len(set(gt_types)  - set(pred_types))
        })

    total_gt = tp + fn
    result = {
        "strategy": strategy,
        "ground_truth_available": has_gt,
        "images_evaluated":   len([p for p in per_image if not p.get("skipped")]),
        "json_parse_rate_pct":     round(parse_ok / parse_total * 100, 1) if parse_total else 0,
    }
    if has_gt:
        result.update({
            "issue_detection_rate_pct": round(tp / total_gt * 100, 1) if total_gt else 0,
            "false_positive_rate_pct": round(fp / (fp + tp) * 100, 1) if (fp + tp) else 0,
            "severity_accuracy_pct":   round(sev_correct / sev_total * 100, 1) if sev_total else 0,
            "non_hallucination_pct":   round(halluc_ok / halluc_total * 100, 1) if halluc_total else 0,
        })
    result["per_image"] = per_image
    return result


# ───── Avaliacao do RAG ────────────────────────────────────────────────────
def eval_rag(queries_file: Path) -> dict:
    """Avalia retrieval e qualidade da resposta RAG.

    Se o ficheiro nao existir, devolve secao com 'skipped' = True (nao falha).

    Schema do queries.json:
    [
      {"query":"...", "relevant_ids":["INS_xxx", ...]},
      ...
    ]
    """
    if not queries_file.exists():
        return {"skipped": True, "reason": f"Ficheiro nao encontrado: {queries_file}"}

    queries = json.loads(queries_file.read_text(encoding="utf-8"))
    if not queries:
        return {"skipped": True, "reason": "Lista de queries vazia."}

    recalls = []
    faithfulness_scores = []
    relevance_scores    = []

    import time
    for i, q in enumerate(queries):
        hits = search(q["query"], k=3)
        hit_ids = [h["inspection_id"] for h in hits]
        relevant = set(q.get("relevant_ids", []))
        found = len(relevant & set(hit_ids))
        recalls.append(1.0 if relevant and found > 0 else 0.0 if relevant else None)

        try:
            res = answer_with_rag(q["query"], k=3)
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "exhausted" in msg:
                print(f"    Query {i+1}: rate limit, aguardar 60s...")
                time.sleep(60)
                try:
                    res = answer_with_rag(q["query"], k=3)
                except Exception:
                    res = {"answer": "", "sources": []}
            else:
                res = {"answer": "", "sources": []}

        f_score = judge_faithfulness(res["answer"], hits)
        r_score = judge_relevance(q["query"], res["answer"])
        faithfulness_scores.append(f_score)
        relevance_scores.append(r_score)

        # Espacar pedidos para nao bater no limite de 5 req/min
        if i < len(queries) - 1:
            time.sleep(15)

    recalls = [r for r in recalls if r is not None]
    return {
        "queries_evaluated": len(queries),
        "recall_at_3_pct":   round(sum(recalls) / len(recalls) * 100, 1) if recalls else 0,
        "faithfulness_avg":  round(sum(faithfulness_scores) / len(faithfulness_scores), 2)
                              if faithfulness_scores else 0,
        "answer_relevance_avg": round(sum(relevance_scores) / len(relevance_scores), 2)
                                 if relevance_scores else 0,
    }


JUDGE_FAITHFUL = """Es um avaliador automatico. Recebes uma resposta gerada por um sistema e o contexto que foi fornecido a esse sistema.
Avalia em que medida a resposta esta suportada pelo contexto.
Retorna apenas um numero entre 0 e 1 (sem texto adicional):
- 1.0 = todas as afirmacoes da resposta estao no contexto
- 0.5 = parcialmente suportada
- 0.0 = a resposta tem afirmacoes nao verificaveis no contexto

CONTEXTO:
{context}

RESPOSTA:
{answer}

Numero:"""

JUDGE_RELEVANCE = """Es um avaliador automatico. A resposta abaixo responde directamente a pergunta?
Retorna apenas um numero entre 0 e 1 (sem texto adicional):
- 1.0 = responde directamente
- 0.5 = parcialmente
- 0.0 = nao responde

PERGUNTA: {query}
RESPOSTA: {answer}

Numero:"""


def judge_with_llm(prompt: str) -> float:
    """Chama o juiz LLM e extrai um numero entre 0 e 1. Com retry em rate limit."""
    if not API_KEY:
        return 0.0
    import time, re
    for attempt in range(3):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            r = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 20}
            )
            txt = (r.text or "").strip()
            m = re.search(r"[01](?:\.\d+)?", txt)
            if m:
                return max(0.0, min(1.0, float(m.group())))
            return 0.0
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "exhausted" in msg:
                wait = 30 * (attempt + 1)
                print(f"    Rate limit, aguardar {wait}s...")
                time.sleep(wait)
            else:
                return 0.0
    return 0.0


def judge_faithfulness(answer: str, hits: list) -> float:
    context = "\n".join(h["summary"] for h in hits)
    return judge_with_llm(JUDGE_FAITHFUL.format(context=context, answer=answer))


def judge_relevance(query: str, answer: str) -> float:
    return judge_with_llm(JUDGE_RELEVANCE.format(query=query, answer=answer))


# ───── Avaliacao do Rule Engine ────────────────────────────────────────────
def eval_rules(rules_file: Path) -> dict:
    """Avalia a conversao de regras em linguagem natural.

    Se o ficheiro nao existir, devolve secao com 'skipped' = True.

    Schema do rules.json:
    [
      {"text":"...", "should_be_ambiguous":false,
       "test_inspection": {... inspeccao sintetica ...},
       "should_match":true},
      ...
    ]
    """
    if not rules_file.exists():
        return {"skipped": True, "reason": f"Ficheiro nao encontrado: {rules_file}"}

    cases = json.loads(rules_file.read_text(encoding="utf-8"))
    parse_ok = correct = ambig_correct = 0

    for c in cases:
        rule = add_rule(c["text"])
        if "error" in rule:
            continue
        parse_ok += 1

        ambig_detected = bool(rule.get("validation", {}).get("ambiguities"))
        if ambig_detected == c.get("should_be_ambiguous", False):
            ambig_correct += 1

        if "test_inspection" in c:
            matches, _ = rule_matches(rule, c["test_inspection"])
            if matches == c.get("should_match", False):
                correct += 1

    n = len(cases)
    return {
        "rules_evaluated":          n,
        "rule_parse_rate_pct":      round(parse_ok / n * 100, 1) if n else 0,
        "rule_correctness_pct":     round(correct / n * 100, 1) if n else 0,
        "ambiguity_detection_pct":  round(ambig_correct / n * 100, 1) if n else 0,
    }


# ───── Main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Harness de avaliacao TP2")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--ground-truth", default="data/ground_truth.json")
    ap.add_argument("--rag-queries",  default="data/rag_queries.json")
    ap.add_argument("--rule-cases",   default="data/rule_cases.json")
    ap.add_argument("--output",       default="evaluation_report.json")
    ap.add_argument("--strategies",   nargs="+",
                     default=["zero_shot", "chain_of_thought", "few_shot"])
    args = ap.parse_args()

    report = {}
    images_dir = Path(args.images_dir)

    def save_partial():
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[1] Avaliacao da analise visual")
    visual_results = {}
    for strat in args.strategies:
        print(f"  Estrategia: {strat}")
        try:
            visual_results[strat] = eval_visual(images_dir, Path(args.ground_truth), strategy=strat)
        except Exception as e:
            visual_results[strat] = {"error": str(e)[:200]}
            print(f"    ERRO: {e}")
    report["visual_analysis"] = visual_results
    save_partial()

    print("\n[2] Avaliacao do RAG")
    try:
        report["rag"] = eval_rag(Path(args.rag_queries))
    except Exception as e:
        report["rag"] = {"error": str(e)[:200], "note": "quota esgotada, resultados parciais"}
        print(f"  ERRO RAG: {str(e)[:200]}")
    save_partial()

    print("\n[3] Avaliacao do Rule Engine")
    try:
        report["rules"] = eval_rules(Path(args.rule_cases))
    except Exception as e:
        report["rules"] = {"error": str(e)[:200], "note": "quota esgotada, secao nao avaliada"}
        print(f"  ERRO Rules: {str(e)[:200]}")

    out = Path(args.output)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("  RESUMO DA AVALIACAO")
    print("=" * 70)
    for strat, r in visual_results.items():
        if "error" in r:
            print(f"  {strat:>20}: ERRO - {r['error'][:60]}")
            continue
        gt_tag = "" if r.get("ground_truth_available") else "  (sem ground-truth)"
        line = f"  {strat:>20}: parse={r.get('json_parse_rate_pct',0)}%"
        if r.get("ground_truth_available"):
            line += (f"  detection={r.get('issue_detection_rate_pct',0)}%"
                     f"  FP={r.get('false_positive_rate_pct',0)}%"
                     f"  sev={r.get('severity_accuracy_pct',0)}%")
        print(line + gt_tag)

    rag = report.get("rag", {})
    if rag.get("skipped"):
        print(f"  {'RAG':>20}: skipped ({rag.get('reason','')[:50]})")
    elif "error" in rag:
        print(f"  {'RAG':>20}: ERRO - {rag['error'][:50]}")
    else:
        print(f"  {'RAG':>20}: recall@3={rag.get('recall_at_3_pct',0)}%"
               f"  faithfulness={rag.get('faithfulness_avg',0)}"
               f"  relevance={rag.get('answer_relevance_avg',0)}")

    rl = report.get("rules", {})
    if rl.get("skipped"):
        print(f"  {'Rule Engine':>20}: skipped ({rl.get('reason','')[:50]})")
    elif "error" in rl:
        print(f"  {'Rule Engine':>20}: ERRO - {rl['error'][:50]}")
    else:
        print(f"  {'Rule Engine':>20}: parse={rl.get('rule_parse_rate_pct',0)}%"
               f"  correct={rl.get('rule_correctness_pct',0)}%"
               f"  ambig={rl.get('ambiguity_detection_pct',0)}%")
    print("=" * 70)
    print(f"\nRelatorio escrito em {out}")


if __name__ == "__main__":
    main()