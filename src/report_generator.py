"""
report_generator.py
Gera um relatorio de inspeccao em Markdown agregando os resultados de uma
sessao de inspeccoes, regras disparadas e contexto historico do RAG.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rule_engine import execute_rules
from rag_memory  import search, answer_with_rag

INSPECTIONS_DIR = Path("data/inspections")


def load_session_inspections(session_date: str = None, zone: str = None) -> list:
    """Carrega inspeccoes filtradas por data e/ou zona."""
    records = []
    for f in sorted(INSPECTIONS_DIR.glob("INS_*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if session_date and not rec.get("timestamp", "").startswith(session_date):
            continue
        if zone and rec.get("zone_id") != zone:
            continue
        records.append(rec)
    return records


def summarise_session(records: list) -> dict:
    """Calcula metricas agregadas da sessao."""
    if not records:
        return {"total": 0, "critical": 0, "warnings": 0, "ok": 0, "zones": []}

    total    = len(records)
    critical = sum(1 for r in records if r.get("overall_status") == "critical")
    warnings = sum(1 for r in records if r.get("overall_status") == "warning")
    ok       = sum(1 for r in records if r.get("overall_status") == "ok")
    zones    = sorted({r.get("zone_id", "?") for r in records})
    avg_fill = sum(r.get("shelf_fill_rate", 0.0) for r in records) / total
    total_issues = sum(len(r.get("issues", [])) for r in records)

    return {
        "total": total,
        "critical": critical,
        "warnings": warnings,
        "ok": ok,
        "zones": zones,
        "avg_fill": round(avg_fill, 3),
        "total_issues": total_issues
    }


def section_executive(summary: dict, session_label: str) -> str:
    """Seccao 1: sumario executivo."""
    if summary["total"] == 0:
        return f"## 1. Sumario Executivo\n\nSem inspeccoes registadas em {session_label}.\n"

    lines = [
        "## 1. Sumario Executivo\n",
        f"Sessao: {session_label}.",
        f"Foram inspeccionadas {summary['total']} zonas, com {len(summary['zones'])} zonas unicas.",
        f"Resultados: {summary['critical']} criticas, {summary['warnings']} com avisos, {summary['ok']} normais.",
        f"Taxa de preenchimento media: {summary['avg_fill']:.0%}.",
        f"Total de problemas detectados: {summary['total_issues']}.\n"
    ]
    return "\n".join(lines)


def section_zones(records: list) -> str:
    """Seccao 2: problemas por zona, com contexto historico do RAG."""
    lines = ["## 2. Problemas por Zona\n"]

    by_zone = {}
    for r in records:
        by_zone.setdefault(r.get("zone_id", "?"), []).append(r)

    for zone, recs in sorted(by_zone.items()):
        relevant = [r for r in recs if r.get("overall_status") in ("warning", "critical")]
        if not relevant:
            continue
        lines.append(f"### Zona {zone}\n")
        for r in relevant:
            lines.append(f"**Inspeccao {r['inspection_id']}** ({r.get('timestamp','')[:10]})")
            lines.append(f"- Estado: {r.get('overall_status')}")
            lines.append(f"- Fill rate: {r.get('shelf_fill_rate', 0.0):.0%}")
            for iss in r.get("issues", []):
                lines.append(
                    f"  - [{iss.get('severity','?')}] {iss.get('type','?')} em "
                    f"{iss.get('location','?')}: {iss.get('description','')[:120]}"
                )
            lines.append("")

        try:
            historical = search(f"problemas na zona {zone}", k=3)
            historical = [h for h in historical
                          if h["inspection_id"] not in {r["inspection_id"] for r in relevant}]
        except Exception:
            historical = []

        if historical:
            lines.append(f"_Contexto historico para {zone}:_")
            for h in historical[:3]:
                lines.append(f"  - {h['inspection_id']} ({h['metadata'].get('date','')}): "
                              f"{h['summary'][:140]}")
            lines.append("")

    if len(lines) == 1:
        lines.append("Nenhuma zona com problemas detectados.\n")
    return "\n".join(lines)


def section_rules(records: list) -> str:
    """Seccao 3: regras disparadas."""
    lines = ["## 3. Regras Disparadas\n"]
    any_fired = False
    for r in records:
        notifs = execute_rules(r)
        fired  = [n for n in notifs if n.get("matched")]
        if not fired:
            continue
        any_fired = True
        lines.append(f"**Inspeccao {r['inspection_id']}** (zona {r.get('zone_id')})")
        for n in fired:
            lines.append(f"- {n.get('alert_level','?').upper()} [{n['rule_id']}]: "
                          f"{n.get('message','')}")
            lines.append(f"  Motivo: {n.get('reasoning','')}")
        lines.append("")

    if not any_fired:
        lines.append("Nenhuma regra disparou nesta sessao.\n")
    return "\n".join(lines)


def section_history(records: list) -> str:
    """Seccao 4: contexto historico recuperado pelo RAG."""
    if not records:
        return "## 4. Contexto Historico\n\nSem dados.\n"

    zones_with_issues = sorted({r["zone_id"] for r in records if r.get("issues")})
    if not zones_with_issues:
        return "## 4. Contexto Historico\n\nSem problemas para contextualizar.\n"

    lines = ["## 4. Contexto Historico Relevante\n"]
    for zone in zones_with_issues[:3]:
        query = f"historico de problemas na zona {zone}"
        try:
            hits = search(query, k=3)
        except Exception:
            hits = []
        if not hits:
            continue
        lines.append(f"### Zona {zone}")
        for h in hits:
            lines.append(f"- [{h['inspection_id']}] {h['metadata'].get('date','')}: "
                          f"{h['summary'][:160]}")
        lines.append("")
    return "\n".join(lines)


def section_recommendations(records: list, summary: dict) -> str:
    """Seccao 5: recomendacoes accionaveis ordenadas por urgencia."""
    lines = ["## 5. Recomendacoes\n"]
    recs = []

    critical_records = [r for r in records if r.get("overall_status") == "critical"]
    for r in critical_records[:3]:
        for iss in r.get("issues", []):
            if iss.get("severity") == "high":
                recs.append({
                    "urgency": "imediata",
                    "text": f"Zona {r.get('zone_id')}: {iss.get('type')} em "
                             f"{iss.get('location')}. {iss.get('description','')[:140]}"
                })

    warning_records = [r for r in records if r.get("overall_status") == "warning"]
    for r in warning_records[:3]:
        if r.get("shelf_fill_rate", 1.0) < 0.6:
            recs.append({
                "urgency": "esta_semana",
                "text": f"Zona {r.get('zone_id')}: reposicao necessaria "
                         f"(fill rate {r.get('shelf_fill_rate'):.0%})."
            })

    if summary.get("avg_fill", 1.0) < 0.7:
        recs.append({
            "urgency": "proximo_mes",
            "text": f"Fill rate medio da loja em {summary['avg_fill']:.0%}. "
                     "Rever processo geral de reposicao."
        })

    if not recs:
        lines.append("Sem accoes correctivas necessarias.\n")
        return "\n".join(lines)

    order = {"imediata": 0, "esta_semana": 1, "proximo_mes": 2}
    recs.sort(key=lambda r: order.get(r["urgency"], 3))

    for i, r in enumerate(recs[:5], 1):
        label = {"imediata": "IMEDIATA", "esta_semana": "ESTA SEMANA",
                  "proximo_mes": "PROXIMO MES"}.get(r["urgency"], "")
        lines.append(f"{i}. [{label}] {r['text']}")
    lines.append("")
    return "\n".join(lines)


def build_report(records: list, session_label: str) -> str:
    """Compoe o relatorio completo em Markdown."""
    summary = summarise_session(records)
    parts = [
        f"# Relatorio de Inspeccao Visual\n",
        f"Gerado em: {datetime.now(timezone.utc).isoformat()}\n",
        "---\n",
        section_executive(summary, session_label),
        section_zones(records),
        section_rules(records),
        section_history(records),
        section_recommendations(records, summary),
        "---\n",
        "_Relatorio gerado pelo TP2 LIACD._\n"
    ]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Gerador de relatorio de inspeccao")
    ap.add_argument("--session", help="Data da sessao (YYYY-MM-DD), default=hoje")
    ap.add_argument("--zone",    help="Filtrar por zona")
    ap.add_argument("--output",  default="data/inspections/report.md")
    args = ap.parse_args()

    session_date  = args.session or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    session_label = f"{session_date}" + (f" zona {args.zone}" if args.zone else "")

    records = load_session_inspections(session_date=session_date, zone=args.zone)
    print(f"Inspeccoes encontradas para {session_label}: {len(records)}")

    report = build_report(records, session_label)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Relatorio escrito em {out} ({len(report):,} caracteres).")


if __name__ == "__main__":
    main()
