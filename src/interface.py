"""
interface.py
Interface conversacional em linha de comandos. Suporta inspeccao, gestao
de regras, consulta historica e geracao de relatorios.
"""

import argparse
import json
import shlex
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from shelf_inspector  import inspect_image
from rule_engine      import add_rule, list_rules, delete_rule, execute_rules
from rag_memory       import index_inspection, search, answer_with_rag, index_all_from_disk
from report_generator import load_session_inspections, build_report

WELCOME = """
Sistema de Inspeccao Visual de Prateleiras (TP2 LIACD)
Comandos disponiveis:
  inspect <zone> --image <path> [--strategy chain_of_thought]
  inspect-dir <dir> --zone <zone>
  add rule "<texto da regra>"
  list rules
  delete rule <RULE_ID>
  test rule <inspection.json>
  history "<pergunta>"
  search "<pergunta>"
  reindex
  report [--session YYYY-MM-DD] [--zone Z_XX]
  help
  exit
"""


def cmd_inspect(args):
    """inspect <zone> --image <path> [--strategy ...]"""
    p = argparse.ArgumentParser(prog="inspect", add_help=False)
    p.add_argument("zone")
    p.add_argument("--image",    required=True)
    p.add_argument("--strategy", default="chain_of_thought",
                    choices=["zero_shot", "chain_of_thought", "few_shot"])
    p.add_argument("--no-cache", action="store_true")
    ns = p.parse_args(args)

    record = inspect_image(Path(ns.image), strategy=ns.strategy,
                            zone_id=ns.zone, use_cache=not ns.no_cache)
    index_inspection(record)
    notifs = execute_rules(record)
    fired  = [n for n in notifs if n.get("matched")]

    print(f"Inspeccao {record['inspection_id']}")
    print(f"  Zona:      {record['zone_id']}")
    print(f"  Estado:    {record['overall_status']}")
    print(f"  Fill rate: {record['shelf_fill_rate']}")
    print(f"  Issues:    {len(record['issues'])}")
    for iss in record["issues"]:
        print(f"    - [{iss.get('severity','?')}] {iss.get('type','?')}: "
               f"{iss.get('description','')[:80]}")
    if fired:
        print(f"  Regras disparadas: {len(fired)}")
        for n in fired:
            print(f"    {n.get('alert_level','?').upper()} [{n['rule_id']}]: "
                   f"{n.get('message','')}")


def cmd_inspect_dir(args):
    """inspect-dir <dir> --zone <zone>"""
    p = argparse.ArgumentParser(prog="inspect-dir", add_help=False)
    p.add_argument("dir")
    p.add_argument("--zone", default="Z_UNKNOWN")
    p.add_argument("--strategy", default="chain_of_thought",
                    choices=["zero_shot", "chain_of_thought", "few_shot"])
    ns = p.parse_args(args)

    image_dir = Path(ns.dir)
    images = sorted([f for f in image_dir.iterdir()
                      if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    print(f"A inspeccionar {len(images)} imagens em {image_dir}...")

    for img in images:
        try:
            rec = inspect_image(img, strategy=ns.strategy, zone_id=ns.zone)
            index_inspection(rec)
            print(f"  {img.name}: {rec['overall_status']} (fill {rec['shelf_fill_rate']})")
        except Exception as e:
            print(f"  {img.name}: ERRO - {e}")


def cmd_add_rule(args):
    """add rule "<texto>" """
    if not args:
        print("Uso: add rule \"<texto da regra>\"")
        return
    text = " ".join(args)
    rule = add_rule(text)
    if "error" in rule:
        print(f"Erro: {rule['error']}")
        return
    print(f"Regra criada: {rule['rule_id']}")
    print(f"  Descricao: {rule.get('description','')}")
    ambig = rule.get("validation", {}).get("ambiguities", []) or []
    if ambig:
        print("  Ambiguidades detectadas:")
        for a in ambig:
            print(f"    - {a}")
        print("  Pressupostos assumidos:")
        for a in rule.get("validation", {}).get("assumptions", []) or []:
            print(f"    - {a}")


def cmd_list_rules(_args):
    rules = list_rules()
    if not rules:
        print("Nenhuma regra registada.")
        return
    print(f"{len(rules)} regra(s):")
    for r in rules:
        print(f"  [{r.get('rule_id')}] {r.get('description','')[:100]}")


def cmd_delete_rule(args):
    if not args:
        print("Uso: delete rule <RULE_ID>")
        return
    rid = args[0]
    print("Regra apagada." if delete_rule(rid) else "Regra nao encontrada.")


def cmd_test_rule(args):
    if not args:
        print("Uso: test rule <inspection.json>")
        return
    path = Path(args[0])
    if not path.exists():
        print(f"Ficheiro nao encontrado: {path}")
        return
    rec = json.loads(path.read_text(encoding="utf-8"))
    notifs = execute_rules(rec)
    for n in notifs:
        tag = "DISPARO" if n["matched"] else "      -"
        print(f"  {tag} {n['rule_id']}: {n['reasoning']}")
        if n["matched"]:
            print(f"           {n.get('alert_level','?').upper()}: {n.get('message','')}")


def cmd_history(args):
    if not args:
        print("Uso: history \"<pergunta>\"")
        return
    query = " ".join(args)
    res = answer_with_rag(query, k=3)
    print(f"\n{res['answer']}\n")
    print(f"Fontes ({len(res['sources'])}):")
    for s in res["sources"]:
        print(f"  - {s['inspection_id']} ({s['metadata'].get('date','')})")


def cmd_search(args):
    if not args:
        print("Uso: search \"<pergunta>\"")
        return
    query = " ".join(args)
    for h in search(query, k=5):
        print(f"  [{h['inspection_id']}] dist={h['distance']:.3f}")
        print(f"    {h['summary'][:160]}")


def cmd_reindex(_args):
    n = index_all_from_disk()
    print(f"{n} inspeccoes (re)indexadas.")


def cmd_report(args):
    p = argparse.ArgumentParser(prog="report", add_help=False)
    p.add_argument("--session")
    p.add_argument("--zone")
    p.add_argument("--output", default="data/inspections/report.md")
    ns = p.parse_args(args)

    session_date = ns.session or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    label = session_date + (f" zona {ns.zone}" if ns.zone else "")
    records = load_session_inspections(session_date=session_date, zone=ns.zone)
    print(f"{len(records)} inspeccoes encontradas para {label}.")
    out = Path(ns.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(records, label), encoding="utf-8")
    print(f"Relatorio: {out}")


COMMANDS = {
    "inspect":     cmd_inspect,
    "inspect-dir": cmd_inspect_dir,
    "list":        cmd_list_rules,
    "search":      cmd_search,
    "reindex":     cmd_reindex,
    "report":      cmd_report,
    "history":     cmd_history,
}


def dispatch(line: str):
    """Despacha uma linha de input para o handler correcto."""
    try:
        tokens = shlex.split(line)
    except ValueError as e:
        print(f"Erro de parsing: {e}")
        return
    if not tokens:
        return

    head = tokens[0]
    rest = tokens[1:]

    if head == "add" and rest and rest[0] == "rule":
        cmd_add_rule(rest[1:])
    elif head == "list" and rest and rest[0] == "rules":
        cmd_list_rules(rest[1:])
    elif head == "delete" and rest and rest[0] == "rule":
        cmd_delete_rule(rest[1:])
    elif head == "test" and rest and rest[0] == "rule":
        cmd_test_rule(rest[1:])
    elif head in COMMANDS:
        try:
            COMMANDS[head](rest)
        except SystemExit:
            pass
        except Exception as e:
            print(f"Erro a executar comando: {e}")
    elif head in {"help", "?"}:
        print(WELCOME)
    elif head in {"exit", "quit"}:
        sys.exit(0)
    else:
        print(f"Comando desconhecido: {head}. Escreve 'help'.")


def main():
    """Loop interactivo. Aceita tambem comando unico via argv."""
    if len(sys.argv) > 1:
        dispatch(" ".join(sys.argv[1:]))
        return

    print(WELCOME)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        dispatch(line)


if __name__ == "__main__":
    main()
