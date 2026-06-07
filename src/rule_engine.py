"""
rule_engine.py
Converte regras em linguagem natural para JSON estruturado e executa-as
sobre os resultados de inspecoes.
"""

import argparse
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY    = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
RULES_DIR  = Path(os.getenv("RULES_DIR", "data/rules"))
RULES_DIR.mkdir(parents=True, exist_ok=True)

if API_KEY:
    genai.configure(api_key=API_KEY)

RULE_PROMPT = """Es um sistema que converte regras de negocio em linguagem natural para JSON estruturado.

Schema obrigatorio:
{
  "rule_id": "RULE_XXX",
  "natural_language": "texto original",
  "description": "reformulacao formal e inequivoca",
  "conditions": {
    "zone_filter": ["lista de zonas ou vazio para qualquer"],
    "time_filter": {"hours_start": 0, "hours_end": 23},
    "issue_types": ["empty_shelf|wrong_product|damaged|misaligned|label_missing"],
    "severity_threshold": "low|medium|high",
    "fill_rate_threshold": 0.0,
    "location_filter": "bottom|middle|top|any"
  },
  "action": {
    "alert_level": "info|warning|critical",
    "notification_message": "template com placeholders {zone}, {fill_rate}, {issue_count}"
  },
  "validation": {
    "is_valid": true,
    "ambiguities": ["lista de aspectos nao claros"],
    "assumptions": ["pressupostos assumidos"]
  }
}

INSTRUCOES:
- Quando a regra for clara, listar ambiguities como [] e is_valid=true.
- Quando houver ambiguidade, listar cada uma em ambiguities e indicar o pressuposto assumido em assumptions.
- O zone_filter vazio [] significa qualquer zona.
- O time_filter ausente significa 0-23 (sempre).
- fill_rate_threshold de 0.0 significa que nao se aplica.

REGRA A CONVERTER:
"{user_rule}"

Responde APENAS com JSON valido. Sem texto antes ou depois."""


def call_gemini(prompt: str) -> str:
    """Chama Gemini para gerar texto (sem imagem)."""
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY nao definida")
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.0, "max_output_tokens": 1500}
    )
    return response.text or ""


def extract_json(text: str) -> dict:
    """Extrai JSON do output do modelo."""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return {}


def add_rule(natural_language: str) -> dict:
    """Converte uma regra de linguagem natural para JSON e persiste."""
    prompt = RULE_PROMPT.replace("{user_rule}", natural_language.replace('"', "'"))
    raw    = call_gemini(prompt)
    parsed = extract_json(raw)

    if not parsed:
        return {
            "error": "Falha na conversao para JSON.",
            "raw_response": raw
        }

    rule_id = parsed.get("rule_id") or f"RULE_{uuid.uuid4().hex[:8].upper()}"
    parsed["rule_id"]   = rule_id
    parsed["created_at"] = datetime.now(timezone.utc).isoformat()
    parsed["natural_language"] = natural_language

    rule_file = RULES_DIR / f"{rule_id}.json"
    rule_file.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed


def list_rules() -> list:
    """Lista todas as regras persistidas."""
    rules = []
    for f in sorted(RULES_DIR.glob("RULE_*.json")):
        try:
            rules.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return rules


def delete_rule(rule_id: str) -> bool:
    """Apaga uma regra pelo ID."""
    rule_file = RULES_DIR / f"{rule_id}.json"
    if rule_file.exists():
        rule_file.unlink()
        return True
    return False


SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


def rule_matches(rule: dict, inspection: dict) -> tuple:
    """Verifica se uma regra dispara face a uma inspecao.
    Retorna (matches, reasoning)."""
    cond = rule.get("conditions", {})
    reasons = []

    zone_filter = cond.get("zone_filter") or []
    if zone_filter and inspection.get("zone_id") not in zone_filter:
        return False, f"Zona {inspection.get('zone_id')} nao esta no filtro {zone_filter}"
    if zone_filter:
        reasons.append(f"zona match {inspection.get('zone_id')}")

    time_filter = cond.get("time_filter") or {}
    if time_filter:
        ts = inspection.get("timestamp", "")
        try:
            hour = datetime.fromisoformat(ts.replace("Z", "+00:00")).hour
            h_start = time_filter.get("hours_start", 0)
            h_end   = time_filter.get("hours_end", 23)
            if not (h_start <= hour <= h_end):
                return False, f"Hora {hour} fora de {h_start}-{h_end}"
            reasons.append(f"hora {hour} no intervalo")
        except (ValueError, TypeError):
            pass

    fill_threshold = cond.get("fill_rate_threshold", 0.0)
    if fill_threshold and inspection.get("shelf_fill_rate", 1.0) > fill_threshold:
        return False, f"fill_rate {inspection.get('shelf_fill_rate')} acima de {fill_threshold}"
    if fill_threshold:
        reasons.append(f"fill_rate {inspection.get('shelf_fill_rate')} abaixo de {fill_threshold}")

    issue_types = cond.get("issue_types") or []
    sev_min     = SEVERITY_RANK.get(cond.get("severity_threshold", "low"), 1)
    matching_issues = []
    for iss in inspection.get("issues", []):
        if issue_types and iss.get("type") not in issue_types:
            continue
        if SEVERITY_RANK.get(iss.get("severity", "low"), 1) < sev_min:
            continue
        matching_issues.append(iss)

    if (issue_types or sev_min > 1) and not matching_issues:
        return False, "Nenhum issue corresponde aos filtros de tipo/severidade"
    if matching_issues:
        reasons.append(f"{len(matching_issues)} issue(s) correspondem")

    return True, "; ".join(reasons) if reasons else "Regra aplicavel sem filtros restritivos"


def execute_rules(inspection: dict) -> list:
    """Percorre todas as regras e devolve as notificacoes geradas."""
    notifications = []
    for rule in list_rules():
        matches, reason = rule_matches(rule, inspection)
        log = {
            "rule_id":    rule.get("rule_id"),
            "inspection_id": inspection.get("inspection_id"),
            "matched":   matches,
            "reasoning": reason
        }
        if matches:
            action = rule.get("action", {})
            template = action.get("notification_message", "Regra {rule_id} disparou na zona {zone}.")
            message = template.format(
                rule_id    = rule.get("rule_id", ""),
                zone       = inspection.get("zone_id", ""),
                fill_rate  = inspection.get("shelf_fill_rate", 0.0),
                issue_count = len(inspection.get("issues", []))
            )
            log["message"]     = message
            log["alert_level"] = action.get("alert_level", "info")
        notifications.append(log)
    return notifications


def main():
    ap = argparse.ArgumentParser(description="Motor de regras em linguagem natural")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add",  help="Adicionar regra")
    p_add.add_argument("text", help="Regra em linguagem natural")

    sub.add_parser("list", help="Listar regras")

    p_del = sub.add_parser("delete", help="Apagar regra")
    p_del.add_argument("rule_id")

    p_test = sub.add_parser("test", help="Testar regras contra uma inspecao")
    p_test.add_argument("inspection_file", help="Caminho do JSON de inspecao")

    args = ap.parse_args()

    if args.cmd == "add":
        rule = add_rule(args.text)
        if "error" in rule:
            print(f"Erro: {rule['error']}")
        else:
            print(f"Regra criada: {rule['rule_id']}")
            print(f"  Descricao: {rule.get('description','')}")
            ambig = rule.get("validation", {}).get("ambiguities", [])
            if ambig:
                print("  Ambiguidades detectadas:")
                for a in ambig:
                    print(f"    - {a}")

    elif args.cmd == "list":
        rules = list_rules()
        print(f"{len(rules)} regra(s) registada(s):")
        for r in rules:
            print(f"  [{r.get('rule_id')}] {r.get('description','')[:80]}")

    elif args.cmd == "delete":
        ok = delete_rule(args.rule_id)
        print("Regra apagada." if ok else "Regra nao encontrada.")

    elif args.cmd == "test":
        inspection = json.loads(Path(args.inspection_file).read_text(encoding="utf-8"))
        notifs = execute_rules(inspection)
        for n in notifs:
            tag = "DISPARO" if n["matched"] else "      -"
            print(f"  {tag} {n['rule_id']}: {n['reasoning']}")
            if n["matched"]:
                print(f"           {n.get('alert_level','?').upper()}: {n.get('message','')}")


if __name__ == "__main__":
    main()
