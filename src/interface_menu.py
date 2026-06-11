"""
interface_menu.py
Interface interactiva com menu numerado para o sistema de inspeccao visual.
Versao amigavel da interface: utilizador escolhe accoes por numero em vez
de digitar comandos. Util para demonstracao e defesa oral.
"""

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from shelf_inspector  import inspect_image
from rule_engine      import add_rule, list_rules, delete_rule, execute_rules
from rag_memory       import (
    index_inspection, search, answer_with_rag,
    index_all_from_disk, compare_chunking_strategies
)
from report_generator import load_session_inspections, build_report


# ============================================================
# Helpers de UI
# ============================================================

def banner(text):
    """Banner visual entre seccoes."""
    print()
    print("=" * 60)
    print(f"  {text}")
    print("=" * 60)


def pause():
    """Pausa para o utilizador ler antes de voltar ao menu."""
    input("\nCarrega Enter para continuar...")


def ask(prompt, default=None):
    """Pede input com valor por defeito."""
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else (default or "")


def ask_choice(prompt, options, default_idx=None):
    """Apresenta lista numerada e devolve escolha do utilizador.

    options: lista de tuplos (valor_devolvido, texto_apresentado)
    """
    print(f"\n{prompt}")
    for i, (_, label) in enumerate(options, 1):
        marker = " (default)" if default_idx and i == default_idx else ""
        print(f"  {i}. {label}{marker}")
    while True:
        raw = input("Escolha: ").strip()
        if not raw and default_idx:
            return options[default_idx - 1][0]
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        except ValueError:
            pass
        print(f"Numero invalido. Escolhe entre 1 e {len(options)}.")


def list_images_in(dir_path):
    """Devolve lista ordenada de imagens validas em dir_path."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    if not Path(dir_path).exists():
        return []
    return sorted([
        f for f in Path(dir_path).iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ])


# ============================================================
# Accoes do menu
# ============================================================

def action_inspect():
    """Inspecionar uma imagem."""
    banner("INSPECIONAR UMA IMAGEM")

    # Escolha do diretorio
    default_dir = "data/images/_pool"
    image_dir = ask("Pasta com imagens", default_dir)
    images = list_images_in(image_dir)

    if not images:
        print(f"\nNenhuma imagem em {image_dir}.")
        pause()
        return

    # Escolha da imagem
    print(f"\nEncontradas {len(images)} imagens. Mostrando as primeiras 20:")
    shown = images[:20]
    for i, img in enumerate(shown, 1):
        print(f"  {i:2d}. {img.name}")
    if len(images) > 20:
        print(f"  ... e mais {len(images) - 20}")

    raw = input("\nNumero da imagem (ou nome completo): ").strip()
    if not raw:
        print("Nenhuma imagem escolhida. Cancelado.")
        pause()
        return
    if raw.isdigit():
        idx = int(raw)
        if not (1 <= idx <= len(shown)):
            print(f"Numero invalido. Escolhe entre 1 e {len(shown)}.")
            pause()
            return
        img_path = shown[idx - 1]
    else:
        img_path = Path(image_dir) / raw
        if not img_path.exists() or not img_path.is_file():
            print(f"Nao encontrei o ficheiro {img_path}.")
            pause()
            return

    # Escolha da zona
    zone = ask("Zona (ex: Z_S1, Z_S2, Z_N1)", "Z_S1")

    # Escolha da estrategia
    strategy = ask_choice(
        "Estrategia de prompting:",
        [
            ("chain_of_thought", "Chain-of-Thought (recomendada)"),
            ("zero_shot",        "Zero-shot (rapida)"),
            ("few_shot",         "Few-shot (com exemplos)"),
        ],
        default_idx=1
    )

    print(f"\nA inspecionar {img_path.name} (zona={zone}, {strategy})...")
    try:
        record = inspect_image(img_path, strategy=strategy, zone_id=zone)
        index_inspection(record)

        print(f"\nInspeccao criada: {record['inspection_id']}")
        print(f"  Estado:    {record['overall_status']}")
        print(f"  Fill rate: {record['shelf_fill_rate']}")
        print(f"  Issues:    {len(record['issues'])}")
        for iss in record["issues"]:
            print(f"    - [{iss.get('severity','?')}] {iss.get('type','?')}: "
                   f"{iss.get('description','')[:80]}")

        # Verificar regras
        notifs = execute_rules(record)
        fired  = [n for n in notifs if n.get("matched")]
        if fired:
            print(f"\n  Regras disparadas: {len(fired)}")
            for n in fired:
                print(f"    {n.get('alert_level','?').upper()} "
                       f"[{n['rule_id']}]: {n.get('message','')}")
    except Exception as e:
        print(f"\nErro: {e}")
    pause()


def action_add_rule():
    """Adicionar uma nova regra."""
    banner("ADICIONAR REGRA")
    print("\nEscreve a regra em linguagem natural. Exemplos:")
    print("  - Avisa quando uma prateleira estiver mais de 30% vazia")
    print("  - Notifica em Z_S1 entre as 10h e 13h se houver produtos danificados")
    print("  - Critico quando o fill rate cair abaixo de 50%")
    text = input("\nRegra: ").strip()
    if not text:
        print("Texto vazio. Operacao cancelada.")
        pause()
        return

    try:
        rule = add_rule(text)
        print(f"\nRegra guardada: {rule['rule_id']}")
        print(f"  Descricao: {rule.get('description','')}")
        amb = rule.get("validation", {}).get("ambiguities", [])
        if amb:
            print(f"\n  Ambiguidades detectadas ({len(amb)}):")
            for a in amb:
                print(f"    - {a}")
        ass = rule.get("validation", {}).get("assumptions", [])
        if ass:
            print(f"  Assumpcoes feitas:")
            for a in ass:
                print(f"    - {a}")
    except Exception as e:
        print(f"\nErro: {e}")
    pause()


def action_list_rules():
    """Listar todas as regras."""
    banner("REGRAS DEFINIDAS")
    rules = list_rules()
    if not rules:
        print("\nNenhuma regra definida.")
    else:
        print(f"\nTotal: {len(rules)} regras\n")
        for r in rules:
            print(f"[{r['rule_id']}]")
            print(f"  NL: {r.get('natural_language','')}")
            print(f"  Desc: {r.get('description','')[:80]}")
            cond = r.get("conditions", {})
            issue_types = cond.get("issue_types", [])
            if issue_types:
                print(f"  Tipos: {', '.join(issue_types)}")
            print()
    pause()


def action_delete_rule():
    """Apagar uma regra."""
    banner("APAGAR REGRA")
    rules = list_rules()
    if not rules:
        print("\nNao ha regras para apagar.")
        pause()
        return

    print("\nRegras existentes:")
    for i, r in enumerate(rules, 1):
        print(f"  {i}. {r['rule_id']} - {r.get('natural_language','')[:60]}")

    raw = input("\nNumero da regra a apagar (Enter ou 0 para cancelar): ").strip()
    if not raw or raw == "0":
        print("Cancelado.")
        pause()
        return
    try:
        idx = int(raw)
        if not (1 <= idx <= len(rules)):
            print("Numero invalido.")
            pause()
            return
        rid = rules[idx - 1]["rule_id"]
        delete_rule(rid)
        print(f"Regra {rid} apagada.")
    except ValueError:
        print("Input invalido.")
    pause()


def action_history():
    """Pesquisar histórico via RAG (com LLM)."""
    banner("PERGUNTAR AO HISTORICO (RAG)")
    print("\nExemplos de perguntas:")
    print("  - Que zonas tiveram problemas esta semana?")
    print("  - Houve prateleiras vazias em Z_S1?")
    print("  - Resume o estado da loja na ultima visita")
    q = input("\nPergunta: ").strip()
    if not q:
        pause()
        return

    print("\nA processar (pode demorar)...")
    try:
        res = answer_with_rag(q, k=3)
        print(f"\nResposta:\n{res['answer']}\n")
        print(f"Fontes consultadas ({len(res['sources'])}):")
        for s in res["sources"]:
            print(f"  - {s['inspection_id']} ({s['metadata'].get('zone_id','?')}, "
                   f"{s['metadata'].get('date','?')})")
    except Exception as e:
        print(f"\nErro: {e}")
    pause()


def action_search():
    """Pesquisa semântica directa (sem LLM)."""
    banner("PESQUISA SEMANTICA (sem LLM)")
    q = input("\nQuery: ").strip()
    if not q:
        pause()
        return
    try:
        results = search(q, k=5)
        if not results:
            print("\nNenhum resultado.")
        else:
            print(f"\n{len(results)} resultado(s):\n")
            for r in results:
                print(f"[{r['inspection_id']}] dist={r['distance']:.3f}")
                print(f"  {r['summary'][:150]}")
                print()
    except Exception as e:
        print(f"\nErro: {e}")
    pause()


def action_compare_chunking():
    """Comparar estratégias de chunking."""
    banner("COMPARAR ESTRATEGIAS DE CHUNKING")

    queries_path = ask("Ficheiro de queries", "data/rag_queries.json")
    if not Path(queries_path).exists():
        print(f"Ficheiro nao encontrado: {queries_path}")
        pause()
        return

    k = ask_choice(
        "Valor de k:",
        [(1, "k=1 (mais severo)"), (3, "k=3 (relaxado)")],
        default_idx=1
    )

    import json
    queries = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    print(f"\nA comparar sobre {len(queries)} queries (pode demorar ~10s)...")
    try:
        results = compare_chunking_strategies(queries, k=k)
        print(f"\nCOMPARACAO (k={k}):\n")
        for strat, m in results.items():
            recall = m.get("recall_at_k_pct")
            r_str = f"{recall}%" if recall is not None else "N/A"
            print(f"  {strat:<12}  inspeccoes={m['inspections']:>2}  "
                   f"chunks={m['chunks_total']:>3}  queries={m['queries']:>2}  "
                   f"Recall@{k}={r_str}")
    except Exception as e:
        print(f"\nErro: {e}")
    pause()


def action_reindex():
    """Re-indexar inspeções no vector store."""
    banner("RE-INDEXAR INSPECCOES")
    strategy = ask_choice(
        "Estrategia de chunking:",
        [("hybrid",    "Hybrid (1 chunk por inspeccao)"),
         ("per_issue", "Per-issue (1 chunk por problema)")],
        default_idx=1
    )
    print("\nA re-indexar...")
    try:
        n = index_all_from_disk(strategy=strategy)
        print(f"Indexadas {n} inspeccoes com estrategia '{strategy}'.")
    except Exception as e:
        print(f"Erro: {e}")
    pause()


def action_report():
    """Gerar relatório Markdown."""
    banner("GERAR RELATORIO MARKDOWN")
    zone    = ask("Zona (opcional, vazio = todas)", "")
    session = ask("Data sessao YYYY-MM-DD (opcional)", "")
    try:
        records = load_session_inspections(
            session or None, zone or None
        )
        if not records:
            print("\nNenhuma inspeccao encontrada para esses filtros.")
            pause()
            return
        label_parts = []
        if session:
            label_parts.append(session)
        if zone:
            label_parts.append(zone)
        label = " / ".join(label_parts) if label_parts else "todas as inspeccoes"
        report = build_report(records, label)
        out = Path("data/inspections/report.md")
        out.write_text(report, encoding="utf-8")
        print(f"\nRelatorio escrito em {out}")
        print(f"  {len(records)} inspeccoes incluidas")
    except Exception as e:
        print(f"Erro: {e}")
    pause()


def action_compare_zones():
    """Comparar metricas agregadas entre 2 ou mais zonas."""
    banner("COMPARAR ZONAS")

    # Listar zonas existentes
    insp_dir = Path("data/inspections")
    files = sorted(insp_dir.glob("INS_*.json"))
    zones_seen = set()
    import json
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            z = d.get("zone_id")
            if z:
                zones_seen.add(z)
        except Exception:
            pass

    if len(zones_seen) < 2:
        print("\nPrecisa de pelo menos 2 zonas inspecionadas para comparar.")
        pause()
        return

    sorted_zones = sorted(zones_seen)
    print("\nZonas disponiveis:")
    for i, z in enumerate(sorted_zones, 1):
        print(f"  {i:2d}. {z}")

    raw = input("\nNumeros das zonas a comparar (separados por espaco, ex: 1 3): ").strip()
    if not raw:
        print("Cancelado.")
        pause()
        return
    try:
        chosen = []
        for s in raw.split():
            idx = int(s)
            if 1 <= idx <= len(sorted_zones):
                chosen.append(sorted_zones[idx - 1])
        if len(chosen) < 2:
            print("Precisa de escolher pelo menos 2 zonas.")
            pause()
            return
    except ValueError:
        print("Input invalido.")
        pause()
        return

    days = ask_choice(
        "Periodo:",
        [(7, "ultimos 7 dias"),
         (14, "ultimos 14 dias"),
         (30, "ultimo mes")],
        default_idx=1
    )

    from datetime import timedelta
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    period_start = start.strftime("%Y-%m-%d")
    period_end = end.strftime("%Y-%m-%d")

    print(f"\nComparacao no periodo {period_start} a {period_end}:\n")
    print(f"{'Zona':<8} {'Insp.':>6} {'Issues':>7} {'FillR':>7} {'Critical':>10}")
    print("-" * 50)

    for f in files:
        pass  # ja temos os files acima
    # Filtrar inspeccoes no periodo
    in_period = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            ts = d.get("timestamp", "")
            if ts and period_start <= ts[:10] <= period_end:
                in_period.append(d)
        except Exception:
            pass

    for zone in chosen:
        zone_records = [r for r in in_period if r.get("zone_id") == zone]
        n = len(zone_records)
        if n == 0:
            print(f"{zone:<8} {0:>6} {'-':>7} {'-':>7} {'-':>10}")
            continue
        total_issues = sum(len(r.get("issues", [])) for r in zone_records)
        avg_fill = sum(r.get("shelf_fill_rate", 0.0) for r in zone_records) / n
        critical = sum(1 for r in zone_records if r.get("overall_status") == "critical")
        print(f"{zone:<8} {n:>6} {total_issues:>7} {avg_fill:>7.2f} {critical:>10}")

    pause()


def action_list_inspections():
    """Listar inspeções guardadas."""
    banner("INSPECCOES GUARDADAS")
    insp_dir = Path("data/inspections")
    files = sorted(insp_dir.glob("INS_*.json"))
    if not files:
        print("\nNenhuma inspeccao guardada.")
    else:
        import json
        print(f"\nTotal: {len(files)} inspeccoes\n")
        for f in files:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                print(f"  {d['inspection_id']}")
                print(f"    zona={d.get('zone_id','?')}  "
                       f"status={d.get('overall_status','?')}  "
                       f"issues={len(d.get('issues',[]))}")
            except Exception:
                pass
    pause()


# ============================================================
# Menu principal
# ============================================================

def show_menu():
    print()
    print("=" * 60)
    print("  Sistema de Inspeccao Visual de Prateleiras")
    print("  TP2 LIACD - Universidade da Beira Interior")
    print("=" * 60)
    print()
    print("  1. Inspecionar uma imagem")
    print("  2. Adicionar uma regra")
    print("  3. Listar regras")
    print("  4. Apagar uma regra")
    print("  5. Perguntar ao historico (RAG com LLM)")
    print("  6. Pesquisa semantica (sem LLM)")
    print("  7. Comparar estrategias de chunking")
    print("  8. Re-indexar inspeccoes")
    print("  9. Listar inspeccoes")
    print(" 10. Gerar relatorio Markdown")
    print(" 11. Comparar zonas (last 7/14/30 days)")
    print()
    print("  0. Sair")
    print()


def main():
    actions = {
        "1":  action_inspect,
        "2":  action_add_rule,
        "3":  action_list_rules,
        "4":  action_delete_rule,
        "5":  action_history,
        "6":  action_search,
        "7":  action_compare_chunking,
        "8":  action_reindex,
        "9":  action_list_inspections,
        "10": action_report,
        "11": action_compare_zones,
    }

    while True:
        try:
            show_menu()
            choice = input("Escolha: ").strip()
            if choice in ("0", "q", "exit", "sair"):
                print("\nAte logo!")
                break
            if choice in actions:
                actions[choice]()
            else:
                print(f"\nOpcao invalida: '{choice}'. Escolhe entre 0 e 10.")
        except KeyboardInterrupt:
            print("\n\nInterrompido pelo utilizador. Ate logo!")
            break
        except Exception as e:
            print(f"\nErro inesperado: {e}")
            traceback.print_exc()
            pause()


if __name__ == "__main__":
    main()