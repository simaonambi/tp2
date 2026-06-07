"""
annotate.py
Script interactivo para criar data/ground_truth.json a partir das tuas imagens.

Mostra cada imagem (abre no visualizador do sistema) e pergunta-te:
- A zona (Z_S1, Z_S2, ...)
- Os problemas visiveis (escolha multipla)
- A severidade de cada um

No fim, escreve um JSON pronto a usar pelo evaluate.py.

Uso:
    python3 scripts/annotate.py
    python3 scripts/annotate.py --images-dir data/images/_pool
    python3 scripts/annotate.py --limit 15
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


ISSUE_TYPES = {
    "1": "empty_shelf",
    "2": "wrong_product",
    "3": "damaged",
    "4": "misaligned",
    "5": "label_missing",
    "6": "other"
}

SEVERITIES = {"l": "low", "m": "medium", "h": "high"}


def open_image(path: Path):
    """Abre a imagem no visualizador do sistema (macOS, Linux, Windows)."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(path)],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        elif sys.platform == "win32":
            subprocess.Popen(["start", "", str(path)], shell=True)
    except Exception as e:
        print(f"  (nao conseguiu abrir o visualizador: {e})")


def prompt_issues() -> list:
    """Pede a lista de problemas para uma imagem."""
    issues = []
    print("\nTipos de problema:")
    for k, v in ISSUE_TYPES.items():
        print(f"  {k} = {v}")
    print("  Enter (vazio) = passar para a proxima imagem")

    while True:
        choice = input("Tipo de problema (1-6) ou Enter para terminar: ").strip()
        if not choice:
            break
        if choice not in ISSUE_TYPES:
            print("  Opcao invalida.")
            continue

        sev = input("  Severidade (l=low, m=medium, h=high): ").strip().lower()
        if sev not in SEVERITIES:
            print("  Severidade invalida, a usar 'medium'.")
            sev = "m"

        location = input("  Localizacao (livre, ex: 'prateleira inferior'): ").strip()

        issues.append({
            "type":     ISSUE_TYPES[choice],
            "severity": SEVERITIES[sev],
            "location": location or "nao especificado"
        })
        print(f"  -> registado: {ISSUE_TYPES[choice]} ({SEVERITIES[sev]})")

    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="data/images/_pool",
                     help="Pasta com as imagens a anotar")
    ap.add_argument("--output", default="data/ground_truth.json")
    ap.add_argument("--limit", type=int, default=20,
                     help="Numero maximo de imagens a anotar (default: 20)")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"ERRO: pasta nao existe: {images_dir}")
        return

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    images = sorted([f for f in images_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in extensions])

    if not images:
        print(f"ERRO: nenhuma imagem em {images_dir}")
        return

    images = images[:args.limit]
    print(f"\nA anotar {len(images)} imagens de {images_dir}")
    print(f"Output: {args.output}\n")

    output_path = Path(args.output)
    existing = []
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            print(f"Ja existem {len(existing)} anotacoes. Vai continuar a partir dai.")
        except json.JSONDecodeError:
            pass
    annotated_names = {e.get("image") for e in existing}

    annotations = list(existing)

    for i, img_path in enumerate(images, 1):
        rel_name = img_path.name
        if rel_name in annotated_names:
            print(f"\n[{i}/{len(images)}] {rel_name} - JA ANOTADA (skip)")
            continue

        print(f"\n[{i}/{len(images)}] A abrir: {rel_name}")
        open_image(img_path)

        zone = input("Zona (ex: Z_S1, Z_S3, Z_C2) [Z_S1, s=saltar, q=sair]: ").strip()
        if zone.lower() == "s":
            print("  (imagem saltada, nao guardada)")
            continue
        if zone.lower() == "q":
            break
        if not zone:
            zone = "Z_S1"

        print("Tem problemas? Se nao tiver, basta Enter:")
        issues = prompt_issues()

        entry = {
            "image": rel_name,
            "zone":  zone,
            "issues": issues
        }
        annotations.append(entry)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(annotations, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"  Guardado ({len(annotations)} no total).")

        if i < len(images):
            cont = input("Continuar? (Enter=sim, q=sair): ").strip().lower()
            if cont == "q":
                break

    print(f"\nConcluido. {len(annotations)} anotacoes em {output_path}")


if __name__ == "__main__":
    main()