"""
build_dataset.py
Constroi o dataset de imagens de prateleiras para o TP2 a partir de fotos
proprias colocadas em data/images/_pool/.

Fluxo:
1. Le imagens reais de data/images/_pool/ (colocadas pelo utilizador).
2. Copia para a categoria 'normal'.
3. Gera variantes sinteticas para as restantes 4 categorias.
4. Documenta a origem de cada imagem em data/images/SOURCES.txt.

Uso:
    # Coloca primeiro 10-20 fotos em data/images/_pool/
    python3 scripts/build_dataset.py
"""

import argparse
import random
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFilter

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False

random.seed(42)

DATA_DIR     = Path("data/images")
SOURCES_FILE = DATA_DIR / "SOURCES.txt"

CATEGORIES = {
    "normal":     225,
    "empty":      150,
    "planogram":  150,
    "messy":      120,
    "ambiguous":  105,
}


def ensure_dirs():
    for cat in CATEGORIES:
        (DATA_DIR / cat).mkdir(parents=True, exist_ok=True)


def add_empty_shelf_mask(img: Image.Image) -> Image.Image:
    """Sobrepoe rectangulos escuros para simular prateleira vazia."""
    img = img.copy().convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    n_holes = random.randint(1, 3)
    for _ in range(n_holes):
        x0 = random.randint(0, int(w * 0.5))
        y0 = random.randint(int(h * 0.3), int(h * 0.7))
        x1 = x0 + random.randint(int(w * 0.2), int(w * 0.4))
        y1 = y0 + random.randint(int(h * 0.1), int(h * 0.3))
        draw.rectangle([x0, y0, min(x1, w), min(y1, h)], fill=(40, 40, 40))
    return img


def add_messy_variant(img: Image.Image) -> Image.Image:
    """Rotaciona pequenas regioes para simular produtos tombados/desalinhados."""
    img = img.copy().convert("RGB")
    w, h = img.size
    n_patches = random.randint(2, 5)
    for _ in range(n_patches):
        pw, ph = random.randint(60, 140), random.randint(60, 140)
        x = random.randint(0, max(1, w - pw))
        y = random.randint(0, max(1, h - ph))
        patch = img.crop((x, y, x + pw, y + ph))
        patch = patch.rotate(random.choice([-25, -15, 15, 25]),
                              expand=False, fillcolor=(80, 80, 80))
        img.paste(patch, (x, y))
    return img


def add_planogram_variant(img: Image.Image) -> Image.Image:
    """Recorta e cola produtos noutras posicoes para simular planograma errado."""
    img = img.copy().convert("RGB")
    w, h = img.size
    for _ in range(random.randint(2, 4)):
        pw, ph = random.randint(80, 160), random.randint(80, 160)
        sx = random.randint(0, max(1, w - pw))
        sy = random.randint(0, max(1, h - ph))
        patch = img.crop((sx, sy, sx + pw, sy + ph))
        dx = random.randint(0, max(1, w - pw))
        dy = random.randint(0, max(1, h - ph))
        img.paste(patch, (dx, dy))
    return img


def add_ambiguous_variant(img: Image.Image) -> Image.Image:
    """Blur, escurecimento ou recorte parcial - casos onde a classificacao nao
    e' obvia."""
    img = img.copy().convert("RGB")
    op = random.choice(["blur", "darken", "crop"])
    if op == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=2.5))
    elif op == "darken":
        img = Image.eval(img, lambda v: int(v * 0.55))
    else:
        w, h = img.size
        cx, cy = random.randint(0, w // 3), random.randint(0, h // 3)
        img = img.crop((cx, cy, cx + w // 2, cy + h // 2))
    return img


def synthetic_variants(seed_imgs: List[Path], category: str,
                        target: int, output_dir: Path,
                        sources_log: list) -> int:
    """Gera variantes sinteticas a partir das imagens base."""
    transform = {
        "empty":     add_empty_shelf_mask,
        "messy":     add_messy_variant,
        "planogram": add_planogram_variant,
        "ambiguous": add_ambiguous_variant,
    }.get(category, lambda x: x)

    count = 0
    idx = 0
    failures = 0
    max_attempts = target * 3   # limite de seguranca para evitar loop infinito
    attempts = 0

    while count < target and seed_imgs and attempts < max_attempts:
        attempts += 1
        src_path = seed_imgs[idx % len(seed_imgs)]
        try:
            img = Image.open(src_path)
            variant = transform(img)
            out = output_dir / f"{category}_synth_{count:04d}.jpg"
            variant.convert("RGB").save(out, "JPEG", quality=85)
            sources_log.append(
                f"{out.name}\torigem: variante sintetica de {src_path.name}\t"
                f"transformacao: {category}"
            )
            count += 1
        except Exception as e:
            failures += 1
            if failures <= 3:
                print(f"  erro em {src_path.name}: {e}")
            elif failures == 4:
                print(f"  ... (mais erros suprimidos)")
        idx += 1

    if failures > 0:
        print(f"  {failures} falhas, {count} sucessos")
    return count


def populate(seed_pool: Path, sources_log: list):
    """Le fotos reais do pool, copia para 'normal' e gera variantes para outras."""
    seed_pool.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] A ler imagens do pool ({seed_pool})...")
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    downloaded = sorted([
        f for f in seed_pool.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ])

    heic_count = sum(1 for f in downloaded if f.suffix.lower() == ".heic")
    if heic_count > 0 and not HEIC_OK:
        print(f"\nERRO: {heic_count} ficheiros .heic encontrados mas pillow-heif")
        print("nao esta instalado. Corre primeiro:")
        print("  pip install pillow-heif")
        return False

    if not downloaded:
        print("\nERRO: Nenhuma imagem encontrada em " + str(seed_pool))
        print("Coloca as tuas fotos de prateleiras nessa pasta e corre de novo.")
        print("Formatos aceites: .jpg, .jpeg, .png, .webp, .heic")
        return False

    print(f"  {len(downloaded)} imagens base encontradas.")
    if len(downloaded) < 10:
        print(f"  AVISO: recomendado >= 10 imagens base para gerar boa variedade.")

    print(f"\n[2/3] A popular categoria 'normal' (copias directas)...")
    normal_dir = DATA_DIR / "normal"
    target_normal = CATEGORIES["normal"]
    n_real = min(len(downloaded), target_normal)

    for i, src in enumerate(downloaded[:n_real]):
        dst = normal_dir / f"normal_{i:04d}.jpg"
        try:
            img = Image.open(src).convert("RGB")
            img.save(dst, "JPEG", quality=88)
            sources_log.append(
                f"{dst.name}\torigem: foto propria ({src.name})\tcategoria: normal"
            )
        except Exception as e:
            print(f"  erro a converter {src.name}: {e}")

    needed_normal = target_normal - n_real
    if needed_normal > 0:
        print(f"  Faltam {needed_normal} para 'normal' - aplicando variacoes ligeiras...")
        for i in range(needed_normal):
            src = downloaded[i % len(downloaded)]
            try:
                img = Image.open(src).convert("RGB")
                w, h = img.size
                v = i % 4
                if v == 0:
                    img = img.crop((int(w*0.05), int(h*0.05),
                                     int(w*0.95), int(h*0.95)))
                elif v == 1:
                    img = Image.eval(img, lambda x: min(255, int(x * 1.05)))
                elif v == 2:
                    img = Image.eval(img, lambda x: int(x * 0.95))
                dst = normal_dir / f"normal_var_{i:04d}.jpg"
                img.save(dst, "JPEG", quality=85)
                sources_log.append(
                    f"{dst.name}\torigem: variacao de {src.name}\tcategoria: normal"
                )
            except Exception as e:
                print(f"  erro: {e}")

    print(f"\n[3/3] A gerar variantes sinteticas...")
    for cat, target in CATEGORIES.items():
        if cat == "normal":
            continue
        out_dir = DATA_DIR / cat
        n = synthetic_variants(downloaded, cat, target, out_dir, sources_log)
        print(f"  {cat}: {n} variantes geradas.")

    return True


def write_sources(sources_log: list):
    SOURCES_FILE.write_text(
        "# Origens das imagens do dataset TP2\n"
        "# Formato: ficheiro<TAB>origem<TAB>categoria\n"
        "# Imagens base: fotos proprias recolhidas em supermercados locais.\n"
        "# Variantes sinteticas geradas em pipeline local (build_dataset.py).\n"
        "# Licenca: imagens proprias - uso academico (TP2 LIACD).\n\n"
        + "\n".join(sources_log) + "\n",
        encoding="utf-8"
    )


def clean_existing():
    """Apaga imagens previamente geradas (mantem o pool)."""
    for cat in CATEGORIES:
        d = DATA_DIR / cat
        if not d.exists():
            continue
        for f in d.glob("*.jpg"):
            f.unlink()


def summary():
    print("\n" + "=" * 50)
    print("  RESUMO DO DATASET")
    print("=" * 50)
    total = 0
    for cat, target in CATEGORIES.items():
        actual = len(list((DATA_DIR / cat).glob("*.jpg")))
        total += actual
        status = "OK" if actual >= target else "FALTA"
        print(f"  {cat:<15} {actual:>4} / {target:<4} {status}")
    print(f"  {'TOTAL':<15} {total}")
    print("=" * 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-pool", default="data/images/_pool")
    ap.add_argument("--clean", action="store_true",
                     help="Apaga imagens geradas anteriormente antes de comecar.")
    args = ap.parse_args()

    if args.clean:
        print("A limpar imagens geradas anteriormente...")
        clean_existing()

    ensure_dirs()
    sources_log = []
    ok = populate(Path(args.seed_pool), sources_log)
    if not ok:
        return
    write_sources(sources_log)
    summary()
    print(f"\nLog de origens escrito em {SOURCES_FILE}")


if __name__ == "__main__":
    main()