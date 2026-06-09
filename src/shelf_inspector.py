"""
shelf_inspector.py
Modulo de inspecao visual de prateleiras usando Google Gemini 1.5 Flash.
Suporta tres estrategias de prompting: zero_shot, chain_of_thought, few_shot.
"""

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

load_dotenv()

API_KEY        = os.getenv("GEMINI_API_KEY")
MODEL_NAME     = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
CACHE_DIR      = Path(os.getenv("CACHE_DIR", "cache"))
INSPECTIONS_DIR = Path(os.getenv("INSPECTIONS_DIR", "data/inspections"))

CACHE_DIR.mkdir(parents=True, exist_ok=True)
INSPECTIONS_DIR.mkdir(parents=True, exist_ok=True)

if API_KEY:
    genai.configure(api_key=API_KEY)


def file_hash(path: Path) -> str:
    """Calcula o hash MD5 do conteudo de um ficheiro."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_path(image_path: Path, strategy: str) -> Path:
    """Caminho do ficheiro de cache para uma combinacao imagem+estrategia."""
    return CACHE_DIR / f"{file_hash(image_path)}_{strategy}.json"


def load_prompt(strategy: str) -> str:
    """Carrega o prompt de disco. Cria com defaults se nao existir."""
    prompts_dir = Path("prompts")
    prompts_dir.mkdir(exist_ok=True)
    path = prompts_dir / f"shelf_{strategy}.txt"
    if not path.exists():
        path.write_text(DEFAULT_PROMPTS[strategy], encoding="utf-8")
    return path.read_text(encoding="utf-8")


DEFAULT_PROMPTS = {
    "zero_shot": """Analisa esta imagem de uma prateleira de supermercado e produz um JSON com a seguinte estrutura exacta:

{
  "overall_status": "ok|warning|critical",
  "issues": [
    {
      "issue_id": "ISS_001",
      "type": "empty_shelf|wrong_product|damaged|misaligned|label_missing|other",
      "location": "descricao textual",
      "severity": "low|medium|high",
      "description": "descricao do problema",
      "confidence": 0.0,
      "affected_area_pct": 0.0
    }
  ],
  "shelf_fill_rate": 0.0,
  "products_detected": [],
  "model_reasoning": "raciocinio passo a passo"
}

Responde APENAS com JSON valido. Sem texto antes ou depois.""",

    "chain_of_thought": """Analisa esta imagem de uma prateleira de supermercado seguindo estes passos:

Passo 1: Descreve em prosa o que ves na imagem (numero aproximado de prateleiras, tipos de produtos, ocupacao visual).
Passo 2: Examina cada zona da imagem (topo, meio, base) e identifica anomalias se existirem.
Passo 3: Classifica cada anomalia identificada em termos de tipo, severidade, e localizacao.
Passo 4: Estima a taxa de preenchimento global (shelf_fill_rate) entre 0.0 e 1.0.
Passo 5: Produz o JSON final.

Coloca o raciocinio dos passos 1-4 no campo model_reasoning. O JSON final tem este schema:

{
  "overall_status": "ok|warning|critical",
  "issues": [{"issue_id":"ISS_001","type":"...","location":"...","severity":"...","description":"...","confidence":0.0,"affected_area_pct":0.0}],
  "shelf_fill_rate": 0.0,
  "products_detected": [],
  "model_reasoning": "passo 1: ... passo 2: ... passo 3: ... passo 4: ..."
}

Responde APENAS com JSON valido.""",

    "few_shot": """Analisa esta imagem de uma prateleira de supermercado e produz um JSON estruturado.

EXEMPLOS DE ANALISES CORRECTAS ANTERIORES:

Exemplo 1 (prateleira normal):
{"overall_status":"ok","issues":[],"shelf_fill_rate":0.92,"products_detected":["cereais","bolachas"],"model_reasoning":"Prateleira bem organizada, produtos alinhados, sem espacos vazios significativos."}

Exemplo 2 (prateleira parcialmente vazia):
{"overall_status":"warning","issues":[{"issue_id":"ISS_001","type":"empty_shelf","location":"prateleira inferior, lado esquerdo","severity":"medium","description":"Espaco de aproximadamente 40cm sem produto na prateleira inferior.","confidence":0.85,"affected_area_pct":0.18}],"shelf_fill_rate":0.71,"products_detected":["leite","iogurtes"],"model_reasoning":"Prateleira inferior tem um vazio claro do lado esquerdo. Restantes prateleiras estao normalmente preenchidas."}

Exemplo 3 (produto tombado):
{"overall_status":"warning","issues":[{"issue_id":"ISS_001","type":"misaligned","location":"prateleira do meio, centro","severity":"low","description":"Garrafa tombada no centro da prateleira do meio.","confidence":0.91,"affected_area_pct":0.05}],"shelf_fill_rate":0.85,"products_detected":["bebidas"],"model_reasoning":"Garrafa visivelmente fora da posicao vertical. Nao bloqueia outros produtos."}

Agora analisa a imagem fornecida usando o mesmo schema e nivel de detalhe.
Responde APENAS com JSON valido."""
}


def extract_json(text: str) -> dict:
    """Extrai JSON do texto retornado pelo modelo."""
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


def call_gemini(prompt: str, image_path: Path, max_retries: int = 3) -> str:
    """Chama o Gemini com backoff exponencial em caso de rate limit."""
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY nao definida no .env")

    model = genai.GenerativeModel(MODEL_NAME)
    img = Image.open(image_path)

    delay = 5
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                [prompt, img],
                generation_config={"temperature": 0.0, "max_output_tokens": 4000}
            )
            return response.text or ""
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "rate" in msg:
                print(f"  Rate limit atingido, aguardar {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise RuntimeError("Limite de tentativas atingido. Possivel quota esgotada.")


def inspect_image(image_path: Path, strategy: str = "chain_of_thought",
                  zone_id: str = "Z_UNKNOWN", use_cache: bool = True) -> dict:
    """Inspeciona uma imagem e devolve o registo de inspecao completo."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Imagem nao encontrada: {image_path}")

    cache_file = cache_path(image_path, strategy)
    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        cached["from_cache"] = True
        return cached

    prompt = load_prompt(strategy)
    raw    = call_gemini(prompt, image_path)
    parsed = extract_json(raw)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    inspection_id = f"INS_{timestamp}_{file_hash(image_path)[:8]}"

    record = {
        "inspection_id":  inspection_id,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "image_path":     str(image_path),
        "zone_id":        zone_id,
        "strategy":       strategy,
        "overall_status": parsed.get("overall_status", "unknown"),
        "issues":         parsed.get("issues", []),
        "shelf_fill_rate": parsed.get("shelf_fill_rate", 0.0),
        "products_detected": parsed.get("products_detected", []),
        "model_reasoning":   parsed.get("model_reasoning", ""),
        "raw_response":      raw,
        "json_parse_ok":     bool(parsed),
        "from_cache":        False
    }

    cache_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    inspection_file = INSPECTIONS_DIR / f"{inspection_id}.json"
    inspection_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    return record


def main():
    ap = argparse.ArgumentParser(description="Inspecao visual de prateleiras")
    ap.add_argument("--image",    required=True, help="Caminho da imagem")
    ap.add_argument("--zone",     default="Z_UNKNOWN", help="Identificador da zona")
    ap.add_argument("--strategy", choices=["zero_shot","chain_of_thought","few_shot"],
                    default="chain_of_thought")
    ap.add_argument("--no-cache", action="store_true", help="Ignorar cache")
    args = ap.parse_args()

    print(f"A inspeccionar {args.image} (zona={args.zone}, estrategia={args.strategy})...")
    record = inspect_image(Path(args.image), strategy=args.strategy,
                            zone_id=args.zone, use_cache=not args.no_cache)

    print(f"\nInspection ID: {record['inspection_id']}")
    print(f"Status:        {record['overall_status']}")
    print(f"Fill rate:     {record['shelf_fill_rate']}")
    print(f"Issues:        {len(record['issues'])}")
    print(f"From cache:    {record['from_cache']}")
    if record["issues"]:
        for iss in record["issues"]:
            print(f"  - [{iss.get('severity','?')}] {iss.get('type','?')}: {iss.get('description','')[:80]}")


if __name__ == "__main__":
    main()