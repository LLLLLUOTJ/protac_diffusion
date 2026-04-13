#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/yanglh/L/protac_diffusion}
DIFFLINKER_ROOT=${DIFFLINKER_ROOT:-/home/yanglh/L/DiffLinker}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-diffusion-server}
OUT_ROOT=${OUT_ROOT:-$PROJECT_ROOT/outputs/difflinker_smoke}
MAX_SOURCES=${MAX_SOURCES:-4}
N_SAMPLES=${N_SAMPLES:-2}
SEED=${SEED:-13}
GENERATED_SUBDIR=${GENERATED_SUBDIR:-generated}
MODEL_PATH=${MODEL_PATH:-$DIFFLINKER_ROOT/models/geom_difflinker_given_anchors.ckpt}
PROTAC_SDF=${PROTAC_SDF:-}

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

mkdir -p "$OUT_ROOT"
PREPARE_CMD=(
  python "$PROJECT_ROOT/prepare_difflinker_inputs.py"
  --weak-anchor-csv "$PROJECT_ROOT/outputs/weak_anchor_best/weak_anchor_dataset.csv"
  --selection-json "$PROJECT_ROOT/outputs/linker_token_eval/all_generations.json"
  --out-dir "$OUT_ROOT/inputs"
  --max-sources "$MAX_SOURCES"
  --seed "$SEED"
)
if [[ -n "$PROTAC_SDF" ]]; then
  PREPARE_CMD+=(--protac-sdf "$PROTAC_SDF")
fi
"${PREPARE_CMD[@]}"

OUT_ROOT="$OUT_ROOT" \
DIFFLINKER_ROOT="$DIFFLINKER_ROOT" \
GENERATED_SUBDIR="$GENERATED_SUBDIR" \
MODEL_PATH="$MODEL_PATH" \
N_SAMPLES="$N_SAMPLES" \
PROJECT_ROOT="$PROJECT_ROOT" \
python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
difflinker_root = Path(os.environ["DIFFLINKER_ROOT"])
generated_subdir = os.environ["GENERATED_SUBDIR"]
model_path = os.environ["MODEL_PATH"]
n_samples = int(os.environ["N_SAMPLES"])
project_root = Path(os.environ["PROJECT_ROOT"])

manifest = json.loads((out_root / "inputs" / "manifest.json").read_text())
for row in manifest:
    sample_dir = Path(row["fragments_path"]).parent
    gen_dir = sample_dir / generated_subdir
    gen_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", str(project_root / "run_difflinker_generate.py"),
        "--difflinker-root", str(difflinker_root),
        "--fragments", row["fragments_path"],
        "--model", model_path,
        "--linker-size", str(row["linker_size"]),
        "--anchors", row["anchors"],
        "--output", str(gen_dir),
        "--n-samples", str(n_samples),
    ]
    print("[run]", row["sample_id"], " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
PY

python "$PROJECT_ROOT/convert_difflinker_outputs.py" \
  --manifest "$OUT_ROOT/inputs/manifest.json" \
  --generated-subdir "$GENERATED_SUBDIR" \
  --output-json "$OUT_ROOT/generated.json" \
  --output-csv "$OUT_ROOT/generated.csv"
