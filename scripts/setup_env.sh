#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

VARIANT="${1:-${TORCH_VARIANT}}"
ENV_FILE="${PROJECT_ROOT}/environment.server.yml"

require_conda

if conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
  echo "[env] updating existing env ${CONDA_ENV_NAME}"
  conda env update -n "${CONDA_ENV_NAME}" -f "${ENV_FILE}" --prune
else
  echo "[env] creating env ${CONDA_ENV_NAME}"
  conda env create -n "${CONDA_ENV_NAME}" -f "${ENV_FILE}"
fi

case "${VARIANT}" in
  cpu)
    echo "[env] installing CPU PyTorch"
    conda remove -y -n "${CONDA_ENV_NAME}" pytorch pytorch-cuda cpuonly torchvision torchaudio >/dev/null 2>&1 || true
    conda install -y -n "${CONDA_ENV_NAME}" --override-channels -c pytorch pytorch=2.2.\* cpuonly
    ;;
  cuda121)
    echo "[env] installing CUDA 12.1 PyTorch"
    conda remove -y -n "${CONDA_ENV_NAME}" pytorch pytorch-cuda cpuonly torchvision torchaudio >/dev/null 2>&1 || true
    conda install -y -n "${CONDA_ENV_NAME}" --override-channels -c pytorch -c nvidia pytorch=2.2.\* pytorch-cuda=12.1
    ;;
  *)
    echo "[error] unsupported TORCH_VARIANT=${VARIANT}. Use cpu or cuda121." >&2
    exit 1
    ;;
esac

TORCH_VARIANT_EXPECTED="${VARIANT}" run_in_env python - <<'PY'
import sys
import os
import pandas
import rdkit
import torch
from rdkit import Chem

print("[verify] python", sys.version.split()[0])
print("[verify] torch", torch.__version__)
print("[verify] torch_cuda", torch.version.cuda)
print("[verify] cuda_available", torch.cuda.is_available())
print("[verify] cuda_device_count", torch.cuda.device_count())
print("[verify] pandas", pandas.__version__)
print("[verify] rdkit", rdkit.__version__)
mol = Chem.MolFromSmiles("[*:1]CC[*:2]")
assert mol is not None
print("[verify] rdkit_smiles", Chem.MolToSmiles(mol, canonical=True))

expected = os.environ.get("TORCH_VARIANT_EXPECTED", "cpu")
if expected.startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit("[error] CUDA PyTorch installation failed: torch.cuda.is_available() is False")
PY
