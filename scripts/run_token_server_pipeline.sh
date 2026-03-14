#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/build_weak_anchor_token_data.sh"
bash "${SCRIPT_DIR}/train_linker_token_diffusion.sh"
bash "${SCRIPT_DIR}/sample_linker_token.sh"
bash "${SCRIPT_DIR}/evaluate_linker_token_generation.sh"
