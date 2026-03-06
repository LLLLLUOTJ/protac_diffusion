#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/build_weak_anchor_data.sh"
"${SCRIPT_DIR}/train_linker_diffusion.sh"
"${SCRIPT_DIR}/sample_linker.sh"
