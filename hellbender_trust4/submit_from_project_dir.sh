#!/bin/bash
set -euo pipefail

PROJECT_DIR="/cluster/pixstor/xudong-lab/wangdu/scTRP"
cd "${PROJECT_DIR}"
mkdir -p logs
PROJECT_DIR="${PROJECT_DIR}" sbatch hellbender_trust4/submit_trust4_cd8_array.sbatch
