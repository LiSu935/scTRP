#!/bin/bash
#SBATCH --mem=50G
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --job-name=trust4_cd8
#SBATCH -p xudong-lab
#SBATCH -A xudong-lab
#SBATCH --array=1-15
#SBATCH --output=logs/trust4_%A_%a.out
#SBATCH --error=logs/trust4_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
MANIFEST="${PROJECT_DIR}/hellbender_trust4/trust4_cd8_manifest.tsv"
TRUST4_DIR="${PROJECT_DIR}/tools/TRUST4_v1.1.9"
OUT_ROOT="${PROJECT_DIR}/trust4_cd8_outputs"
THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "${OUT_ROOT}" "${PROJECT_DIR}/logs"

line="$(awk -v task="${SLURM_ARRAY_TASK_ID}" 'NR == task + 1 {print; exit}' "${MANIFEST}")"
if [[ -z "${line}" ]]; then
  echo "No manifest row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

IFS=$'\t' read -r sample srr r1_barcode_umi r2_cdna <<< "${line}"

R1="${PROJECT_DIR}/${r1_barcode_umi}"
R2="${PROJECT_DIR}/${r2_cdna}"
OUT_DIR="${OUT_ROOT}/${sample}"
PREFIX="${srr}_${sample}_CD8_TRUST4"

test -s "${R1}"
test -s "${R2}"
test -x "${TRUST4_DIR}/run-trust4"
test -f "${TRUST4_DIR}/human_IMGT+C.fa"

mkdir -p "${OUT_DIR}"

echo "[$(date)] sample=${sample} srr=${srr}"
echo "R1 barcode/UMI: ${R1}"
echo "R2 cDNA:        ${R2}"

"${TRUST4_DIR}/run-trust4" \
  -f "${TRUST4_DIR}/human_IMGT+C.fa" \
  --ref "${TRUST4_DIR}/human_IMGT+C.fa" \
  -u "${R2}" \
  --barcode "${R1}" \
  --UMI "${R1}" \
  --readFormat "r1:0:-1,bc:0:15,um:16:-1" \
  -t "${THREADS}" \
  -o "${PREFIX}" \
  --od "${OUT_DIR}" \
  --clean 1

echo "[$(date)] done: ${OUT_DIR}"
