#!/bin/bash
set -euo pipefail

PROJECT_DIR="${1:-$PWD}"
TRUST4_DIR="${PROJECT_DIR}/tools/TRUST4_v1.1.9"

cd "${TRUST4_DIR}"
make -j 4

test -x ./run-trust4
test -x ./trust4
test -f ./human_IMGT+C.fa
test -f ./human_vdjc.list

echo "TRUST4 build/check OK: ${TRUST4_DIR}"
