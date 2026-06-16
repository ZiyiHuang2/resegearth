#!/usr/bin/env bash
# Smoke alias: pure TCPD-only train for 1 step (delegates to tcpd_train_pure.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=test4_common.sh
source "${SCRIPT_DIR}/test4_common.sh"

export MAX_STEPS="${MAX_STEPS:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-${TEST4_OUTPUT_ROOT}/tcpd-only-smoke-${MAX_STEPS}steps}"
exec bash "${SCRIPT_DIR}/tcpd_train_pure.sh"
