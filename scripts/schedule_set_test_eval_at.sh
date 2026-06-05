#!/usr/bin/env bash
# Schedule SET all-datasets test eval at a wall-clock time (default: tomorrow 04:00).
set -euo pipefail

SCHEDULE_AT="${SCHEDULE_AT:-2026-06-05 04:00:00}"
OUT_DIR="${OUT_DIR:-/root/rivermind-data/huangziyi/reseg/output/set/a3-frozen-lasers-set-5w}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_all_datasets_test_eval.sh"
PID_FILE="${OUT_DIR}/scheduled_test_eval.pid"
LOG_FILE="${OUT_DIR}/scheduled_test_eval_launcher.log"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  chmod +x "${RUN_SCRIPT}"
fi

TARGET_EPOCH="$(date -d "${SCHEDULE_AT}" +%s)"
NOW_EPOCH="$(date +%s)"
SLEEP_SEC=$((TARGET_EPOCH - NOW_EPOCH))

if (( SLEEP_SEC < 0 )); then
  echo "[ERROR] SCHEDULE_AT is in the past: ${SCHEDULE_AT} (now=$(date -Is))"
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}")"
  if kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "[INFO] stopping previous scheduler pid=${OLD_PID}"
    kill "${OLD_PID}" 2>/dev/null || true
    sleep 1
  fi
fi

nohup bash -c "
  echo '[SCHEDULER] armed for ${SCHEDULE_AT} (sleep ${SLEEP_SEC}s)' >> '${LOG_FILE}'
  sleep ${SLEEP_SEC}
  echo '[SCHEDULER] firing at '\$(date -Is) >> '${LOG_FILE}'
  OUT_DIR='${OUT_DIR}' WAIT_FOR_MERGED=1 bash '${RUN_SCRIPT}'
  echo '[SCHEDULER] finished at '\$(date -Is) >> '${LOG_FILE}'
" >> "${LOG_FILE}" 2>&1 &

NEW_PID=$!
echo "${NEW_PID}" > "${PID_FILE}"

echo "Scheduled SET all-datasets test eval"
echo "  time      : ${SCHEDULE_AT}"
echo "  sleep     : ${SLEEP_SEC}s (~$((SLEEP_SEC / 3600))h $(( (SLEEP_SEC % 3600) / 60 ))m)"
echo "  pid       : ${NEW_PID}"
echo "  launcher  : ${LOG_FILE}"
echo "  eval log  : ${OUT_DIR}/all_datasets_test_eval.log"
echo ""
echo "Behavior at fire time:"
echo "  1) wait until ${OUT_DIR}/merged_model/config.json exists"
echo "  2) run test eval: LaSeRS, RRSISD, RefSegRS, RISBench, EarthReason"
echo "  (does NOT merge checkpoints — pipeline handles that)"
echo ""
echo "Cancel: kill ${NEW_PID}"
