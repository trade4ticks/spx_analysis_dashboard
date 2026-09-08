#!/bin/bash
# Steps 3 and 6 in one downtime window, on the VPS.
#
#   bash scripts/run_scan_checks.sh
#
# THE SERVICE MUST COME BACK. This stops spx-live, and the one unacceptable
# outcome is leaving it stopped -- a dropped SSH session, a hung socket or a
# killed terminal all have to end with the tape running. Hence the trap on
# every exit path, `timeout` around each check, and detaching from the shell
# that launched it.
#
# THE TWO CHECKS ARE INDEPENDENT. Step 3 exercises the tier against the live
# upstream; step 6 measures ring memory at 600 symbols. Step 6 is the one that
# decides whether the growth rule delivered, so a failure in step 3 must not
# skip it -- the downtime has already been paid for by then.

set -u
LOG=/tmp/scan_checks.log
cd /spx_analysis_dashboard || exit 1
PY=./.venv/bin/python

restart() {
  echo "" >>"$LOG"
  echo "== restarting spx-live ==" >>"$LOG"
  systemctl start spx-live >>"$LOG" 2>&1
  sleep 6
  curl -s --max-time 5 http://127.0.0.1:8001/status >>"$LOG" 2>&1
  echo "" >>"$LOG"
  echo "DONE $(date)" >>"$LOG"
}
trap restart EXIT INT TERM

: >"$LOG"
echo "start $(date)" >>"$LOG"
echo "== stopping spx-live ==" >>"$LOG"
systemctl stop spx-live >>"$LOG" 2>&1
sleep 3

echo "" >>"$LOG"
echo "######## STEP 3: the tier against the real upstream ########" >>"$LOG"
timeout 400 "$PY" scripts/check_scan_live.py \
  --symbols 200 --seconds 120 >>"$LOG" 2>&1
echo "step 3 exit: $?" >>"$LOG"

echo "" >>"$LOG"
echo "######## STEP 6: ring memory at 600 symbols ########" >>"$LOG"
# SEVEN MINUTES, not two, and the length is the whole validity of the number.
# A ring grows until it holds a full retention window; measured over two
# minutes against a six-minute retention, every symbol is still sized for two
# minutes of data and the saving reads about three times better than it is.
# The run has to outlast LIVE_SCAN_RETAIN_S before ring MB means anything.
timeout 600 "$PY" scripts/measure_scan_capacity.py \
  --steps 600 --seconds 420 --warmup 10 --channels T \
  --out /tmp/scan_rings_600.json >>"$LOG" 2>&1
echo "step 6 exit: $?" >>"$LOG"

exit 0
