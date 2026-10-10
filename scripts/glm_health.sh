#!/usr/bin/env bash
# Phase 1 companion: probe the CPA/GLM endpoint every 10 s while a test runs. Usage:
#   PLAUDIT_SUMMARY_API_KEY=... scripts/glm_health.sh http://oracle-host:8000/v1 glm-5.3-flash
set -u
BASE=${1:?base url}; MODEL=${2:-glm-5.3-flash}
while true; do
  t0=$(date +%s.%N)
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 30 -H "Authorization: Bearer ${PLAUDIT_SUMMARY_API_KEY:-}" \
    -H 'Content-Type: application/json' "$BASE/chat/completions" \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":5,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}")
  printf '%s  GLM %s  %.1fs  MemAvailable %s MB\n' "$(date +%T)" "$code" "$(echo "$(date +%s.%N) - $t0" | bc)" \
    "$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 ))"
  sleep 10
done
