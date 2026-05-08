#!/usr/bin/env bash
# Pre-flight: verify TELEGRAM_BOT_TOKEN resolves to @Luciuslabsbot.
# Per memory `feedback_verify_bot_identity.md`: never start without this check.
# Non-zero exit blocks systemd unit's ExecStartPre.

set -euo pipefail

ENV_FILE="${HERMES_ENV_FILE:-/home/brucewayne9/.hermes/.env}"
EXPECTED_USERNAME="Luciuslabsbot"
EXPECTED_BOT_ID="8750983299"

# Source token from .env without leaking
TOKEN=$(grep -E '^TELEGRAM_BOT_TOKEN=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")
[[ -n "$TOKEN" ]] || { echo "[identity-guard] TELEGRAM_BOT_TOKEN missing in $ENV_FILE"; exit 2; }

RESPONSE=$(curl -sS --max-time 10 "https://api.telegram.org/bot${TOKEN}/getMe")
OK=$(echo "$RESPONSE" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('ok'))" 2>/dev/null || echo "False")
USERNAME=$(echo "$RESPONSE" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('result',{}).get('username',''))" 2>/dev/null || echo "")
BOT_ID=$(echo "$RESPONSE" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('result',{}).get('id',''))" 2>/dev/null || echo "")

if [[ "$OK" != "True" ]]; then
  echo "[identity-guard] Telegram getMe FAILED: ${RESPONSE:0:200}"
  exit 3
fi
if [[ "$USERNAME" != "$EXPECTED_USERNAME" ]]; then
  echo "[identity-guard] BOT MISMATCH — got @${USERNAME}, expected @${EXPECTED_USERNAME}"
  exit 4
fi
if [[ "$BOT_ID" != "$EXPECTED_BOT_ID" ]]; then
  echo "[identity-guard] BOT_ID MISMATCH — got ${BOT_ID}, expected ${EXPECTED_BOT_ID}"
  exit 5
fi
echo "[identity-guard] OK — @${USERNAME} (id ${BOT_ID})"
exit 0
