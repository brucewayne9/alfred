#!/usr/bin/env bash
# Lucius healthcheck — used by 105's claw_monitor.py via SSH.
# Stdout 'ok' = healthy, 'down' = anything else.
# Returns:
#   - 'ok' if hermes-gateway.service is active
#   - 'down' otherwise
# Always exits 0 — the *output* signals state, not the exit code.
# Why: claw_monitor reads the stdout. A non-zero exit would mask the actual state.

export XDG_RUNTIME_DIR=/run/user/$(id -u)

if systemctl --user is-active hermes-gateway.service >/dev/null 2>&1; then
  echo "ok"
else
  echo "down"
fi
exit 0
