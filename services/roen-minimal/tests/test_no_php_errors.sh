#!/usr/bin/env bash
# Test: child theme activates without PHP fatals or activation-time errors
set -euo pipefail

# Activate; capture both stdout and stderr so we can detect PHP errors that
# WP-CLI emits to stderr while loading the theme. WP-CLI returns "Success"
# on first activate and "already active" on re-runs — both are healthy states.
ssh server-104 'docker exec roenhandmade-wp wp theme activate roen-minimal --allow-root --path=/var/www/html 2>&1' \
  | tee /tmp/roen-activate.log \
  | grep -qE "Success|already active" || { echo "FAIL: theme did not activate cleanly"; cat /tmp/roen-activate.log; exit 1; }

# Filter out WP-CLI's benign "already active" warning before grepping for PHP errors.
if grep -v -i "already active" /tmp/roen-activate.log | grep -qiE "fatal error|warning:|notice:|deprecated"; then
  echo "FAIL: PHP errors during theme activation"
  grep -v -i "already active" /tmp/roen-activate.log | grep -iE "fatal error|warning:|notice:|deprecated" | head -5
  exit 1
fi

# Hit the homepage and confirm 200 + no PHP error string
HTTP=$(timeout 15 curl -ksS -o /tmp/roen-home.html -w "%{http_code}" https://www.roenhandmade.com/)
[ "$HTTP" = "200" ] || { echo "FAIL: homepage returned $HTTP"; exit 1; }

if grep -qiE "fatal error|warning:|notice:|deprecated" /tmp/roen-home.html; then
  echo "FAIL: PHP error markers found in homepage HTML"
  grep -iE "fatal error|warning:|notice:|deprecated" /tmp/roen-home.html | head -5
  exit 1
fi

echo "PASS: theme activated and homepage rendered without PHP errors"
