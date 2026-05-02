#!/usr/bin/env bash
# Test: child theme activates without PHP fatals
set -euo pipefail

ssh server-104 'docker exec roenhandmade-wp wp theme activate roen-minimal --allow-root --path=/var/www/html' \
  | grep -q "Success" || { echo "FAIL: theme did not activate cleanly"; exit 1; }

# Hit the homepage and confirm 200 + no PHP error string
HTTP=$(timeout 15 curl -ksS -o /tmp/roen-home.html -w "%{http_code}" https://www.roenhandmade.com/)
[ "$HTTP" = "200" ] || { echo "FAIL: homepage returned $HTTP"; exit 1; }

if grep -qiE "fatal error|warning:|notice:|deprecated" /tmp/roen-home.html; then
  echo "FAIL: PHP error markers found in homepage HTML"
  grep -iE "fatal error|warning:|notice:|deprecated" /tmp/roen-home.html | head -5
  exit 1
fi

echo "PASS: theme activated and homepage rendered without PHP errors"
