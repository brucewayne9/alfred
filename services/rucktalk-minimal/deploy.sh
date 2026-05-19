#!/usr/bin/env bash
# Deploy rucktalk-minimal child theme to rt-wordpress on server-100.
# Idempotent — safe to re-run.
#
# T3 ACTION per CLAUDE.md — requires Mike's explicit go before EVERY run.
#
# Mirrors the proven services/roen-minimal/deploy.sh pattern. Tar-pipe is
# used because snap-installed Docker on 104/100 sandboxes /tmp, so
# `docker cp /tmp/...` fails — piping a tar stream through `docker exec`
# stdin works regardless of where the host staged files.

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_HOST="server-100"
STAGE_DIR="/tmp/rucktalk-minimal"
CONTAINER="rt-wordpress"
WP_PATH="/var/www/html"
TARGET="${WP_PATH}/wp-content/themes/rucktalk-minimal"

echo "==> rsync source to ${SSH_HOST}:${STAGE_DIR}"
rsync -av --delete \
  --exclude 'tests/' \
  --exclude 'deploy.sh' \
  --exclude 'README.md' \
  --exclude '.DS_Store' \
  --exclude '*.bak' \
  --exclude '*.bak-*' \
  "${SRC_DIR}/" "${SSH_HOST}:${STAGE_DIR}/"

echo "==> tar-pipe into ${CONTAINER}:${TARGET}"
ssh "${SSH_HOST}" "
  set -e
  timeout 30 docker exec ${CONTAINER} mkdir -p ${TARGET}
  tar -C ${STAGE_DIR} -cf - . | timeout 60 docker exec -i ${CONTAINER} tar -C ${TARGET} -xf -
  timeout 30 docker exec ${CONTAINER} chown -R www-data:www-data ${TARGET}
"

echo "==> done. Theme files at ${CONTAINER}:${TARGET}"
echo "==> theme NOT activated automatically. To activate:"
echo "    ssh ${SSH_HOST} \"docker exec ${CONTAINER} wp theme activate rucktalk-minimal --allow-root\""
echo "==> rollback (sonaar-child is empty boilerplate, safe):"
echo "    ssh ${SSH_HOST} \"docker exec ${CONTAINER} wp theme activate sonaar-child --allow-root\""
