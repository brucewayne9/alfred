#!/usr/bin/env bash
# Deploy alfred-seo plugin to roenhandmade-wp.
# Idempotent — safe to re-run. First site only; multi-site loop arrives in Plan 3.

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_HOST="${SSH_HOST:-server-104}"
CONTAINER="${CONTAINER:-roenhandmade-wp}"
WP_PATH="${WP_PATH:-/var/www/html}"
STAGE_DIR="/tmp/alfred-seo"
TARGET="${WP_PATH}/wp-content/plugins/alfred-seo"

echo "==> rsync source to ${SSH_HOST}:${STAGE_DIR}"
rsync -av --delete \
  --exclude 'tests/' \
  --exclude 'deploy.sh' \
  --exclude 'phpunit.xml.dist' \
  --exclude '.DS_Store' \
  "${SRC_DIR}/" "${SSH_HOST}:${STAGE_DIR}/"

echo "==> tar-pipe into ${CONTAINER}:${TARGET}"
ssh "${SSH_HOST}" "
  set -e
  timeout 30 docker exec ${CONTAINER} mkdir -p ${TARGET}
  tar -C ${STAGE_DIR} -cf - . | timeout 60 docker exec -i ${CONTAINER} tar -C ${TARGET} -xf -
  timeout 30 docker exec ${CONTAINER} chown -R www-data:www-data ${TARGET}
"

echo "==> activate plugin via wp-cli"
ssh "${SSH_HOST}" "timeout 30 docker exec ${CONTAINER} wp plugin activate alfred-seo --allow-root --path=${WP_PATH}"

echo "==> flush rewrite rules"
ssh "${SSH_HOST}" "timeout 30 docker exec ${CONTAINER} wp rewrite flush --allow-root --path=${WP_PATH}"

echo "==> flush WP cache"
ssh "${SSH_HOST}" "timeout 30 docker exec ${CONTAINER} wp cache flush --allow-root --path=${WP_PATH} || true"

echo "==> done"
