#!/usr/bin/env bash
# Deploy the roen-minimal child theme to roenhandmade.com.
# Idempotent — safe to re-run.

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_HOST="server-104"
STAGE_DIR="/tmp/roen-minimal"
CONTAINER="roenhandmade-wp"
WP_PATH="/var/www/html"
TARGET="${WP_PATH}/wp-content/themes/roen-minimal"

echo "==> rsync source to ${SSH_HOST}:${STAGE_DIR}"
rsync -av --delete \
  --exclude 'tests/' \
  --exclude 'deploy.sh' \
  --exclude 'README.md' \
  --exclude '.DS_Store' \
  "${SRC_DIR}/" "${SSH_HOST}:${STAGE_DIR}/"

echo "==> tar-pipe into ${CONTAINER}:${TARGET}"
# Snap-installed Docker on 104 sandboxes /tmp, so 'docker cp /tmp/...' fails.
# Pipe a tar stream through 'docker exec' stdin instead — works regardless
# of where the host staged files.
ssh "${SSH_HOST}" "
  set -e
  timeout 30 docker exec ${CONTAINER} mkdir -p ${TARGET}
  tar -C ${STAGE_DIR} -cf - . | timeout 60 docker exec -i ${CONTAINER} tar -C ${TARGET} -xf -
  timeout 30 docker exec ${CONTAINER} chown -R www-data:www-data ${TARGET}
"

echo "==> verify theme is recognized"
ssh "${SSH_HOST}" "timeout 20 docker exec ${CONTAINER} wp theme list --allow-root --path=${WP_PATH} --format=csv --fields=name,status,version" \
  | grep -q '^roen-minimal' || { echo "ERROR: roen-minimal not seen by WP-CLI"; exit 1; }

echo "==> flush WP cache"
ssh "${SSH_HOST}" "timeout 20 docker exec ${CONTAINER} wp cache flush --allow-root --path=${WP_PATH}" || true

echo "==> done. To activate: ssh ${SSH_HOST} 'docker exec ${CONTAINER} wp theme activate roen-minimal --allow-root --path=${WP_PATH}'"
