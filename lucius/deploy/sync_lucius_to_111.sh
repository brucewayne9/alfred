#!/usr/bin/env bash
# Idempotent rsync from 117 → 111 for the day-one Lucius tool set.
# Driven by lucius/deploy/source_manifest.txt.
# Run from 105 (alfred home).

set -euo pipefail

MANIFEST="$(dirname "$0")/source_manifest.txt"
SRC_HOST="brucewayne9@75.43.156.117"
SRC_DIR="~/.openclaw/workspace/scripts/integrations/"
DST_HOST="brucewayne9@75.43.156.111"
DST_DIR="~/.lucius/workspace/scripts/integrations/"

echo "Sync from ${SRC_HOST}:${SRC_DIR} → ${DST_HOST}:${DST_DIR}"
echo "Manifest: ${MANIFEST}"

# Fetch from 117 → /tmp on 105 (avoid direct 117↔111 SSH agent forwarding)
STAGE=$(mktemp -d)
trap "rm -rf $STAGE" EXIT

while read -r f; do
  [[ -z "$f" || "$f" =~ ^# ]] && continue
  scp "${SRC_HOST}:${SRC_DIR}${f}" "${STAGE}/${f}"
done < "$MANIFEST"

# Push from 105 → 111
rsync -av --checksum --chmod=u=rwx,go= "${STAGE}/" "${DST_HOST}:${DST_DIR}"

# Verify count matches manifest
EXPECTED=$(grep -cv -E '^(#|$)' "$MANIFEST")
ACTUAL=$(ssh "${DST_HOST}" "ls ${DST_DIR}*.py 2>/dev/null | wc -l")
echo "Expected: ${EXPECTED}, Actual on 111: ${ACTUAL}"
[[ "$EXPECTED" -eq "$ACTUAL" ]] || { echo "COUNT MISMATCH"; exit 1; }
echo "Sync OK."
