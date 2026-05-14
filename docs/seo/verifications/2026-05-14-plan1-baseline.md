# Plan 1 Roen baseline backfill — verification

**Status:** Plan 1 is **code-complete**. Roen 14-day backfill is **BLOCKED — pending Mike's OAuth consent.**

---

## What's deployed

All 32 tasks of Plan 1 have shipped to `feat/lucius-hermes-on-111`:

- `alfred-seo` WP plugin deployed to roenhandmade.com (Tasks 1-19)
- Alfred orchestrator skeleton on 105 (Tasks 20-25): 9 `seo_*` tables, SQLAlchemy models, site registry, FastAPI `/admin/seo` routes, Roen registered as Site #1
- Google API OAuth helper (Task 26)
- 4 ingest modules + CLIs (Tasks 27-30): GSC, GA4, PageSpeed Insights CWV, backlinks Layer 1
- 4 systemd timers, all enabled + active (Task 31):
  - `alfred-seo-gsc-sync.timer` — 09:00 UTC daily (05:00 EDT)
  - `alfred-seo-ga4-sync.timer` — 09:05 UTC daily (05:05 EDT)
  - `alfred-seo-cwv-sync.timer` — 10:00 UTC daily (06:00 EDT)
  - `alfred-seo-backlinks-sync.timer` — 11:00 UTC daily (07:00 EDT)

## Blocker — what Mike needs to do

The 14-day Roen baseline backfill requires Google OAuth consent + a PSI API key. This is a one-time interactive setup that Alfred (running headless on 105) cannot complete.

Steps (full instructions: `docs/seo/OAUTH_SETUP.md`):

1. Create an OAuth client at https://console.cloud.google.com/apis/credentials
   - Type: Desktop app
   - Name: "Alfred SEO Desktop"
   - Scopes: `webmasters.readonly`, `analytics.readonly`
2. Enable in the same project: Search Console API, Google Analytics Data API, PageSpeed Insights API
3. Create a PageSpeed Insights API key (restricted to that API)
4. Paste into `/home/aialfred/alfred/config/.env`:
   ```
   SEO_GOOGLE_OAUTH_CLIENT_ID=...apps.googleusercontent.com
   SEO_GOOGLE_OAUTH_CLIENT_SECRET=...
   SEO_PSI_API_KEY=AIza...
   ```
5. (Optional) Set `ROEN_GA4_PROPERTY_ID` if not already discovered + register it on the Roen site row via the seo_sites table.
6. SSH-tunnel + run the consent flow ONCE from a Mac with a browser:
   ```
   ssh -L 8080:localhost:8080 server-105
   cd /home/aialfred/alfred
   ./venv/bin/python -m integrations.google_seo.oauth authorize
   ```
   A browser opens, click through Google consent, refresh token persists at `data/seo/google_oauth_token.json`.

## Once OAuth lands — backfill commands

```bash
# 14 days of GSC
for d in $(seq 1 14); do
  date=$(date -u -d "$d days ago" +%Y-%m-%d)
  PYTHONPATH=/home/aialfred/alfred ./venv/bin/python scripts/seo_gsc_sync.py --date "$date"
done

# 14 days of GA4
for d in $(seq 1 14); do
  date=$(date -u -d "$d days ago" +%Y-%m-%d)
  PYTHONPATH=/home/aialfred/alfred ./venv/bin/python scripts/seo_ga4_sync.py --date "$date"
done

# CWV (one shot)
PYTHONPATH=/home/aialfred/alfred ./venv/bin/python scripts/seo_cwv_sync.py --limit 20

# Backlinks (one shot — empty under Phase 1 stub, real data ships in Plan 2)
PYTHONPATH=/home/aialfred/alfred ./venv/bin/python scripts/seo_backlinks_sync.py
```

## Verification checklist (run after backfill)

- [ ] Dashboard `https://aialfred.groundrushcloud.com/admin/seo` shows Roen row with GSC + GA4 + CWV columns populated
- [ ] Site detail `https://aialfred.groundrushcloud.com/admin/seo/sites/roen` shows GSC property + GA4 property + brand profile path
- [ ] All 4 systemd timers active (verified pre-backfill):
  ```
  systemctl list-timers --all | grep alfred-seo
  ```
- [ ] DB spot-check:
  ```bash
  psql -h localhost -U alfred -d alfred_main -c "
    SELECT COUNT(*) AS query_rows FROM seo_queries WHERE site_id=(SELECT id FROM seo_sites WHERE slug='roen');
    SELECT COUNT(*) AS page_rows FROM seo_pages WHERE site_id=(SELECT id FROM seo_sites WHERE slug='roen');
    SELECT COUNT(*) AS backlink_rows FROM seo_backlinks WHERE site_id=(SELECT id FROM seo_sites WHERE slug='roen');
  "
  ```
  Expected: query_rows >= 50, page_rows >= 10, backlink_rows >= 0.

## Counts after backfill — fill in when complete

- Date completed: _PENDING_
- seo_queries: _PENDING_
- seo_pages: _PENDING_
- seo_backlinks: _PENDING_
- CWV scanned URLs: _PENDING_

## Sign-off

When all four checklist items pass and counts are filled in, Plan 1 is fully done. Plan 2 (content engine) starts next.
