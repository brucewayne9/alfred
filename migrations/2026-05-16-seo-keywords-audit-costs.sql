-- migrations/2026-05-16-seo-keywords-audit-costs.sql
-- Alfred's SEO — Phase 1 schema add for keyword punch list, audit issues,
-- and DataForSEO cost tracking. Runs against alfred_main.

BEGIN;

-- ─────────────────────────────────────────────────────────────────────────────
-- Curated keyword punch list. One row per (site, keyword) we're actively
-- tracking. Distinct from seo_queries (raw GSC dump) — these are the keywords
-- we DECIDED to attack, with target URL + intent + difficulty enriched from
-- DataForSEO Labs.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS seo_keywords (
    id                  BIGSERIAL PRIMARY KEY,
    site_id             INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    keyword             TEXT NOT NULL,
    target_url          TEXT,                          -- which page should rank for this kw
    search_volume       INTEGER,                       -- monthly searches (DFS Keywords Data)
    keyword_difficulty  INTEGER,                       -- 0–100 (DFS Labs KD)
    cpc                 NUMERIC(10,4),                 -- $ if anyone is bidding on it
    competition         NUMERIC(6,4),                  -- 0–1 (Google Ads competition index)
    competition_level   VARCHAR(16),                   -- LOW / MEDIUM / HIGH
    search_intent       VARCHAR(32),                   -- informational | commercial | transactional | navigational
    current_rank        NUMERIC(6,2),                  -- best known position
    current_rank_url    TEXT,                          -- which URL is ranking (may ≠ target_url)
    rank_source         VARCHAR(16),                   -- gsc | serp_api | none
    rank_checked_at     TIMESTAMPTZ,
    priority            SMALLINT DEFAULT 2,            -- 1=hot, 2=normal, 3=long-tail
    status              VARCHAR(16) DEFAULT 'active',  -- active | paused | archived
    discovered_at       TIMESTAMPTZ DEFAULT now(),
    last_refreshed_at   TIMESTAMPTZ DEFAULT now(),
    meta_payload        JSONB,                         -- raw DFS response for forensics
    UNIQUE (site_id, keyword)
);
CREATE INDEX IF NOT EXISTS idx_seo_keywords_site_status   ON seo_keywords (site_id, status);
CREATE INDEX IF NOT EXISTS idx_seo_keywords_site_priority ON seo_keywords (site_id, priority);
CREATE INDEX IF NOT EXISTS idx_seo_keywords_target_url    ON seo_keywords (site_id, target_url);

-- ─────────────────────────────────────────────────────────────────────────────
-- Granular audit findings. One row per distinct issue on a page (e.g. missing
-- alt text on image X, broken link to URL Y). seo_pages has rollup status
-- fields; this table has the per-issue grain we need for the dashboard.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS seo_audit_issues (
    id                  BIGSERIAL PRIMARY KEY,
    site_id             INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    page_url            TEXT NOT NULL,
    issue_type          VARCHAR(48) NOT NULL,          -- missing_alt | missing_meta_desc | missing_schema | broken_link | redirect_chain | thin_content | slow_lcp | duplicate_title | h1_missing | etc.
    severity            VARCHAR(16) NOT NULL,          -- critical | warning | info
    detail              TEXT,                          -- human-readable summary
    detail_payload      JSONB,                         -- structured (e.g. {image_src, broken_target, current_meta})
    first_detected_at   TIMESTAMPTZ DEFAULT now(),
    last_detected_at    TIMESTAMPTZ DEFAULT now(),
    fixed_at            TIMESTAMPTZ,
    fix_method          VARCHAR(32),                   -- manual | auto_alt | auto_schema | wp_rewrite | etc.
    -- detail can be long; use issue_fingerprint (sha-ish or short stable id) for dedup
    issue_fingerprint   VARCHAR(64) NOT NULL,
    UNIQUE (site_id, page_url, issue_type, issue_fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_seo_audit_open    ON seo_audit_issues (site_id, fixed_at) WHERE fixed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_seo_audit_type    ON seo_audit_issues (site_id, issue_type);

-- ─────────────────────────────────────────────────────────────────────────────
-- DataForSEO (and other paid SEO API) spend tracker. Every paid call writes
-- one row. Drives the "spend this month" widget on the dashboard + budget
-- alerts before we run out.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS seo_api_costs (
    id          BIGSERIAL PRIMARY KEY,
    api_name    VARCHAR(32) NOT NULL,                 -- dataforseo | psi | gsc | ga4
    endpoint    VARCHAR(128),                          -- e.g. /v3/keywords_data/google_ads/search_volume/live
    cost_usd    NUMERIC(10,6) DEFAULT 0,
    site_id     INTEGER REFERENCES seo_sites(id) ON DELETE SET NULL,
    purpose     VARCHAR(64),                           -- keyword_discovery | site_audit | rank_check | etc.
    meta        JSONB,
    called_at   TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_seo_api_costs_called ON seo_api_costs (called_at DESC);
CREATE INDEX IF NOT EXISTS idx_seo_api_costs_api    ON seo_api_costs (api_name, called_at DESC);

COMMIT;
