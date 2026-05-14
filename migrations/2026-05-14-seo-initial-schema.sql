-- migrations/2026-05-14-seo-initial-schema.sql
-- Phase 1 SEO foundation schema. Runs against alfred_main.

BEGIN;

CREATE TABLE IF NOT EXISTS seo_sites (
    id                  SERIAL PRIMARY KEY,
    slug                VARCHAR(64) UNIQUE NOT NULL,
    domain              VARCHAR(255) NOT NULL,
    display_name        VARCHAR(255) NOT NULL,
    wp_rest_url         VARCHAR(255) NOT NULL,
    wp_username         VARCHAR(255),
    wp_app_password     TEXT,                          -- encrypted at app layer
    gsc_property        VARCHAR(255),
    ga4_property_id     VARCHAR(64),
    brand_profile_path  VARCHAR(255),
    business_type       VARCHAR(32) DEFAULT 'Organization',
    status              VARCHAR(32) DEFAULT 'active',
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS seo_queries (
    id            BIGSERIAL PRIMARY KEY,
    site_id       INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    query         TEXT NOT NULL,
    position      NUMERIC(6,2),
    impressions   INTEGER DEFAULT 0,
    clicks        INTEGER DEFAULT 0,
    ctr           NUMERIC(6,4),
    captured_at   DATE NOT NULL,
    UNIQUE (site_id, query, captured_at)
);
CREATE INDEX IF NOT EXISTS idx_seo_queries_site_date ON seo_queries (site_id, captured_at DESC);

CREATE TABLE IF NOT EXISTS seo_pages (
    id                  BIGSERIAL PRIMARY KEY,
    site_id             INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    url                 TEXT NOT NULL,
    page_type           VARCHAR(64),                  -- product | post | page | category
    indexed_at          TIMESTAMPTZ,
    last_audit_at       TIMESTAMPTZ,
    schema_status       VARCHAR(32),                  -- ok | invalid | missing
    meta_status         VARCHAR(32),                  -- ok | missing
    cwv_lcp_ms          INTEGER,
    cwv_cls             NUMERIC(6,3),
    cwv_inp_ms          INTEGER,
    organic_sessions    INTEGER DEFAULT 0,
    conversions         INTEGER DEFAULT 0,
    last_seen_at        TIMESTAMPTZ DEFAULT now(),
    UNIQUE (site_id, url)
);
CREATE INDEX IF NOT EXISTS idx_seo_pages_site ON seo_pages (site_id);

CREATE TABLE IF NOT EXISTS seo_briefs (
    id                BIGSERIAL PRIMARY KEY,
    site_id           INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    topic             TEXT NOT NULL,
    content_type      VARCHAR(32),                    -- product_enrichment | cluster | blog | ad_landing
    target_keywords   JSONB,
    audience          TEXT,
    status            VARCHAR(32) DEFAULT 'queued',
    brief_payload     JSONB,
    source_signal     JSONB,
    created_at        TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS seo_pending (
    id              BIGSERIAL PRIMARY KEY,
    site_id         INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    brief_id        BIGINT REFERENCES seo_briefs(id) ON DELETE SET NULL,
    content_type    VARCHAR(32),
    title           TEXT,
    body_payload    JSONB,
    source_signal   JSONB,
    status          VARCHAR(32) DEFAULT 'pending',
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS seo_decided (
    id              BIGSERIAL PRIMARY KEY,
    site_id         INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    brief_id        BIGINT REFERENCES seo_briefs(id) ON DELETE SET NULL,
    content_type    VARCHAR(32),
    title           TEXT,
    body_payload    JSONB,
    decided_at      TIMESTAMPTZ DEFAULT now(),
    decided_by      VARCHAR(64),
    outcome         VARCHAR(32),                       -- approved | rejected
    wp_post_id      BIGINT,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS seo_backlinks (
    id              BIGSERIAL PRIMARY KEY,
    site_id         INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    source_url      TEXT NOT NULL,
    target_url      TEXT,
    anchor_text     TEXT,
    first_seen      TIMESTAMPTZ DEFAULT now(),
    last_seen       TIMESTAMPTZ DEFAULT now(),
    lost_at         TIMESTAMPTZ,
    UNIQUE (site_id, source_url, target_url)
);

CREATE TABLE IF NOT EXISTS seo_haro_opps (
    id                   BIGSERIAL PRIMARY KEY,
    site_id              INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    source_email_id      VARCHAR(255),
    query_text           TEXT,
    deadline             TIMESTAMPTZ,
    draft_pitch_payload  JSONB,
    status               VARCHAR(32) DEFAULT 'pending',
    response_sent_at     TIMESTAMPTZ,
    created_at           TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS seo_rankings_daily (
    id              BIGSERIAL PRIMARY KEY,
    site_id         INTEGER NOT NULL REFERENCES seo_sites(id) ON DELETE CASCADE,
    query           TEXT NOT NULL,
    position        NUMERIC(6,2),
    captured_at     DATE NOT NULL,
    UNIQUE (site_id, query, captured_at)
);

COMMIT;
