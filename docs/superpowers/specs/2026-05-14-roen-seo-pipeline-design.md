# Roen-First SEO Pipeline — Design Spec

**Date**: 2026-05-14
**Owner**: Mike Johnson
**Status**: Approved design, ready for implementation plan
**Guinea pig site**: roenhandmade.com
**Pipeline scope (Phase 1)**: Roen Handmade, LoovaCast, LumaBot, AG Entertainment, RuckTalk, AIROI, GRL

## Context

Mike runs seven WordPress (and adjacent) properties: Roen Handmade, LoovaCast, LumaBot, AG Entertainment, RuckTalk, AIROI, and GRL. None of them have a meaningful SEO foundation today. Roen specifically is a 93-product e-commerce site with zero schema, no Open Graph tags, no meta descriptions, and a broken sitemap chain — a clean slate.

This spec defines a build that delivers three things in a single cohesive system:

1. **Foundation**: every page on every site has correct schema (Product, Article, Organization, BreadcrumbList, FAQ, LocalBusiness, Podcast, SoftwareApplication, Service, Event, WebSite+SearchAction as applicable), Open Graph + Twitter Cards, meta descriptions, image alt text, internal linking, and a valid sitemap.
2. **Tracking**: Search Console + GA4 + Core Web Vitals + backlinks pulled daily into a cross-site dashboard. Every content decision targets a real signal, not a guess.
3. **Content engine**: AI-generated content in four types (product enrichment, cluster/pillar pages, blog posts, ad landing pages) at a paced 2-3 pieces/week per site, flowing through a single approval queue Mike already understands from the Roen social pipeline.

Plus off-page and supporting work: HARO/Connectively opportunity surface (semi-active backlink building), Google Business Profile management (Roen Atlanta local SEO), passive backlink monitoring, Core Web Vitals alerting.

The system is internal-first. Roen is the guinea pig; the other six Mike-owned sites are second-class tenants from day one. The architecture is designed to bolt on multi-tenant SaaS (signup, billing, client-facing onboarding) as a Phase 2 if and only if the engine proves itself on Mike's sites.

## Goal

Make Roen and the other Mike-owned sites measurably more findable in 90 days via real organic search traction. Build the engine that makes maintaining that traction routine — not a special project Mike has to babysit.

Concretely:
- Roen ranks for "evil eye bracelet meaning" (and similar high-intent long-tails) top 10 by Day 90
- Roen indexed pages > 100 by Day 60 (currently ~95)
- Every Roen product page has rich result eligibility in Google
- Approval queue averages 5 min/day of Mike's time across all 7 sites combined

## North star principles

1. **Pipeline-first, Roen-first.** Every component is generic. Roen is configured as Site #1. Adding another site = config file + brand profile + voice examples + WP plugin install. Zero code changes.
2. **Same approval pattern Mike already trusts.** `/admin/seo/pending` is a clone of `/admin/roen/social-pending`. Nothing new to learn.
3. **Free-data-driven targeting.** Content briefs come from Search Console gaps — queries each site already appears for but doesn't click. Cheaper, sharper, real.
4. **The site stays up no matter what.** The WP plugin has sane local fallbacks for every Alfred dependency. Alfred going down never breaks a site.
5. **Never silent-drop a draft.** Content that fails generation, fails validation, or fails to publish stays in the queue with the failure reason. Never lost, never forgotten.

## Scope — what's in

**Foundation (per-site WP plugin):**
- JSON-LD schema: Product, Offer, AggregateRating, Brand, CollectionPage, ItemList, Article, Person, Organization, BreadcrumbList, FAQPage, LocalBusiness, Podcast, PodcastEpisode, AudioObject, SoftwareApplication, Service, Event, Review, WebSite+SearchAction
- Open Graph + Twitter Card auto-generation, override via Alfred-set custom field
- Meta description override (from Alfred custom field, fallback to excerpt)
- Multi-sitemap index: products, pages, posts, categories. Auto-pings GSC + Bing on update.
- Image alt text auto-fill on upload (calls Alfred's vision endpoint, local fallback to filename)
- Internal linking filter on `the_content` per a per-site phrase→URL map
- `robots.txt` management (appends WC-safe directives, never overrides custom rules)
- REST API for Alfred orchestrator: `/audit`, `/content`, `/meta`, `/internal-links`, `/sitemap-ping`
- ~600-900 LOC PHP, single plugin codebase, deployed identically to all sites

**Tracking (Alfred orchestrator on 105):**
- Google Search Console daily sync per site (queries, positions, impressions, clicks, CTR)
- GA4 daily sync (organic sessions, conversions per page)
- Core Web Vitals daily sync via PageSpeed Insights API (top 20 pages per site)
- Passive backlink monitor via GSC top-linking-sites (daily diff)
- Cross-site dashboard at `/admin/seo`
- Per-site deep view at `/admin/seo/sites/<slug>`
- Daily Telegram digest at 8am ET (queue depth, ranking changes, new/lost backlinks, CWV alerts)

**Content engine (Alfred orchestrator on 105):**
- Site registry: DB-backed, supports multi-tenancy from day one
- Brand voice profiles per site: YAML schema + 5-10 few-shot voice examples per site
- Brief generator: picks topics from GSC gaps, drafts brief with target keywords + audience + content type
- Content writer: 4 content types (product enrichment, cluster, blog, ad landing), uses `kimi-k2.6:cloud` for long-form, `gemma4:31b-cloud` for brief/short
- Post-validation: never_say regex scan, Flesch readability check, target keyword presence
- Approval queue at `/admin/seo/pending` — content drafts, HARO pitches, GMB drafts in one surface
- Publisher: approved content → WP REST API → plugin auto-injects schema/meta on next render

**Off-page (Alfred orchestrator on 105):**
- HARO/Connectively inbox monitor (every 15 min)
- Per-site relevance filter, pitch drafter using brand voice
- Pitches enqueue in `/admin/seo/pending` tagged "haro"
- Google Business Profile for Roen (hours, photos, posts, Q&A) — Atlanta local SEO

**Schema validation:**
- Daily run of Google Rich Results Test API on 5 sample pages per site
- Failures alert via Telegram, surfaced in `/admin/seo`

## Scope — what's out (Phase 2 / deferred)

- **Active backlink outreach** (competitor backlink analysis + cold outreach automation) — high effort, spam-adjacent, defer
- **Pinterest auto-pin** (jewelry-specific Roen feature) — gated on Pinterest business account setup
- **Multi-tenant SaaS shell** (signup, Stripe billing, client onboarding wizard, marketing site) — only build if the engine proves itself on Mike's sites first
- **YouTube SEO** (RuckTalk videos, AG hype) — out of Phase 1
- **AMP, hreflang, programmatic SEO at scale** — not needed for current sites

## Architecture

Two distinct codebases, one shared config layer, four data sources.

```
┌─────────────────────────────────────────────────────────────────────┐
│  ALFRED ORCHESTRATOR (105) — central brain                          │
│  /admin/seo                                                         │
│                                                                     │
│  • Site registry (DB-backed, multi-tenant-ready)                    │
│  • Brand voice profiles + few-shot examples per site                │
│  • Content engine (4 types)                                         │
│  • GSC + GA4 + CWV + backlinks sync daemons                         │
│  • Approval queue (clone of Roen social pattern)                    │
│  • Cross-site dashboard                                             │
│  • HARO monitor + pitch drafter                                     │
│  • GMB integration (Roen)                                           │
│  • Daily Telegram digest                                            │
└─────────────────────────────────────────────────────────────────────┘
        │                          ▲
        │ WP REST API              │ GSC + GA4 + PSI + GMB APIs
        │ (publish content)        │ (read-only ingest)
        ▼                          │
┌─────────────────────────────────────────────────────────────────────┐
│  PER-SITE WP PLUGIN — "alfred-seo" (thin, frontend-only)            │
│  Installed on each of the 7 sites.                                  │
│                                                                     │
│  • JSON-LD schema injection per page type                           │
│  • OG + Twitter Card tags                                           │
│  • Meta description (from Alfred custom field)                      │
│  • Sitemap generation                                               │
│  • Image alt text auto-fill on upload                               │
│  • Internal linking filter                                          │
│  • REST endpoints Alfred orchestrator calls                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### Per-site WP plugin (`alfred-seo`)

Single plugin codebase deployed identically to each site. Per-site differences come from a single options row Alfred pushes (brand profile fingerprint, internal-link map, etc.).

**File layout (`services/alfred-seo/`):**
```
alfred-seo.php             # Main plugin file
inc/
  schema/                  # JSON-LD builders per page type
    product.php
    article.php
    organization.php
    breadcrumb.php
    faq.php
    localbusiness.php
    podcast.php
    softwareapplication.php
    service.php
    event.php
    review.php
    website.php            # WebSite + SearchAction
  open-graph.php           # OG + Twitter Card head tags
  meta.php                 # Meta description override
  sitemap.php              # Multi-sitemap index + per-type sitemaps
  alt-text.php             # Upload hook → Alfred vision endpoint
  internal-links.php       # the_content filter
  robots.php               # robots.txt management
  rest/                    # REST API endpoints
    audit.php
    content.php
    meta.php
    internal-links.php
    sitemap-ping.php
  settings.php             # Options row (managed by Alfred)
tests/                     # WP unit tests
readme.txt
```

**REST endpoints:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/wp-json/alfred-seo/v1/audit` | GET | Returns missing-meta, missing-schema, missing-alt, page list |
| `/wp-json/alfred-seo/v1/content` | POST | Accepts new content (title, body, meta, slug, type) — writes to drafts |
| `/wp-json/alfred-seo/v1/meta` | POST | Updates meta/title/OG/schema overrides on existing posts |
| `/wp-json/alfred-seo/v1/internal-links` | POST | Updates per-site phrase→URL map |
| `/wp-json/alfred-seo/v1/sitemap-ping` | POST | Triggers regen + GSC/Bing ping |

**Auth:** WP application password per site (created during onboarding, stored encrypted in Alfred `seo_sites.wp_app_password`).

### Alfred orchestrator (`core/seo/`)

```
core/seo/
├── sites/
│   ├── registry.py        # DB CRUD for seo_sites
│   └── profile.py         # Brand voice profile loader (YAML + examples)
├── ingest/
│   ├── gsc.py             # Google Search Console daily sync
│   ├── ga4.py             # GA4 daily sync
│   ├── cwv.py             # Core Web Vitals via PageSpeed Insights
│   ├── backlinks.py       # GSC top-linking-sites + diff
│   └── haro.py            # HARO/Connectively inbox monitor
├── analyze/
│   ├── gap_finder.py      # Queries with impressions, low CTR
│   ├── content_audit.py   # Pages missing meta/schema/alt
│   └── topic_picker.py    # Picks next briefs from gaps + site goals
├── content/
│   ├── brief.py           # Topic → brief
│   ├── writer.py          # Brief + brand profile → draft
│   ├── enricher.py        # Existing page → FAQ/cross-sell/story
│   ├── adapter_wp.py      # Draft → WP REST API publish
│   └── schema_builder.py  # Page type + data → JSON-LD
├── queue/
│   ├── pending.py         # Approval state machine (mirrors core/jewelry/social_queue.py)
│   └── publisher.py       # On approve → adapter_wp → plugin
├── monitor/
│   ├── ranking.py         # Position change detection
│   ├── cwv_alerts.py      # CWV regression alerts
│   └── digest.py          # Daily Telegram digest
└── gmb/
    └── client.py          # Google Business Profile API (Roen only initially)

core/api/seo_admin.py      # FastAPI routes for /admin/seo/*
integrations/google_seo/   # GSC + GA4 + PSI + GMB client libs
```

**Database schema (PostgreSQL, `alfred_main` DB):**

| Table | Purpose |
|---|---|
| `seo_sites` | id, slug, domain, wp_rest_url, wp_app_password (encrypted), gsc_property, ga4_property_id, brand_profile_path, status, created_at |
| `seo_queries` | site_id, query, position, impressions, clicks, ctr, captured_at (daily snapshot per query) |
| `seo_pages` | site_id, url, page_type, indexed_at, last_audit_at, schema_status, meta_status, cwv_lcp, cwv_cls, cwv_inp, organic_sessions, conversions |
| `seo_briefs` | id, site_id, topic, content_type, target_keywords, audience, status, brief_payload, created_at |
| `seo_pending` | id, site_id, brief_id, content_type, title, body_payload, source_signal, status, created_at |
| `seo_decided` | Same shape as pending + decided_at, decided_by, outcome, wp_post_id |
| `seo_backlinks` | site_id, source_url, target_url, anchor_text, first_seen, last_seen, lost_at |
| `seo_haro_opps` | id, site_id, source_email_id, query_text, deadline, draft_pitch_payload, status, response_sent_at |
| `seo_rankings_daily` | site_id, query, position, captured_at (for change detection) |

**Scheduled jobs (systemd timers):**

| Job | Cadence | Action |
|---|---|---|
| `seo_gsc_sync` | Daily 4am ET | Per-site query data + indexed pages |
| `seo_ga4_sync` | Daily 4am ET | Per-page organic sessions + conversions |
| `seo_cwv_sync` | Daily 5am ET | Top 20 pages per site, LCP/CLS/INP |
| `seo_backlinks_sync` | Daily 6am ET | GSC top-linking-sites diff |
| `seo_brief_generator` | Daily 7am ET | 1-2 briefs per site, queued for writer |
| `seo_content_writer` | Event-driven (brief queue) | Draft content per type, validate, enqueue |
| `seo_haro_monitor` | Every 15 min | Scan inbox, filter, draft pitches |
| `seo_gmb_sync` | Daily 8am ET (Roen only) | Sync hours/photos/posts |
| `seo_schema_validator` | Daily 11pm ET | Run sample pages through Google Rich Results Test |
| `seo_daily_digest` | Daily 8am ET | Telegram summary |

**LLM model assignments:**

| Job | Model | Why |
|---|---|---|
| Content writer (long-form) | `kimi-k2.6:cloud` | Proven on RuckTalk + Roen product copy |
| Brief / topic picker | `gemma4:31b-cloud` | Fast, sharp, structured output |
| HARO pitch drafter | `kimi-k2.6:cloud` | Editorial quality matters |
| Schema/meta auto-fill | `gemma4:31b-cloud` | Structured, fast |
| Image alt text | `qwen3-vl:235b-cloud` | Already in use for Roen bot |

## Brand voice handling

### Per-site brand profile schema (`data/seo/sites/<slug>/brand.yaml`)

```yaml
slug: roen
domain: roenhandmade.com
display_name: Roen
brand_one_liner: "A small Atlanta jewelry studio."

voice:
  perspective: third_person   # third_person | first_plural | first_singular
  descriptors: [minimal, poetic, Mejuri-aritzia-DNA, quiet-confidence]
  energy: low                 # low | medium | high
  reading_level: 8            # Flesch-Kincaid target
  tagline: "decorate yourself."

never_say:
  - "I"
  - "we"
  - "Sarah"
  - "the maker"
  - "handcrafted with love"
  - "artisanal"
  - "one-of-a-kind"
  - "♥"

always:
  - "Roen is..." (third-person framing)
  - lowercase wordmark in display contexts
  - "Atlanta" (local anchor when natural)

target_audience: |
  Women 25-45 buying handmade jewelry as everyday wear, gift, or
  small indulgence. Discovers via IG, Pinterest, or Atlanta-local
  search. Price-sensitive ($10-65), trust-signal-sensitive.

target_keywords:
  primary: [handmade jewelry, evil eye bracelet, beaded bracelet]
  local: [Atlanta jewelry, handmade Atlanta, jewelry maker Atlanta]
  long_tail: [evil eye bracelet meaning, handmade bracelet under $30]

locked_decisions:
  - "Marble tabletop backdrop only (no on-model imagery v1)"
  - "Accent color terracotta #B85C3D, CTAs and hover only"
  - "No founder photo, no surname"

content_type_preferences:
  product_enrichment: [story_section, materials_care, faq, pair_with]
  cluster_pages: [meaning_history, style_guide, gift_guide]
  blog: [seasonal, occasion_based, materials_education]
  ad_landing: [single_collection_focus, scarcity_when_real]

voice_examples_dir: data/seo/sites/roen/voice_examples/
```

### Few-shot voice examples directory

```
data/seo/sites/<slug>/voice_examples/
├── 01_about_page.md
├── 02_product_red_bead.md
├── 03_product_olive_evil.md
├── 04_ig_caption_evil_eye.md
├── 05_cluster_evil_eye_history.md
├── 06_email_welcome.md
```

5-10 real, in-voice pieces per site. Writer samples 3-4 per generation call.

### Writer pipeline per generation

```
Brief arrives (topic, type, keywords)
    │
    ▼
Load brand.yaml + sample 3-4 voice examples
    │
    ▼
Build prompt:
    [system]   You are writing for {brand}. {voice constraints}. {never_say list}.
    [examples] Here are 4 pieces in this brand's voice: <example bodies>
    [user]     Write {content_type} about {topic} targeting {keywords}.
    │
    ▼
Run kimi-k2.6:cloud (or gemma4:31b-cloud for short content)
    │
    ▼
Post-validation:
    - never_say regex scan
    - Flesch reading score within ±2 of target
    - target keyword present in first 100 words
    - length sane for content type
    │
    ▼
If validation fails → retry once with stricter prompt
If still fails → flag for manual review in queue
If passes → enqueue in /admin/seo/pending
```

### Voice onboarding per site

Each non-Roen site needs voice encoded before content generation:

1. **30-min interview** with Mike (brand promise, audience, never-say list, sample target keywords)
2. **Alfred drafts the brand.yaml**, Mike red-pens
3. **5-10 voice examples** curated from existing site copy (or Mike-approved pieces from past work)
4. Site marked **voice-ready** in `seo_sites.status`

Status as of design date:
- **Roen** — voice locked (from existing memory + brand decisions)
- **AG, AIROI, GRL** — partial (memory has some descriptors, need fuller profile)
- **RuckTalk, LumaBot, LoovaCast** — no locked voice yet

## Data flow

```
4am ET — Overnight sync
  ├─ GSC daily pull           → seo_queries
  ├─ GA4 daily pull           → seo_pages organic_sessions/conversions
  ├─ CWV (top 20 pages/site)  → seo_pages cwv_*
  └─ Backlinks diff           → seo_backlinks (new + lost flagged)

7am ET — Brief generation
  ├─ Gap finder: queries with impressions, low/no CTR
  ├─ Topic picker per site (1-2 briefs/site/day, paced)
  └─ → seo_briefs

Event-driven — Content writer
  └─ Pulls queued briefs → loads brand.yaml + samples voice examples
     → drafts → validates → seo_pending

8am ET — Telegram digest
  └─ Queue depth + ranking changes + backlinks + CWV alerts

Throughout the day — Mike approves
  └─ /admin/seo/pending → approve → publisher → adapter_wp
     → WP REST POST → plugin auto-injects schema/OG → live

Every 15 min — HARO opp monitor
  └─ Scan inbox → match per site → draft pitch → seo_pending tagged "haro"
```

### Approval queue UX

`/admin/seo/pending` is the single surface for all approvals. Each card:
- Site tag (Roen, AG, etc.) — color-coded
- Content type tag (cluster, blog, enrichment, pitch)
- Title
- Preview (~100 chars)
- **Source signal**: the specific gap or query data that justifies this piece
- Three actions: ✓ Approve & publish, ✎ Edit before publish, ✗ Reject

Filter chips at the top scope to one site or one type. Mockup approved 2026-05-14.

## Phasing (10 weeks)

| Week | Milestone | Shippable to Roen |
|---|---|---|
| 1 | WP plugin v1: schema + OG + meta + sitemap + robots + alt text + internal-link filter | Roen products + pages get foundational SEO injected |
| 2 | Alfred orchestrator skeleton: 9 DB tables, FastAPI routes, JWT auth, site registry. Roen as Site #1. | Cross-site dashboard exists (empty). Site registry tested. |
| 3 | Data ingest: GSC + GA4 + CWV + backlinks Layer 1 on systemd timers | Daily data flowing. Roen organic baseline visible. |
| 4 | Brand profile loader, content writer (`kimi-k2.6:cloud`), validator | Writer produces voice-correct Roen content on demand |
| 5 | Approval queue: pending/decided tables, admin UI, publisher.py → WP REST. Brief generator picking from GSC gaps. | End-to-end content flow live for Roen. First real content published. |
| 6 | All 4 content types fully built: product enrichment, cluster, blog, ad landing. Per-type writer adapters. | Roen pipeline at full content velocity. |
| 7 | HARO monitor + pitch drafter. Backlinks Layer 2. GMB for Roen. Schema validator daily run. Telegram digest. | Off-page SEO starts. First HARO pitches go out. Roen GMB live for Atlanta local search. |
| 8 | Voice onboarding interviews × 6 sites. brand.yaml drafted, Mike red-pens. 5-10 voice examples curated per site. | All 7 sites voice-ready. WP plugin deployed to Site #2. |
| 9 | Multi-site rollout. Plugin deployed to sites 2-7. Site-specific schema (Podcast for RuckTalk, SoftwareApplication for LumaBot, Service for AIROI, Event for AG). | All sites in pipeline. Approval queue serves all 7 brands. |
| 10 | Polish + observability: error handling pass, retry queues, alert thresholds, DB indexes, caching, docs, tests covering critical paths. | System fully alive across 7 sites. Phase 1 complete. |

## Error handling

| Failure | Containment |
|---|---|
| External API (GSC/GA4/PSI/HARO/GMB) | Retry with exponential backoff (3 attempts), log to `seo_errors`, surface in admin "data health" widget. Job continues — partial data better than no data. |
| LLM call (writer/brief) | Retry once with stricter prompt → fallback to `gemma4:31b-cloud` → mark brief `needs_manual_review`. Never silent-drop. |
| WP REST publish | Queue with 5 retries over 30 min, then surface in admin with one-click "retry now." Draft NEVER lost — lives in `seo_pending`. |
| Schema validation | Plugin validates JSON-LD against schema.org BEFORE injection. If invalid: skip injection, log to plugin error log, page renders fine. |
| Alfred down (plugin POV) | Plugin uses sane local fallbacks (filename for alt text, WC defaults, no internal-link rewriting). Site stays up. |
| Database | Daemons retry next tick (idempotent). FastAPI 503 with retry hint. Every write transactional. |

## Observability

- **Daily Telegram digest** (8am ET): queue depth, ranking changes, overnight errors
- **🔴 alerts via Telegram**: WP publish stuck > 2h, GSC sync failed 2 days, schema failure rate > 10%
- **`/admin/seo/errors`**: full error log, searchable
- **`/admin/seo` overview**: each job shows ✅/⚠️/❌ status with last-success timestamp

## Testing

| Layer | Approach | Coverage |
|---|---|---|
| Plugin (PHP) | WP unit tests | Schema builders per page type, sitemap output, meta filter, REST auth. ~30 tests. |
| Schema correctness | Google Rich Results Test API nightly | 5 sample pages per site validated against Google's real parser |
| Orchestrator writers/validators | pytest unit tests | Brand-voice validators, schema_builder pure functions, topic picker. ~50 tests. |
| Publish pipeline E2E | pytest integration against disposable WP container | One full flow per content type |
| GSC/GA4 ingest | pytest with VCR-style recorded fixtures | Deterministic daily sync against real-shape API responses |
| Cross-site rollout | Manual smoke checklist post-deploy | 10-min validation per new site |

No load tests — 7 sites × 2-3 pieces/week = ~150 ops/week. Postgres + FastAPI handle that without strain.

## Open questions / known risks

1. **GSC verification status for non-Roen sites** — system needs each site verified in GSC before sync works. Check + verify during Week 2 setup.
2. **WP application passwords** — each site needs one created for Alfred. Created during plugin onboarding (Week 1 for Roen, Week 8-9 for others).
3. **GMB ownership for Roen** — Roen needs a claimed Google Business Profile before Week 7 GMB integration ships. Mike to confirm status by Week 5.
4. **HARO inbox routing** — HARO emails currently go where? Need to route to a dedicated address Alfred monitors (likely `alfred@groundrushinc.com` with a filter, or a fresh inbox).
5. **Voice onboarding 6 sites × 30 min = ~3 hrs Mike-time in Week 8** — if revenue work is pressing that week, push to Week 9 and onboard fewer sites at Week 10. Roen unaffected.

## Out of scope (Phase 2)

- Active backlink outreach automation (competitor analysis + cold outreach)
- Pinterest auto-pin for Roen (gated on Pinterest business account setup)
- Multi-tenant SaaS shell (signup, Stripe billing, client onboarding, marketing site, support tooling) — built only if engine proves itself on Mike's 7 sites first
- YouTube SEO (RuckTalk videos, AG hype)
- AMP, hreflang, programmatic SEO at scale

## Success criteria

- Roen ranks top 10 for "evil eye bracelet meaning" and 2-3 other high-intent long-tails by Day 90
- Roen indexed pages > 100 by Day 60
- 100% of Roen product pages have rich result eligibility in Google Search Console
- Average daily Mike-time in approval queue ≤ 5 min across all 7 sites
- Approval queue rejection rate < 20% (signal that brand voice handling works)
- Zero pages broken by faulty schema injection
- Daily Telegram digest delivered ≥ 95% of days

## Related work

- Current Roen site: `/home/aialfred/alfred/services/roen-minimal/` (theme), `services/alfred-seo/` to be created (plugin)
- Mirrors approval pattern: `/home/aialfred/alfred/core/jewelry/social_queue.py`, `core/api/roen_admin.py`
- Reuses LLM stack: existing Ollama on 105 (kimi-k2.6:cloud, gemma4:31b-cloud, qwen3-vl:235b-cloud)
- Storage: existing PostgreSQL `alfred_main` DB
- Auth: existing JWT cookie pattern in `core/security/auth.py`
- Visual mockup of approval queue: `.superpowers/brainstorm/2290634-1778790179/content/approval-queue.html`
