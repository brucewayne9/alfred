# Mainstay Forge — Plan 1b: Dashboard (Web UI) Implementation Plan

**Goal:** Ship a polished, mobile-first Mainstay Forge dashboard that talks to the Plan-1 job API — five surfaces (Create · Library · Queue · Distribution · Intelligence) — and a safe local demo harness so Mike can see real screenshots without touching the live Alfred service.

**Architecture decision:** Forge is its own product (own subdomain `forge.groundrushcloud.com` per spec §7), NOT a view inside the Alfred butler SPA. So Plan 1b builds a **self-contained dashboard** that can be served standalone. v1b is a single-page app (HTML/CSS/vanilla-JS, brand-matched) wired to the real `/forge/*` API — robust to build, fully functional for Create+Queue, and a high-fidelity designed skeleton for Library/Distribution/Intelligence (which have no backend yet). It can be ported to the React/Vite stack later without changing the API contract.

**Brand:** black `#0e0e10` / gold `#c9a14a`, editorial serif display + clean sans UI — matches the Forge spec emails.

**Tech:** Self-contained `services/forge-web/index.html`. Demo harness: isolated FastAPI on `:8099` (localhost) that mounts `core/api/forge.register`, registers handlers + worker, overrides auth for the demo, serves the dashboard, and seeds demo jobs. Screenshots via Python `playwright` + chromium.

## Surfaces (v1b)
- **Create** — functional: pick format (kinetic-lyric / montage / leak-graphic), enter caption + subfolder, submit → POSTs a job to `/forge/jobs`.
- **Queue** — functional: polls `GET /forge/jobs`, shows each job's type/status/time with status chips (pending/running/done/error).
- **Library** — designed skeleton: vessel grid with source tabs (Upload / Higgsfield / Stock / Fan GIFs) and mood filters.
- **Distribution** — designed skeleton: account groups (burner vs main), a posting calendar strip, assignment rows, "human posts" badge.
- **Intelligence** — designed skeleton: per-sound + per-variant leaderboard, winning-sound callout, funnel recommendation.
Skeletons are clearly labeled "coming online" so they read as vision, not broken features.

## Files
- Create `services/forge-web/index.html` — the dashboard SPA.
- Create `services/forge-web/demo_server.py` — isolated demo/screenshot harness (localhost:8099, auth overridden, seeds jobs, serves the dashboard).
- Create `services/forge-web/shots/` — generated screenshots (desktop + mobile).
- (No change to `core/api/main.py` — live activation remains the documented 3-line step.)

## Safety
- Demo binds localhost:8099 only; never touches alfred.service (:8000) or its DB (uses a throwaway `FORGE_DB_PATH` in /tmp).
- No production deploy overnight; deploy to the subdomain is offered to Mike, not done unattended.

## Done = 
Dashboard builds + loads; Create posts a real job that the worker runs to "done"; Queue reflects it live; all five tabs screenshot cleanly (desktop + mobile); screenshots emailed to Mike with how-to-view + deploy offer.
