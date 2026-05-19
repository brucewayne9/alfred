# RuckTalk Rebuild — Phase 0: Brand & Positioning Spec

**Date:** 2026-05-19
**Owner:** Mike Johnson
**Author:** Alfred
**Status:** Approved (Phase 0 design)
**Supersedes:** [`2026-04-14-rucktalk-content-pipeline-design.md`](./2026-04-14-rucktalk-content-pipeline-design.md) (pipeline survives; positioning replaced)
**Unblocks:** Phase 1 (Site redesign + SEO plug-in)

---

## 0. What this doc is — and isn't

**Is:** the strategic and positioning spec for turning RuckTalk from "a podcast site" into a personal-brand media business owned by Mike Johnson. Defines identity, IA, key surfaces, ecosystem cross-promo, and migration policy.

**Isn't:** an implementation plan. No code, no task list, no file paths to touch. Phase 1+ docs handle the build.

---

## 1. Identity

**RuckTalk is a personal brand fronted by Mike Johnson.** Not a faceless network, not "RuckTalk-the-show with rotating hosts." Mike is the face, the voice in copy, the host of the podcast, the person on the YouTube long-form recordings, and the signature on the shop.

### 1a. Editorial DNA (already locked, captured here for the brand profile)

**What the show is:**
> Regular guys talking to other regular guys about real life — work, decisions, money, family, what to do when things break, recovery when the day was brutal. Training, gear, and nutrition come up because they matter, not because the show is about them. No expert posture. No soapbox.

**Mike's pillar statement (verbatim, 2026-05-13):**
> "Stay healthy. Find my peace. Be a good partner to my wife, a good husband to my wife, a good friend to my wife. And everything else everybody else is going through. I have my political beliefs, but I'm not on a soapbox — I'm just a regular guy that wants to talk to other regular guys that are strong and how to get stronger."

**Five pillars:** Health · Peace · Family · Strength · Shared experience

**Tone:** Regular guys talking to other regular guys. No expert/coach posture. No soapbox. Practical, not panicked.

**Realism filter:** The only hard content filter. Topics and framings must be realistic for today's regular guy. Politics, "tips," training — all fair game IF realistic. Doomsday/movie-plot/survival-fantasy framings get cut.

### 1b. Domain
**rucktalk.com.** Cloudflare 525 on the `www.` subdomain (broken since 2026-05-14) gets fixed as part of Phase 1 cutover.

### 1c. Sub-brand consolidation
**fitasruck.com 301-redirects to rucktalk.com/training.** Full consolidation. One site, one email list, all SEO juice merges. The hello@fitasruck.com Mailcow inbox stays alive for reply continuity.

---

## 2. Site Information Architecture

Top navigation (6 items, in order):

| Item | Purpose |
|------|---------|
| **Home** | Hero + cross-section preview + signup |
| **Podcast** | Sonaar episode archive, RSS, Spotify/Apple/YouTube subscribe links |
| **Blog** | Auto-blog archive (keeps the existing daily engine that's "doing great") |
| **Training** | Free 8-week PDF (email-gated) + $29 paid FaR SKU + future training content |
| **Shop** | WC catalog (hats + tees at launch, expand later) |
| **About** | Mike's bio, the five pillars, what the show is and isn't |

Footer (always-visible):
- Newsletter signup form
- Social links (IG, FB, YouTube, X, TikTok)
- Legal (terms, privacy, cookies)
- Contact
- Ecosystem strip (see §12d)

---

## 3. Homepage layout

### 3a. Persistent floating radio bar (top of every page, not just home)
Mini Spotify-style player. Pause/play, scrolling track title, "Powered by **LoovaCast** →" pill in the corner. Stream = RuckTalk station on LoovaCast (24/7 autoDJ of past episodes, bed music between).

### 3b. Hero (above the fold)
- **Visual:** Mike's photo, mid-shot, looking at camera, no gym props or weighted vest. Sturdy and present.
- **Tagline:** "Real talk for guys in the thick of it." (see §4 for alternatives)
- **Primary CTA:** "Get the free 8-week RuckTalk plan" → email capture (the lead magnet)
- **Secondary CTA:** "Listen to today's episode" → latest podcast page

### 3c. Below the fold (scroll order)
1. Latest podcast episode card (large, with play button)
2. Latest 3 blog posts (cards in a row)
3. Shop teaser (3 product cards)
4. Inline newsletter signup block (second chance for email)
5. "About RuckTalk" 1-paragraph blurb + link to About page
6. Footer + ecosystem strip

**Rationale:** north-star metric is names + emails captured. Email capture is the primary CTA. Podcast is brand-heart → secondary. Everything below the fold reinforces both.

---

## 4. Tagline

**Selected:** *"Real talk for guys in the thick of it."*

Alternatives considered (rejected for Phase 0 but available for A/B testing later):
- B. "Stay healthy. Find peace. Show up for your family. Get stronger." — pillar-stack, more SEO-friendly, slightly preachy
- C. "A podcast and a small shop for guys building a real life — and not falling apart doing it." — more informational, longer, weaker hook

---

## 5. Newsletter

### 5a. Platform & list
- **Platform:** Brevo (existing wiring; no new vendor)
- **List name:** `RuckTalk` (the FaR list migrates into this on Phase 1 cutover)
- **Sender:** mike@rucktalk.com (or similar — Phase 3 confirms address)

### 5b. Double opt-in flow (mandatory)
1. User submits email via any signup surface (§5d)
2. Brevo `subscribe` API creates pending contact + auto-sends verify email
3. User clicks verify link → contact marked `confirmed` in Brevo
4. **Brevo confirmation webhook fires** → n8n workflow `o9cIjGWj8z9pwknY` (see §5c)
5. Brevo automation delivers the free 8-week PDF immediately
6. Auto-sequence (§5f) kicks off from confirmation timestamp

**Why mandatory:**
- Brevo deliverability scoring rewards double opt-in heavily (sender reputation)
- Filters bot signups and typo'd addresses
- Clean GDPR/CAN-SPAM posture
- The n8n weekly list only ever sees verified humans

### 5c. n8n integration
- **Workflow:** `o9cIjGWj8z9pwknY` at `https://automate.groundrushlabs.com/workflow/o9cIjGWj8z9pwknY` (Mike's existing weekly newsletter pipeline)
- **Integration point:** Brevo webhook → n8n inbound webhook node
- **Payload contract:** `{email, first_name, signup_source, confirmed_at, brevo_contact_id}`
- **Action:** n8n adds the contact to its weekly-newsletter source list (Phase 3 build inspects which sheet/list)
- **Failure mode:** if n8n is down, Brevo still has the contact. n8n side becomes eventually-consistent via a daily reconciliation cron (Phase 3 spec).

### 5d. Signup surfaces
1. **Hero CTA** — primary homepage above-the-fold (§3b)
2. **Inline content blocks** — embedded in blog/episode posts (after intro paragraph and again before footer)
3. **Footer form** — always visible at bottom of every page
4. **Popup** — the one allowed popup (§5e)

### 5e. The one allowed popup

**Single popup, newsletter only.** No other popups exist anywhere on the site for any reason.

| Field | Value |
|-------|-------|
| Trigger | First-time visitor + (50% scroll-depth on any page, OR exit-intent on second page view). Never on bounce. |
| Frequency cap | Once per visitor. 14-day cookie suppression after dismissal. |
| Layout | Centered modal, PDF cover hero image (~300px wide), headline, ONE field (email), one button, micro-copy. |
| Headline | "Get the free 8-week RuckTalk plan" |
| Sub-headline | "What I'd do in your first 8 weeks if I were starting over." |
| Field | Email only (first name optional, defer to Phase 3 A/B) |
| Button | "Send it to me" |
| Micro-copy | "We'll email you to confirm. No spam, unsubscribe any time." |
| Post-submit | Modal flips to: "✓ Check your email to confirm. The plan lands in your inbox the moment you click verify." |
| Mobile | Full-width modal, safe-area padding, dismiss X top-right (44×44 touch target) |
| Accessibility | Trap focus inside modal, ESC dismisses, ARIA labels, dismissal NOT tied to time |

### 5f. Auto-sequence (post-confirm)
| T+ | Email |
|----|-------|
| Instant | Deliver PDF + welcome email + intro to what RuckTalk is |
| 3 days | "What brought you here?" — replies route to Mike's Telegram |
| 7 days | Shop intro (link to /shop, 3 featured products) |
| 14 days | $29 paid FaR pitch (segment excludes existing buyers) |
| 21 days | Drop into weekly Sunday cadence |

### 5g. Weekly cadence (steady state)
- **Day/time:** Sunday **8 AM ET** (deliberately after the 6 AM blog publish in §9 so the email can reference the new post)
- **Sender:** Mike (n8n workflow composes + sends)
- **Content:** Best blog of the week + 1-2 episode highlights + shop drop or training tip
- **Length:** ~300-500 words, scannable

---

## 6. Shop v1

### 6a. Launch SKUs (4 products)
| SKU | Type | Price |
|-----|------|-------|
| RuckTalk classic cap | Hat | $25 |
| RuckTalk patch cap (alt color) | Hat | $25 |
| RuckTalk logo tee | Tee | $30 |
| "Health. Peace. Family. Strength." pillar tee | Tee | $30 |

**Source:** Hats produced on Mike's existing hat press downstairs. Tees TBD (Phase 2 picks blank vendor + print method).

### 6b. Platform & ordering pattern
- **Storefront:** WooCommerce on the rucktalk.com WP install
- **Order management:** **@RuckTalkBot** Telegram bot — exact pattern as @roenhandmadebot
  - `/orders` lists processing orders
  - Tap order → Ship & Complete → paste tracking → auto-saves WC + USPS/UPS/FedEx/DHL + flips status + customer email fires
- **Payment:** Stripe (existing wiring)
- **Fulfillment:** Mike ships from downstairs. Manual at launch. Print-on-demand evaluated if volume warrants.

### 6c. Brand voice in product copy
Same DNA: practical, conversational, no hype. A hat description reads like Mike telling a friend why he wears it, not like Bonobos copy.

---

## 7. Sonaar theme approach

### 7a. Strategy
**Stay on Sonaar.** Mike confirmed: redesign within theme constraints, not a theme rip-and-replace.

### 7b. Why
- Sonaar already does the heavy lifting on the podcast post type (custom meta, RSS feed, Spotify/Apple syndication)
- Switching themes risks breaking the podcast pipeline (which is working — see [rucktalk-pipeline-status])
- A Sonaar **child theme** gives safe override paths for templates, CSS, and functions

### 7c. Phase 1 starts with a theme audit (NOT code)
Before any redesign work, Phase 1 produces a Sonaar audit doc covering:
- Available template hooks and filter points
- Sonaar's CSS variable system (what we can theme cleanly)
- Conflicts with custom homepage, shop pages, training pages
- Areas where a custom page template is needed vs. theme override

The audit defines what's possible inside Sonaar's constraints before any redesign code is written.

---

## 8. SEO plug-in

### 8a. Strategy
RuckTalk gets the same treatment Roen Handmade got. Plug into Alfred SEO, use the existing weekly blog engine, run keyword discovery + audit against the new site.

### 8b. Concrete steps (Phase 1 implementation, listed here for scope clarity)
1. Encode RuckTalk brand profile into Alfred SEO (closes [Task #73](task)) — editorial DNA from §1a becomes the writer's voice constraint
2. Run keyword discovery against rucktalk.com (DataForSEO, ~$2 spend)
3. Run site audit against rucktalk.com (post-redesign, so we audit the final state)
4. Wire the weekly blog engine to include `rucktalk` in `ACTIVE_SITES`
5. Tune the engine for RuckTalk's editorial DNA (longer-form, more conversational)

### 8c. RuckTalk vs Roen — different keyword strategy
- **Roen:** product-modifier queries ("turquoise gold beaded bracelet"). High intent, lower volume, transactional.
- **RuckTalk:** long-tail conversational queries ("how do I get healthier when I have kids", "how to stay strong working from home", "what to do when work and family pull you apart"). Lower intent, higher volume, informational + brand-building.

The blog writer prompt and keyword pull strategy differ accordingly. Phase 1 spec details the prompt tuning.

---

## 9. Auto-blog: what stays, what changes

### 9a. Keeps running
The existing daily auto-blog (`scripts/rucktalk_daily_social.py` + blog engine) stays. Mike's words: "doing great." No code changes to the existing engine.

### 9b. Adds
The new SEO weekly blog engine (`scripts/seo_weekly_blog_engine.py`, shipped 2026-05-18) extends to RuckTalk. Produces ONE deeper SEO-anchor post per week, Sunday 6 AM ET.

### 9c. Net cadence
- 5 daily auto-posts (Mon–Fri, existing engine)
- 1 weekly SEO anchor post (Sunday, new engine)
- = 6 posts/week

### 9d. Open question for Phase 1 review
If 6 posts/week feels like content overload after a month of data, the alternative is dropping daily to 3x/week (Mon/Wed/Fri) and letting the SEO weekly carry analytical depth. Phase 1 ships the 6/week cadence; Phase 1 + 30 days of GA4 data tells us if we throttle.

---

## 10. FaR migration

### 10a. Cutover policy
- `fitasruck.com` → 301 → `rucktalk.com/training` (root)
- All fitasruck.com paths 301 to `rucktalk.com/training` (root unless specific mapping needed)
- Free 8-week PDF lives at `rucktalk.com/training/free` (email-gated)
- Paid $29 SKU lives at `rucktalk.com/training/8-week-plan` (WooCommerce product)
- Existing $29 customer accounts: migrated to RuckTalk WP install, password-reset email sent

### 10b. SEO risk & mitigation
- **Expected dip:** ~30-day SERP volatility on fitasruck.com keywords (acceptable — FaR isn't ranking heavily anyway per [FaR campaign decommission notes])
- **Mitigation:** 301s for every indexed URL, Search Console change-of-address signal, sitemap re-submission for rucktalk.com
- **Recovery target:** SERP positions restored to fitasruck.com baseline within 45 days

### 10c. Inbox continuity
- `hello@fitasruck.com` Mailcow inbox stays alive
- Auto-forward set to mike@rucktalk.com (or wherever Mike wants replies)
- Existing reply threads stay on the original address (don't break trust)

---

## 10.5. Podcast distribution — audit + reconnect

**Problem:** Mike is unsure of the current state of his podcast distribution. He thinks he has a Spotify account, isn't sure about Apple, and Google Podcasts no longer exists as a standalone product (discontinued April 2024 — feed inheritors are YouTube Music for podcasts).

**Goal:** Before Phase 1 launch, audit every podcast platform, confirm or create the RuckTalk show on each, and verify the Sonaar-generated RSS feed is being polled correctly.

### 10.5a. Pre-launch audit checklist (Alfred can do most of this read-only)
| Platform | What to check | Who acts |
|----------|---------------|----------|
| **Spotify for Podcasters** | Does the RuckTalk show exist? Is `rucktalk.com/feed/podcast` (or current Sonaar feed URL) the configured RSS? | Mike logs in, Alfred checks feed mapping |
| **Apple Podcasts Connect** | Same — show exists? feed correct? | Mike logs in (Apple ID), Alfred checks |
| **YouTube Music for Podcasts** | New replacement for Google Podcasts. Distribution is via YouTube Studio podcast settings. Show exists? Linked to Mike's YouTube channel? | Mike logs in, Alfred checks |
| **Amazon Music / Audible** | Low-traffic but cheap to add | Mike applies, Alfred submits feed |
| **iHeartRadio** | Low priority — defer | — |
| **Sonaar RSS feed URL** | Confirm the canonical feed URL on rucktalk.com, validate XML, test in [castfeedvalidator.com](https://castfeedvalidator.com) | Alfred |

### 10.5b. Reconnect & verify flow (Phase 1)
1. **Alfred:** pull current Sonaar feed URL from `rucktalk.com` WP REST API, validate it's still serving (and that the recent NotebookLM episodes are listed correctly)
2. **Mike:** log in to each platform listed above, confirm the show exists OR create it
3. **Mike + Alfred (paired session):** verify the feed URL configured on each platform matches the canonical Sonaar feed
4. **Alfred:** for any platform where the feed needs to update, submit/re-submit. Document each platform's feed-refresh delay (Spotify: ~few hours, Apple: ~24h, YouTube Music: ~48h)
5. **Alfred:** add a weekly Sunday cron that re-validates the feed URL is reachable and that each platform's "last episode date" matches what's on rucktalk.com
6. **Mike:** add the "Listen on Spotify / Apple / YouTube" buttons to the redesigned homepage and Podcast page hero

### 10.5c. New episodes propagation
Each platform pulls from RSS on its own cadence. Sonaar already publishes new episodes to the feed when the WP `podcast` post type is published. No additional work needed — IF the feed URLs are correct everywhere.

### 10.5d. Risk: orphan/duplicate shows
If Mike has an old test/orphan show on Spotify or Apple with a different feed, that needs to be merged or killed BEFORE adding the new one. Otherwise listeners end up on the wrong show. Audit step (10.5a) catches this.

### 10.5e. Phase positioning
This is a **Phase 1 task** (paired with site redesign). It's not Phase 0 strategy — it's pre-launch checklist work. Phase 1 plan will detail tasks.

---

## 11. Out of scope for Phase 0 — Deferred to later phase docs

| Topic | Phase doc |
|-------|-----------|
| Shop product photography, SKU production process, fulfillment SOPs | Phase 2 |
| Newsletter sequence copy, segment rules, A/B variants | Phase 3 |
| Social repurposing pipeline architecture | Phase 4 |
| YouTube long-form pipeline (NextCloud watcher extension) | Phase 5 |
| Membership / paid community tier (not in current scope) | TBD |
| Sponsorship / podcast ad inventory | TBD |

---

## 12. Ecosystem cross-promo

**Principle:** functional placements first, attribution second, ecosystem visible without being a billboard. No banner-ad energy anywhere except the one allowed newsletter popup (§5e).

### 12a. LoovaCast — always-on RuckTalk radio
- **Surface:** persistent floating player bar at top of every page (§3a)
- **Stream:** new "RuckTalk" station on LoovaCast (24/7 autoDJ rotation of past episodes, fills with bed music between shows)
- **Attribution:** tiny "Powered by **LoovaCast** →" pill in player corner, links to loovacast.com
- **Why this works:** the player is genuinely useful (one-tap listen any time) → the LoovaCast credit powers the feature, not a billboard
- **Note:** AzuraCast was retired 2026-05-17 — this is exactly the right replacement use case
- **Sonaar consideration:** Phase 1 audit checks for theme audio player conflicts

### 12b. LumaBot — site-aware chat
- **Surface:** standard chat bubble bottom-right
- **Connected to:** WordPress REST API (site-aware) so it can answer episode/blog/shop questions
- **Attribution:** "Powered by **LumaBot** →" in chat header
- **Fallback behavior:** when it can't answer, captures email + question + "I'll get Mike to follow up." Every miss becomes a lead.
- **Phase 1 questions to resolve:** (1) does LumaBot have a WordPress connector or are we building one? (2) embed model — `<script>` tag or iframe widget?

### 12c. AIROI — contextual promo (not blanket)
- **Surface:** does NOT appear on every page
- **Where it DOES appear:**
  - Blog posts whose topic touches AI / business / time / efficiency (auto-tagged via the SEO engine's topic classifier)
  - A `/tools` or `/resources` page if Phase 1 adds one
- **Placement:** inline content block: "Curious what AI could actually save your business? → Try the AI Savings Calculator" (button)
- **Why contextual only:** AIROI is for entrepreneurs. A husband-and-dad reader doesn't care. Don't dilute by spraying everywhere.

### 12d. Ecosystem footer strip
Above the standard footer, a thin strip:

> **Part of the Ground Rush ecosystem**
>
> LoovaCast · LumaBot · AIROI · Roen Handmade · Ground Rush Labs

Each entry is a logo wordmark + 2-word descriptor. Greyscale until hover, full color on hover. Each links to its respective site. Tasteful, professional, signals "this isn't a lone-wolf project."

### 12e. What we are NOT doing
- ❌ No popup ads for sister brands (LoovaCast/LumaBot/AIROI all stay subtle per 12a-d)
- ❌ No interstitials or "before you go" popups *other than* the newsletter (§5e)
- ❌ No homepage hero real estate spent on ecosystem
- ❌ No multi-tab "Our products" page — footer strip handles it
- ✅ ONE allowed popup: the newsletter signup per §5e

---

## 13. North star metric & success criteria

### 13a. Phase 0 (this doc)
Spec written, reviewed, approved by Mike, committed to git, Phase 1 plan kicks off.

### 13b. Phase 1+ ongoing metric
**Names + emails captured per week.**

Secondary:
- Podcast plays (GA4 audio_play events, already wired)
- Shop conversion (WC analytics)
- Newsletter open + click rates (Brevo)
- Cross-brand traffic from footer strip (UTM-tracked)

### 13c. 90-day target (set in Phase 3 newsletter spec)
TBD — Phase 3 picks a baseline target based on current FaR list size at migration + organic traffic baseline.

---

## 14. Open items requiring Mike's input or external dependency

| # | Item | Blocker |
|---|------|---------|
| 1 | Mike's homepage hero photo | Need a new shot, or pick from existing library |
| 2 | LumaBot WordPress connector status | Mike confirms whether to build or use existing |
| 3 | Hat & tee mockup approval | Phase 2 produces mockups for Mike to sign off |
| 4 | n8n workflow `o9cIjGWj8z9pwknY` source-list inspection | Phase 3 build pulls workflow JSON via n8n API |
| 5 | Newsletter sender email (mike@rucktalk.com vs. ruck@rucktalk.com vs. ?) | Mike picks in Phase 3 |
| 6 | $29 FaR customer migration cutover date | Coordinated with Phase 1 launch |

---

## 15. Phase sequencing (recap for context)

| # | Phase | Blocked by |
|---|-------|------------|
| 0 | **This doc** | — |
| 1 | Site redesign + SEO plug-in + **podcast distribution audit/reconnect** (§10.5) — paired build | Phase 0 |
| 2 | Shop (WC + @RuckTalkBot) | Phase 1 |
| 3 | Newsletter + FaR lead magnet + n8n wiring | Phase 1 |
| 4 | Social repurposing (NotebookLM → reels) | Phase 1 |
| 5 | YouTube long-form pipeline | Phase 1 |

Phases 2-5 can be built in any order (or parallel) after Phase 1 ships.

---

## 16. References

- Editorial DNA & podcast pipeline: [`project_rucktalk_daily_podcast.md`](memory) — 2026-05-13 brand correction
- Pipeline state: [`rucktalk_pipeline_status.md`](memory) — daily blog, social, episode engines
- Roen pattern (shop + bot): [`project_roens_bracelet_box.md`](memory), [`project_roen_bot_orders_flow.md`](memory)
- SEO weekly engine (extending to RuckTalk): `scripts/seo_weekly_blog_engine.py` (commit `be8a287`)
- LoovaCast platform: [`project_loovacast_billing_rebuild.md`](memory) — billing rebuild complete 2026-04-23
- AzuraCast retirement (do not use): [`azuracast_retired.md`](memory)
- n8n workflow: `https://automate.groundrushlabs.com/workflow/o9cIjGWj8z9pwknY`
- PDF source: `fitasruck.com` checkout asset (Phase 3 lifts the file)
