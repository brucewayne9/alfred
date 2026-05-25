# RuckTalk Phase 1B-1 verification
_2026-05-24 (UTC: 2026-05-25T01:37:03Z)_

Plan: `docs/superpowers/plans/2026-05-24-rucktalk-seo-phase-1b-1-foundation.md`

Branch: `feat/lucius-hermes-on-111` @ `35d7d1e`

## 1. alfred-seo plugin status on rt-wordpress
```
name,status,version
alfred-seo,active,0.2.0
```

## 2. Rank Math gone
```
seo-by-rank-math,inactive
seo-by-rank-math-pro,inactive
```
Both seo-by-rank-math and -pro show `inactive` — Rank Math no longer emitting head/sitemap output.

## 3. Homepage JSON-LD types
```
"@type":"EntryPoint"
"@type":"Organization"
"@type":"PodcastSeries"
"@type":"SearchAction"
"@type":"WebSite"
```
Expected to include: `Organization`, `WebSite`, `PodcastSeries`. Rank Math's separate Organization/ImageObject/WebPage block from the pre-kill baseline (collision-2026-05-25-0049.md) is gone.

## 4. Sample episode JSON-LD types
URL: `https://rucktalk.com/podcast/episode-210-the-biology-of-focus-managing-your-daily-brain-energy/`
```
"@type":"PodcastEpisode"
"@type":"PodcastSeries"
```
Expected: `PodcastEpisode`, `PodcastSeries` (partOfSeries embedded), `BreadcrumbList`.
`AudioObject` will appear once Plan 1B-2 (transcript/audio pipeline) populates `_rt_audio_url` post_meta — the alfred_seo_schema_audio_object() module returns null-safe when the meta is missing, so PodcastEpisode renders cleanly without it.

## 5. Stale Sonaar sitemap is dead
```
HTTP/2 404 
```

## 6. orchestrator: rucktalk registered as Site #2
```
id=64
slug=rucktalk
domain=rucktalk.com
wp_rest_url=https://rucktalk.com/wp-json
wp_username=alfred-seo
wp_app_password_set=True (length=24)
gsc_property=sc-domain:rucktalk.com
ga4_property_id=123456789  # FAKE from Task 2 test run; real numeric ID TBD
brand_profile_path=data/seo/sites/rucktalk/brand.yaml
business_type=Organization
```

## 7. Brand profile loads cleanly
```
display_name=RuckTalk
voice.perspective=first_person_plural
voice.flesch_target=65
never_say count=14
keywords_for(blog)[:3]=['how to transition from CEO mode to dad mode', 'entrepreneur father work life integration', 'podcast for entrepreneur dads']
voice_examples loaded=5
```

## 8. Taxonomy cleanup
Pre-cleanup: **38 terms** across 16 non-builtin taxonomies.
Post-cleanup: **24 terms**, 14 deleted (all empty Sonaar music/podcast leftovers).
WC/Elementor system taxonomies left untouched per their plugin contracts.

## Commits landed in this phase
```
35d7d1e chore(seo): prune 14 stale Sonaar/podcast taxonomies on rucktalk
e3eb460 feat(seo): kill Rank Math on rt-wordpress + capture post-kill audit
3993fdd chore(seo): audit script for rucktalk Rank Math/Sonaar collision baseline
1938c84 feat(alfred-seo): wire podcast schemas into dispatcher (v0.2.0)
879553a feat(alfred-seo): PodcastEpisode schema module (embeds AudioObject)
ad80f31 feat(alfred-seo): AudioObject schema module
4c4781c feat(alfred-seo): PodcastSeries schema module
472f8f8 feat(seo): register rucktalk.com as Site #2 in orchestrator
04c1b42 feat(seo): seed RuckTalk brand profile + voice examples
```

## Open items

- **GSC OAuth backfill** — blocker for Plan 1's brief generator pulling RuckTalk GSC gap data. ~5 min interactive click per `docs/seo/OAUTH_SETUP.md`. Same blocker applies to all 7 sites; not RuckTalk-specific.
- **Real GA4 numeric property ID** — currently `123456789` (Task 2 test fake). Mike needs to look this up in GA4 admin and run: `RUCKTALK_GA4_PROPERTY_ID=<real-id> ./venv/bin/python scripts/seo_init_rucktalk.py` to overwrite.
- **PodcastEpisode audio fields** — `_rt_audio_url`, `_rt_audio_duration`, `_rt_audio_bytes` post_meta unpopulated on existing episodes. Plan 1B-2 (transcript + audio pipeline integration) backfills these and AudioObject lights up automatically via the existing schema module.
- **Manual Rich Results validation** — Google's tool at https://search.google.com/test/rich-results requires a browser session; Mike to verify PodcastSeries on home + PodcastEpisode on a sample episode and confirm no errors.
- **4 podcast-category survivors with weird auto-imported names** (`184`/`183`/etc.) — rename/merge candidates for the content-IA pass (Plan 1B-3 pillar pages).
