# Tools Reference
`S=~/.openclaw/workspace/scripts/integrations`
Pattern: `python3 $S/<script>.py <cmd> [args]`
## 1. CRM (Twenty) — `crm.py`
```
search-people "name" | list-people [limit] | get-person <id>
create-person '{"name":{"firstName":"J","lastName":"D"}}'
update-person <id> '{"jobTitle":"CEO"}' | delete-person <id>
list-companies [limit] | search-companies "Acme" | get-company <id>
create-company '{"name":"Acme Corp"}' | pipeline | list-opportunities [limit]
create-note <person_id> "note text" | list-tasks [limit]
create-task '{"title":"Follow up","assigneeId":"..."}'
create-linked-note <person_id> <body_markdown> [title] — Creates note + links to contact atomically; rolls back on link failure (no orphans). Returns noteId + targetId.
create-linked-task <person_id> <title> [status] — Creates task + links to contact atomically; status defaults to TODO. Rolls back on link failure. Returns taskId + targetId.
```
search-people returns up to 500 results. Multiple matches: presents numbered list for user to pick from. Zero matches: offers to create a new contact.
## 2. Email — `email_client.py`
Accounts: alfred (alfred@groundrushlabs.com), groundrush (mjohnson@groundrushlabs.com), rucktalk (info@rucktalk.com), loovacast (info@loovacast.com), lumabot (lumabot@groundrushlabs.com), support (support@loovacast.com), groundrush_info (info@groundrushlabs.com)
```
accounts | inbox <acct> <n> | unread [acct]
send <acct> <to> "Subject" "Body"
search <acct> "query" | mark-read <acct> <uid>
mark-unread <acct> <uid> | trash <acct> <uid>
```
Outbound: always use alfred. Others READ ONLY (except groundrush for Mike-authorized). See SOUL.md.
## 3. Calendar — `google_calendar.py`
```
today | events <days> | delete <event_id>
create '{"summary":"Mtg","start":{"dateTime":"2026-02-15T14:00:00-05:00"},"end":{"dateTime":"2026-02-15T15:00:00-05:00"}}'
update <event_id> '{"summary":"New Title"}'
add-attendees <event_id> a@x.com,b@x.com | free-time <date> <min>
```
## 4. Stripe — `stripe_api.py`
```
balance | customers [search] | create-customer '{"name":"J","email":"j@x.com"}'
invoices [status] | create-invoice '{"customer":"cus_xxx"}'
finalize <inv_id> | send <inv_id> | payments [limit]
products | subscriptions | refund <pay_id> [amount]
revenue [days] | payment-links | create-link <price_id>
```
## 5. Meta Ads — `meta_ads.py`
```
campaigns [period] | adsets [campaign_id] | ads [adset_id]
spend [days] | account | issues
pause <campaign_id> | enable <campaign_id>
pause-ad <ad_id> | enable-ad <ad_id>
budget <campaign_id> <daily_dollars>
```
## 6. Google Ads — `google_ads.py`
```
accounts | campaigns [cid] | performance [cid] [days]
ad-groups <camp_id> [cid] | keywords [camp_id] [cid]
spend [cid] [days] | pause <camp_id> [cid] | enable <camp_id> [cid]
```
## 7. Google Analytics — `google_analytics.py`
Properties: rucktalk, nightlife, rodwave, lenssniper, loovacast, lumabot, myhands, agentertainment
```
properties | summary <prop> <days> | realtime <prop>
sources <prop> <days> | pages <prop> <days>
devices <prop> <days> | countries <prop> <days>
daily <prop> <days> | all <days>
```
## 8. WordPress — `wordpress.py`
Sites: groundrush, loovacast, rucktalk, nightlife, lumabot, myhandscarwash
```
sites | posts <site> [n] | get-post <site> <id>
create-post <site> '{"title":"T","content":"<p>C</p>","status":"draft"}'
update-post <site> <id> '{"title":"U"}' | delete-post <site> <id>
pages <site> | create-page <site> '{"title":"T","content":"<p>C</p>"}'
plugins <site> | activate-plugin <site> <slug>
media <site> | users <site> | categories <site> | tags <site>
seo <site> <post_id> | health <site>
```
## 9. Radio — `azuracast.py`
Station: News Mews Radio (studiob.loovacast.com)
```
now-playing | history [limit] | status | stations
skip | restart | playlists | playlist <id>
toggle-playlist <id> | create-playlist "name" | delete-playlist <id>
reshuffle <pl_id> | media [limit] | search-media "query"
queue | clear-queue | requestable | request <song_id>
streamers | mounts | webhooks
```
## 10. Smart Home — `homeassistant.py`
URL: home.groundrushlabs.com (SSL cert broken)
```
status | devices | state <eid> | turn-on <eid> | turn-off <eid> | toggle <eid>
light <eid> <0-100> | room "living room" on | thermostat 72 | weather
scenes | activate-scene scene.movie_night
scripts | run-script script.goodnight
media-players | media <player> play | volume <player> 50
```
## 11. Nextcloud — `nextcloud.py`
URL: groundrushcloud.com
```
files [path] | search "query" | mkdir <path> | delete <path>
move <src> <dst> | notes | create-note "title" "content"
update-note <id> '{"content":"U"}' | conversations
send-message <room_token> "msg" | user | storage
```
## 12. n8n — `n8n.py`
URL: automate.groundrushlabs.com
```
list [limit] | get <wf_id> | delete <wf_id>
create '{"name":"N","nodes":[],"connections":{}}'
update <wf_id> '{"name":"U"}' | activate <wf_id> | deactivate <wf_id>
execute <wf_id> [data_json] | executions [wf_id] [limit]
```
## 13. Knowledge Graph — `lightrag_client.py`
URL: greymatter.groundrushlabs.com
```
health | query "question" | insert "fact" | search "topic"
```
## 14. Firecrawl — `firecrawl_api.py`
```
scrape <url> [format] | crawl <url> [limit] [depth]
crawl-status <id> | search "query" [limit] | extract <url> "prompt"
```
## 15. Weather — `weather.py`
```
current [coords] | forecast | hourly
```
Default location: Atlanta. Coords format: "33.749,-84.388"
## 16. Google Workspace — `google_workspace.py`
```
drive-list [query] [type] | drive-search "query" | drive-get <fid>
drive-mkdir "name" [parent_id] | drive-share <fid> email [writer]
drive-delete <fid> | docs-create "title" "content"
docs-read <did> | docs-append <did> "text"
sheets-create "title" | sheets-read <sid> [range]
sheets-append <sid> '[["r1c1","r1c2"]]' [range]
sheets-update <sid> "A1:B2" '[["a","b"]]'
slides-create "title" | slides-get <pid>
```
## 17. Travel — `~/workspace/skills/travel_agent.py`
Env: BRAVE_API_KEY, FIRECRAWL_API_KEY
```
search --from "Atlanta" --to "Paris" --depart "2026-03-15" --return "2026-03-22" [--name "Mike" --email "m@g.com" --calendar --crm]
flights --from "ATL" --to "CDG" --depart "2026-03-15" --return "2026-03-22"
hotels --destination "Paris" --checkin "2026-03-15" --checkout "2026-03-22"
```
## 18. Telegram (Built-in OpenClaw Tool)
Use OpenClaw's built-in send-message tool, NOT a Python script.
Target must be numeric chat ID only. Do not use display names.
- Mike's Telegram: 7582976864
- Wrong: "MJ (@groundrushlabs) id:7582976864"
- Right: 7582976864

## Alfred Labs Bridge
$S = call_alfred_labs.py
| Command | What it does |
|---------|-------------|
| $S ask "<request>" | Send any task to Alfred Labs (353 tools: WordPress, Meta Ads, Google Analytics, Stripe, n8n, Nextcloud, Home Assistant, Google Workspace, Firecrawl, etc.) |

Use this for ANY task your 16 scripts can't handle. Labs will execute with its full toolkit and return results.
