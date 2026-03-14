# The Morning Brew - Automated AI Morning Show

## Build Guide for News Mews Radio (AzuraCast Station 22)

**Station:** studiob.loovacast.com
**Server:** 75.43.156.101 (Alfred Claw) + AzuraCast on 75.43.156.100
**Format:** 4-hour morning show, 5 segments per hour, 20 total segments
**Style:** Fox & Friends / NBC Today Show — two AI hosts, conversational, current events

---

## 1. Overview

The Morning Brew is a fully automated AI morning radio show. Every day before air time, a pipeline gathers current news, generates 20 two-host audio segments via Google NotebookLM, and uploads them to AzuraCast for sequential playback.

### Daily Timeline

| Time | Action |
|------|--------|
| 2:00 AM | News gathering — RSS feeds scraped, weather pulled, stories curated |
| 2:15 AM | Script/source prep — 20 topic bundles created for NotebookLM |
| 2:30 AM | Audio generation begins — NotebookLM creates 20 segments (~3-5 min each to generate) |
| 4:30 AM | All 20 audio files downloaded |
| 4:45 AM | Files renamed with sequential naming, SFTP'd to AzuraCast |
| 5:00 AM | Playlist updated and verified |
| 6:00 AM | The Morning Brew goes live on News Mews Radio |

### Show Format (Per Hour)

| Segment | Name | Content | ~Length |
|---------|------|---------|--------|
| Break 1 | **The Wake Up** | Top headlines, breaking news | 8-10 min |
| Break 2 | **The Scoreboard** | Sports recap, scores, hot takes | 8-10 min |
| Break 3 | **The Buzz** | Entertainment, pop culture, trending | 8-10 min |
| Break 4 | **The Forecast** | Weather, lifestyle, health tips | 8-10 min |
| Break 5 | **The Roast** | Hosts banter, hot takes, fun segment | 8-10 min |

Music/jingles play between segments via AzuraCast Auto-DJ.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CRON (2:00 AM Daily)                  │
│              or n8n Scheduled Trigger                    │
└─────────────┬───────────────────────────────────────────┘
              │
              v
┌─────────────────────────────┐
│  Step 1: NEWS GATHERING     │
│  - RSS feeds (AP, ESPN,     │
│    Variety, etc.)           │
│  - Weather API (Open-Meteo) │
│  - Trending topics          │
└─────────────┬───────────────┘
              │
              v
┌─────────────────────────────┐
│  Step 2: SOURCE PREP        │
│  - Bundle stories by topic  │
│  - Create 20 topic docs     │
│  - Add day prompt           │
│    ("Today is Monday...")   │
└─────────────┬───────────────┘
              │
              v
┌─────────────────────────────┐
│  Step 3: AUDIO GENERATION   │
│  - NotebookLM CLI (nlm)    │
│  - Create notebook per      │
│    segment                  │
│  - Add sources + prompt     │
│  - Generate audio overview  │
│  - Download MP3s            │
└─────────────┬───────────────┘
              │
              v
┌─────────────────────────────┐
│  Step 4: FILE DELIVERY      │
│  - Rename with sequential   │
│    naming convention        │
│  - SFTP to AzuraCast        │
│    (port 2022)              │
│  - OR upload via API        │
└─────────────┬───────────────┘
              │
              v
┌─────────────────────────────┐
│  Step 5: PLAYLIST UPDATE    │
│  - AzuraCast API            │
│  - Import M3U to playlist   │
│  - Verify schedule is set   │
│  - 6:00 AM → Show airs      │
└─────────────────────────────┘
```

---

## 3. File Naming Convention

AzuraCast sequential playlists play files in alphabetical/numerical order.

### Format
```
MB_YYYYMMDD_XX_SegmentName.mp3
```

### Example (Full Day - 20 Segments)
```
MB_20260227_01_WakeUp.mp3
MB_20260227_02_Scoreboard.mp3
MB_20260227_03_Buzz.mp3
MB_20260227_04_Forecast.mp3
MB_20260227_05_Roast.mp3
MB_20260227_06_WakeUp.mp3
MB_20260227_07_Scoreboard.mp3
MB_20260227_08_Buzz.mp3
MB_20260227_09_Forecast.mp3
MB_20260227_10_Roast.mp3
MB_20260227_11_WakeUp.mp3
MB_20260227_12_Scoreboard.mp3
MB_20260227_13_Buzz.mp3
MB_20260227_14_Forecast.mp3
MB_20260227_15_Roast.mp3
MB_20260227_16_WakeUp.mp3
MB_20260227_17_Scoreboard.mp3
MB_20260227_18_Buzz.mp3
MB_20260227_19_Forecast.mp3
MB_20260227_20_Roast.mp3
```

The `XX` number ensures AzuraCast plays them in the correct order.

---

## 4. NotebookLM Audio Generation

### Tool: `notebooklm-mcp-cli` (Recommended)

Best option for automation — clean CLI, scriptable, supports audio generation and download.

### Installation (on server 101)
```bash
pip install notebooklm-mcp-cli
# or
uv tool install notebooklm-mcp-cli
```

### First-Time Auth
```bash
nlm login
```
This opens Chrome, you sign into your Google account once. Session persists at `~/.notebooklm-mcp-cli/`.

### Automation Commands (Per Segment)
```bash
# 1. Create a notebook for the segment
nlm notebook create "Morning Brew - Feb 27 - Wake Up H1"

# 2. Add news sources (URLs or raw text)
nlm source add <notebook_id> --url "https://apnews.com/article/..."
nlm source add <notebook_id> --url "https://reuters.com/article/..."
nlm source add <notebook_id> --text "Today is Thursday, February 27th. Discuss these top news stories with energy and banter. Never mention a specific time. Keep it conversational like a morning show."

# 3. Generate audio overview
nlm studio create <notebook_id> --type audio

# 4. Download the audio file
nlm download audio <notebook_id> <artifact_id>
```

### Alternative Tool: `notebooklm-podcast-automator`

If the CLI tool is unreliable, this FastAPI + Playwright option runs as a local server.

```bash
# Install
git clone https://github.com/israelbls/notebooklm-podcast-automator
cd notebooklm-podcast-automator
uv sync && uv run playwright install chromium

# Run
uv run run-server --notebook-url "..." --port 8000

# API calls
POST /api/sources     # Add news sources
POST /api/audio       # Generate audio overview
GET  /api/audio/status  # Check progress
GET  /api/audio/download  # Download MP3
```

### Prompt Template (Baked Into Every Source)

Each segment gets a text source added with the show instructions:

```
SHOW: The Morning Brew on News Mews Radio
DATE: {today's day and date, e.g. "Thursday, February 27th"}
SEGMENT: {segment name, e.g. "Top Headlines"}
HOUR: {hour number}

INSTRUCTIONS FOR HOSTS:
- You are the hosts of The Morning Brew, a lively morning radio show
- Today is {day_of_week}, {month} {day}{suffix}
- NEVER mention a specific time of day
- Be energetic, conversational, and opinionated
- React to each other's points naturally
- Include light humor and banter between stories
- Wrap up with a smooth transition
- Style: Think Fox & Friends meets Good Morning America
- Keep this segment focused on: {segment_topic}

NEWS STORIES TO DISCUSS:
{compiled news content here}
```

---

## 5. News Sources (RSS Feeds)

### Top Headlines / Breaking News
| Source | Feed URL |
|--------|----------|
| AP News (US) | `http://hosted.ap.org/lineups/USHEADS.rss` |
| AP News (World) | `http://hosted.ap.org/lineups/WORLDHEADS.rss` |
| BBC News | `http://feeds.bbci.co.uk/news/rss.xml` |
| NPR News | `https://www.npr.org/rss/rss.php?id=1003` |
| Reuters | `http://feeds.reuters.com/Reuters/domesticNews` |

### Sports
| Source | Feed URL |
|--------|----------|
| ESPN Top | `https://www.espn.com/espn/rss/news` |
| ESPN NFL | `https://www.espn.com/espn/rss/nfl/news` |
| ESPN NBA | `https://www.espn.com/espn/rss/nba/news` |
| CBS Sports | `https://www.cbssports.com/rss/headlines/` |
| Yahoo Sports | `https://sports.yahoo.com/rss/` |

### Entertainment / Pop Culture
| Source | Feed URL |
|--------|----------|
| Variety | `https://variety.com/feed/` |
| Rolling Stone | `https://www.rollingstone.com/feed/` |
| Billboard | `https://www.billboard.com/feed/` |
| E! News | `https://www.eonline.com/syndication/feeds/rssfeeds/topstories.xml` |
| People | `https://people.com/feed/` |

### Lifestyle / Health
| Source | Feed URL |
|--------|----------|
| NPR Health | `https://www.npr.org/rss/rss.php?id=1128` |
| Lifehacker | `https://lifehacker.com/feed/rss` |
| WebMD | `https://rssfeeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC` |

### Weather API (Free, No Key Needed)

**Open-Meteo** — completely free, no registration.

```
https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode&timezone=America/New_York
```

**NWS API** (US only, free, no key):
```
https://api.weather.gov/points/{lat},{lon}
# Follow the "forecast" link in the response
```

---

## 6. AzuraCast Integration

### API Authentication
- Create an API key in AzuraCast: User Menu > My API Keys
- Header: `X-API-Key: {your_key}`
- Station ID: **22** (News Mews Radio)
- Base URL: `https://studiob.loovacast.com`

### Upload Audio Files via API
```bash
curl -X POST 'https://studiob.loovacast.com/api/station/22/files/upload' \
  -H 'X-API-Key: YOUR_KEY' \
  -F 'path="morning-brew/MB_20260227_01_WakeUp.mp3"' \
  -F 'file=@"/tmp/morning-brew/MB_20260227_01_WakeUp.mp3"' \
  -F 'currentDirectory="morning-brew"'
```

### SFTP Upload (Alternative)
- **Host:** studiob.loovacast.com
- **Port:** 2022
- **Protocol:** SFTP
- **User:** Create an SFTP user in AzuraCast (Station > Utilities > SFTP Users)
- Files land in the station's media root — create a `morning-brew/` subdirectory
- AzuraCast auto-detects new files and adds them to the media library

### Create the Morning Brew Playlist (One-Time Setup)
```bash
curl -X POST 'https://studiob.loovacast.com/api/station/22/playlists' \
  -H 'X-API-Key: YOUR_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Morning Brew",
    "type": "default",
    "order": "sequential",
    "source": "songs",
    "is_enabled": true,
    "weight": 5,
    "include_in_automation": true
  }'
```

### Schedule the Playlist (One-Time Setup)
```bash
curl -X POST 'https://studiob.loovacast.com/api/station/22/playlist/{playlist_id}/schedule' \
  -H 'X-API-Key: YOUR_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "start_time": "0600",
    "end_time": "1000",
    "days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
    "loop_once": true
  }'
```

`loop_once: true` means it plays all 20 segments once through, then returns to normal programming.

### Daily Playlist Refresh via M3U Import
```bash
# Generate M3U file with today's segments in order
cat > /tmp/morning-brew.m3u << EOF
morning-brew/MB_20260227_01_WakeUp.mp3
morning-brew/MB_20260227_02_Scoreboard.mp3
morning-brew/MB_20260227_03_Buzz.mp3
morning-brew/MB_20260227_04_Forecast.mp3
morning-brew/MB_20260227_05_Roast.mp3
morning-brew/MB_20260227_06_WakeUp.mp3
morning-brew/MB_20260227_07_Scoreboard.mp3
morning-brew/MB_20260227_08_Buzz.mp3
morning-brew/MB_20260227_09_Forecast.mp3
morning-brew/MB_20260227_10_Roast.mp3
morning-brew/MB_20260227_11_WakeUp.mp3
morning-brew/MB_20260227_12_Scoreboard.mp3
morning-brew/MB_20260227_13_Buzz.mp3
morning-brew/MB_20260227_14_Forecast.mp3
morning-brew/MB_20260227_15_Roast.mp3
morning-brew/MB_20260227_16_WakeUp.mp3
morning-brew/MB_20260227_17_Scoreboard.mp3
morning-brew/MB_20260227_18_Buzz.mp3
morning-brew/MB_20260227_19_Forecast.mp3
morning-brew/MB_20260227_20_Roast.mp3
EOF

# Import into the Morning Brew playlist
curl -X POST 'https://studiob.loovacast.com/api/station/22/playlist/{playlist_id}/import' \
  -H 'X-API-Key: YOUR_KEY' \
  -F 'playlist_file=@"/tmp/morning-brew.m3u"'
```

---

## 7. Automation Options

### Option A: Cron + Python Script (Simplest)

A single Python script on server 101 that runs via cron at 2:00 AM.

```
0 2 * * 1-5 /usr/bin/python3 /home/brucewayne9/.openclaw/workspace/scripts/integrations/morning_brew.py
```

The script handles all 5 steps: gather → prep → generate → rename → upload.

**Pros:** Simple, no extra services, runs on existing server
**Cons:** All sequential, no visual workflow editor

### Option B: n8n Workflow (Visual, More Flexible)

n8n can orchestrate the entire pipeline with a visual workflow.

**Capabilities needed (all supported by n8n):**
- Schedule Trigger (cron at 2:00 AM weekdays)
- HTTP Request nodes (RSS feeds, weather API, AzuraCast API)
- Code nodes (compile news, generate prompts)
- SFTP node (upload to AzuraCast port 2022)
- Wait nodes (poll NotebookLM for audio completion)
- Error handling and notifications

**Workflow:**
```
Schedule Trigger (2:00 AM Mon-Fri)
  → HTTP Request: Fetch 5 RSS feeds in parallel
  → HTTP Request: Fetch weather from Open-Meteo
  → Code Node: Compile 20 topic bundles with prompts
  → Loop: For each of 20 segments:
      → HTTP Request: Create NotebookLM notebook
      → HTTP Request: Add sources + prompt
      → HTTP Request: Trigger audio generation
      → Wait: Poll until audio ready
      → HTTP Request: Download MP3
      → Code Node: Rename file (MB_YYYYMMDD_XX_Name.mp3)
  → SFTP: Upload all 20 files to AzuraCast morning-brew/
  → HTTP Request: Import M3U to Morning Brew playlist
  → HTTP Request (optional): Notify via Telegram that show is ready
```

**Pros:** Visual workflow, easy to modify, retry logic, notifications
**Cons:** Needs n8n instance running (can use existing Dokploy on 117)

### Option C: Hybrid (Recommended)

- **Python script** on 101 handles news gathering + NotebookLM automation (since nlm CLI runs locally)
- **Cron** triggers the script at 2:00 AM
- **Script calls AzuraCast API** directly for upload + playlist refresh
- **Telegram notification** to Mike when show is ready (via Alfred Claw)

This keeps it self-contained on the existing infrastructure with no new services.

---

## 8. Content Rotation Strategy

### Keeping It Fresh Across 4 Hours

Each hour covers the same 5 segment types but with DIFFERENT stories:

| | Hour 1 (6-7 AM) | Hour 2 (7-8 AM) | Hour 3 (8-9 AM) | Hour 4 (9-10 AM) |
|---|---|---|---|---|
| **Wake Up** | Top 3 headlines | Next 3 headlines | Deep dive story 1 | Deep dive story 2 |
| **Scoreboard** | Last night's scores | Trade/signing news | Fantasy/betting | Player spotlight |
| **Buzz** | Biggest celeb story | Music/streaming news | Movie/TV preview | Social media viral |
| **Forecast** | Today's weather | Health tip | Lifestyle hack | Weekend preview |
| **Roast** | Hot take #1 | Hot take #2 | Listener topic | Weekly theme |

### Day-of-Week Themes (Optional)

| Day | Special Segment Flavor |
|-----|----------------------|
| Monday | "Weekend Recap" — sports scores, box office, viral moments |
| Tuesday | "Tech Tuesday" — gadgets, apps, AI news |
| Wednesday | "Hump Day Hot Takes" — controversial opinions |
| Thursday | "Throwback Thursday" — nostalgic topic mixed in |
| Friday | "Friday Vibes" — weekend plans, lighter tone, fun predictions |

---

## 9. Prompt Engineering for Hosts

### Host Personalities

**Host 1 — "The Anchor"**
- Confident, informed, drives the conversation
- Delivers the main story, sets up the topic
- Slightly more serious, keeps things on track

**Host 2 — "The Color"**
- Reactive, opinionated, brings the energy
- Adds hot takes, humor, personal anecdotes
- Challenges Host 1's points, creates natural debate

### Prompt Rules (Applied to Every Segment)
1. Always say the current day of the week and date naturally ("Happy Thursday, February 27th!")
2. NEVER say a specific time ("Good morning at 7:15 AM" — NO)
3. NEVER say "today is [wrong day]" — date must be dynamically injected
4. Keep each segment focused on its topic category
5. End each segment with a natural handoff ("Coming up next..." or "More after the break")
6. Reference the show name naturally ("Here on The Morning Brew...")
7. Occasional references to News Mews Radio
8. Light banter between stories, don't rush through content
9. React to surprising or interesting stories with genuine emotion
10. If covering weather, mention the actual forecast data (injected from API)

---

## 10. Cleanup & Maintenance

### Daily Cleanup (After Show Airs)
- Previous day's audio files can be archived or deleted
- Keep last 7 days for replay/podcast purposes
- AzuraCast playlist gets refreshed daily with new M3U import

### Cleanup Script
```bash
# Delete audio files older than 7 days from AzuraCast morning-brew/ folder
find /path/to/morning-brew/ -name "MB_*.mp3" -mtime +7 -delete
```

### Monitoring
- Telegram notification when show is generated and uploaded
- Telegram alert if any step fails (news fetch, audio generation, upload)
- Log file for debugging: `/tmp/morning-brew/morning_brew_YYYY-MM-DD.log`

### NotebookLM Session Maintenance
- Google auth session may expire — check weekly
- `nlm login` to re-authenticate if needed
- NotebookLM notebooks accumulate — clean up old ones monthly

---

## 11. What Needs to Be Built

### Scripts / Components

| # | Component | Description | Location |
|---|-----------|-------------|----------|
| 1 | `morning_brew.py` | Main orchestrator script — runs the full pipeline | Server 101: `~/.openclaw/workspace/scripts/integrations/` |
| 2 | News gathering module | Fetches RSS feeds + weather, curates stories by topic | Part of morning_brew.py |
| 3 | Prompt builder module | Creates 20 segment prompts with correct day/date | Part of morning_brew.py |
| 4 | NotebookLM automation | Creates notebooks, adds sources, generates + downloads audio | Part of morning_brew.py (uses `nlm` CLI) |
| 5 | AzuraCast uploader | SFTP or API upload + M3U playlist refresh | Part of morning_brew.py |
| 6 | Cron job | Triggers morning_brew.py at 2:00 AM Mon-Fri | Server 101 crontab |
| 7 | Notification handler | Telegram alerts on success/failure | Uses `openclaw message send` |

### AzuraCast Setup (One-Time, Manual)

| # | Task | Details |
|---|------|---------|
| 1 | Create SFTP user | Station 22 > Utilities > SFTP Users |
| 2 | Create `morning-brew/` directory | Via SFTP or media manager |
| 3 | Create "Morning Brew" playlist | Sequential order, scheduled 6-10 AM weekdays |
| 4 | Create API key | User Menu > My API Keys |
| 5 | Upload show jingles/bumpers | Intro, outro, transition sounds between segments |
| 6 | Configure Auto-DJ gaps | Music/jingles play between Morning Brew segments |

### Dependencies to Install (Server 101)

```bash
pip install notebooklm-mcp-cli feedparser paramiko
# feedparser = RSS parsing
# paramiko = SFTP client (if not using API upload)

# First-time NotebookLM auth
nlm login
```

---

## 12. Future Enhancements

- **Podcast feed** — Publish each day's Morning Brew as a podcast episode on LovaCast
- **Weekend edition** — Lighter Saturday show with week recap
- **Listener topics** — Pull topics from Telegram/social media for "The Roast" segment
- **Ad insertion** — Dynamic ad spots between segments (monetization)
- **Multiple shows** — Same pipeline, different prompts = different shows (sports-only, entertainment-only)
- **ElevenLabs upgrade** — Switch to ElevenLabs for custom host voices with more personality control
- **Live breaking news** — Mid-show segment injection if major news breaks
