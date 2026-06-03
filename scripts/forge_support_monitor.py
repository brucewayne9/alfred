#!/usr/bin/env python3
"""Forge Support "Crown" monitor — auto-answer the Mainstay team's Forge questions.

THE CROWN = an authorization trigger. When a *crowned* sender (the Forge team)
emails alfred@groundrushinc.com about Mainstay Forge, Alfred drafts an answer
grounded in the Forge manual + Grey Matter, replies to them, and ALWAYS CCs Mike.
Anything Alfred can't answer confidently is deferred to Mike (the reply loops him
in rather than guessing).

Run:  python3 scripts/forge_support_monitor.py            (one pass, live)
      python3 scripts/forge_support_monitor.py --dry-run  (draft, don't send)
Cron: */15 (registered via --ensure-cron).

Guardrails (per the inbox-monitor rules): crowned-sender whitelist (no free-email
domains can trigger it), SINCE window, per-run cap, state file to avoid double
replies.
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, "/home/aialfred/alfred")
import requests
from dotenv import load_dotenv

load_dotenv("/home/aialfred/alfred/config/.env")
from integrations.email.client import EmailClient  # noqa: E402

ACCOUNT = "alfred-gw"                              # alfred@groundrushinc.com
MIKE = "mjohnson@groundrushinc.com"               # always CC'd
STATE_FILE = Path("/home/aialfred/alfred/data/forge_support_monitor_state.json")
MANUAL_URL = "https://aialfred.groundrushcloud.com/clips/forge-manual.html"
LLM_URL = "http://75.43.156.105:11434/v1/chat/completions"
LLM_MODEL = "gemma4:31b-cloud"
MAX_REPLIES_PER_RUN = 5
SINCE_DAYS = 14

# THE CROWN — only these senders can trigger an auto-reply.
CROWNED = {
    "jordan@mainstaymusicgroup.com",
    "mello@mainstaymusicgroup.com",
    "markion@mainstaymusicgroup.com",
    "dharmic@mainstaymusicgroup.com",
    "zo@mainstaymusicgroup.com",
    MIKE,
}
# It must clearly be about Forge. The team replies to the manual/login thread
# (subject contains "Mainstay Forge"), so requiring the word "forge" is a tight,
# reliable gate that avoids false-matching unrelated team mail.
def is_forge_topic(subject: str, body: str) -> bool:
    return "forge" in f"{subject} {body}".lower()

# Plain-text grounding — the facts Alfred may answer from.
MANUAL_FACTS = """\
MAINSTAY FORGE — what it is and how it works (the only facts you may answer from):
- Forge is Mainstay's in-house clip factory at forge.groundrushcloud.com. Each person has their own login; it records who made each clip. Logins are added/removed by Mike only.
- Tabs: Create, Topic, Queue, Library, Distribute, and Intel (Intel is NOT switched on yet — Mike is still setting it up).
- CREATE (build from scratch): (A) Leak Graphic — title + tracklist + caption; you can upload your OWN graphic and Forge bakes the text over it, or leave it empty to AI-generate. (B) Kinetic Lyric — upload an audio hook + type the lyric -> word-synced lyric video. (C) Film Montage — list YouTube/TikTok/IG links or 'search: ...' + upload a song hook -> auto fast-cut montage as long as the hook. Shared options: caption font, deliver-to folder, remix looks, variations.
- TOPIC (cut clips from a long video): 1) Add a source — paste a YouTube/TikTok/Instagram LINK and hit Fetch & transcribe, OR drop a big file in the drop folder. 2) Search the topic in plain words. 3) Pick & trim the moments. 4) Caption look: style (Clean / Karaoke word-by-word / Bold caps), position (lower/center/upper), font, color. 5) Assemble variants. Montage tray: pick moments across MULTIPLE sources, drag to reorder, optional song bed that ducks under the voice (or replaces it), then Assemble montage.
- WHERE THINGS GO: you hit Forge it/Assemble -> Queue (rendering) -> Library (finished, preview/download) -> Distribute pushes to Postiz as DRAFTS. A human always hits send; Forge never auto-posts. Everything saves to the shared Mainstay cloud.
- RULES: every clip is vertical 9:16, 10-60 seconds, branded. YouTube/TikTok/IG are sources to pull from, never post destinations.
- BE THOUGHTFUL: each clip uses real compute/AI credits — make deliberate clips, don't spam. A few strong variants beat a hundred throwaways.
- A link that won't load is usually age-locked/private — download it and use the drop folder. A wrong caption word is the transcript guessing slang.
"""

SYSTEM = (
    "You are Alfred, Mike Johnson's assistant, helping the Mainstay social team "
    "(Jordan, Mello, Dharmic and Mike) use Mainstay Forge, our video clip tool. "
    "Answer ONLY using the FACTS provided. Be warm, brief, and plain — like a helpful "
    "colleague, a few short sentences. Do NOT invent features or steps. "
    "If the question is not clearly covered by the FACTS, or it's a bug/account/access "
    "issue, or you are at all unsure, do NOT guess: reply with exactly the single token "
    "DEFER on its own line and nothing else. Never share passwords, links to internal "
    "systems, or anything not in the FACTS. Sign off as 'Alfred'."
)


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"seen_ids": [], "last_run": None}


def save_state(state):
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["seen_ids"] = state["seen_ids"][-500:]
    STATE_FILE.write_text(json.dumps(state, indent=2))


def addr(raw: str) -> str:
    m = re.search(r"[\w.+-]+@[\w.-]+", raw or "")
    return m.group(0).lower() if m else ""


def _recent(date_str: str) -> bool:
    """True if the message date is within SINCE_DAYS (skips old mail)."""
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(date_str)
        if dt is None:
            return True
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt >= datetime.now(timezone.utc) - timedelta(days=SINCE_DAYS)
    except Exception:
        return True


def grey_matter(question: str) -> str:
    try:
        r = subprocess.run(
            ["python3", "/home/aialfred/.openclaw/workspace/scripts/integrations/lightrag_client.py",
             "recall", f"Mainstay Forge: {question}"],
            capture_output=True, text=True, timeout=60,
        )
        return (r.stdout or "")[:2500]
    except Exception:
        return ""


def draft_answer(question: str, gm: str) -> str | None:
    """Return the answer text, or None if Alfred should defer to Mike."""
    facts = MANUAL_FACTS + ("\n\nADDITIONAL KNOWLEDGE (from memory):\n" + gm if gm.strip() else "")
    try:
        resp = requests.post(LLM_URL, json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"FACTS:\n{facts}\n\nTEAM QUESTION:\n{question}"},
            ],
            "temperature": 0.2,
        }, timeout=120)
        resp.raise_for_status()
        out = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("  LLM error:", e)
        return None
    if out.upper().startswith("DEFER") or out.strip().upper() == "DEFER":
        return None
    return out


def send_reply(to_addr: str, subject: str, answer: str | None, question: str, dry: bool):
    ec = EmailClient()
    re_subj = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    if answer:
        body = (
            f"{answer}\n\n"
            f"Full manual any time: {MANUAL_URL}\n"
            "If anything's off or you hit a snag, just reply — Mike's copied here too.\n\n"
            "— Alfred"
        )
    else:
        # Deferral — loop Mike in rather than guess.
        body = (
            "Good question — let me get you the right answer on that rather than guess. "
            "I've looped in Mike (copied here) and he'll follow up.\n\n"
            f"In the meantime the full manual is here: {MANUAL_URL}\n\n"
            "— Alfred"
        )
    if dry:
        print(f"  [DRY] -> {to_addr} (cc {MIKE})\n  SUBJ: {re_subj}\n  BODY:\n{body}\n")
        return
    ec.send_email(ACCOUNT, to_addr, re_subj, body, html=False, cc=[MIKE])
    tag = "answered" if answer else "DEFERRED to Mike"
    print(f"  replied ({tag}) -> {to_addr}, cc {MIKE}")


def run(dry: bool = False):
    state = load_state()
    seen = set(state["seen_ids"])
    ec = EmailClient()
    # Pull recent team mail (FROM the mainstay domain), then filter to crowned senders.
    res = ec.search_emails(ACCOUNT, "mainstaymusicgroup.com", limit=30)
    msgs = res.get("messages", []) if isinstance(res, dict) else []
    # Also catch Mike's own test emails.
    res2 = ec.search_emails(ACCOUNT, MIKE, limit=10)
    msgs += res2.get("messages", []) if isinstance(res2, dict) else []

    # Collect the messages that actually qualify (crowned + recent + Forge).
    candidates = []
    for m in msgs:
        mid = str(m.get("id"))
        if not mid or mid in seen:
            continue
        sender = addr(m.get("from", ""))
        subject = m.get("subject", "") or ""
        if sender not in CROWNED:
            continue
        if not _recent(m.get("date", "")):
            continue
        full = ec.read_email(ACCOUNT, mid)
        body = (full.get("body") or "")[:4000] if isinstance(full, dict) else ""
        if not is_forge_topic(subject, body):
            continue
        candidates.append((mid, sender, subject, body))

    # First run: set a baseline — mark everything currently in the inbox as seen
    # WITHOUT replying, so we only ever answer mail that arrives after setup.
    if not state.get("baseline"):
        for mid, *_ in candidates:
            seen.add(mid)
        state["baseline"] = True
        state["seen_ids"] = list(seen)
        if not dry:
            save_state(state)
        print(f"baseline set — {len(candidates)} existing message(s) marked seen, no replies sent")
        return

    fired = 0
    for mid, sender, subject, body in candidates:
        if fired >= MAX_REPLIES_PER_RUN:
            print("  per-run cap reached; stopping")
            break
        question = f"Subject: {subject}\n\n{body}".strip()
        print(f"CROWNED Forge question from {sender}: {subject[:60]}")
        gm = grey_matter(subject + " " + body[:300])
        answer = draft_answer(question, gm)
        send_reply(sender, subject, answer, question, dry)
        seen.add(mid)
        fired += 1

    state["seen_ids"] = list(seen)
    if not dry:
        save_state(state)
    print(f"done — {fired} handled this run")


def ensure_cron():
    line = "*/15 * * * * cd /home/aialfred/alfred && /home/aialfred/alfred/venv/bin/python scripts/forge_support_monitor.py >> data/forge_support_monitor.log 2>&1"
    cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    if "forge_support_monitor.py" in cur:
        print("cron already present")
        return
    new = (cur.rstrip("\n") + "\n" + line + "\n") if cur.strip() else line + "\n"
    subprocess.run(["crontab", "-"], input=new, text=True)
    print("cron installed (*/15)")


if __name__ == "__main__":
    if "--ensure-cron" in sys.argv:
        ensure_cron()
    else:
        run(dry="--dry-run" in sys.argv)
