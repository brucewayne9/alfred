#!/usr/bin/env python3
"""Design review tool — screenshot + vision AI analysis with brand guidelines.

Usage: python3 design_review.py <command> [args]

Commands:
  review <url>                    Full design review (desktop screenshot + vision analysis)
  review <url> --mobile           Review mobile layout
  review <url> --tablet           Review tablet layout
  review <url> --all              Review all viewports (desktop + tablet + mobile)
  compare <url1> <url2>           Compare two pages/versions side-by-side
  audit <url>                     Full brand audit (all viewports + sections + guidelines check)
  check <image_path>              Analyze an existing screenshot/image file
  guidelines                      Show current brand guidelines from Grey Matter

Options (append to any command):
  --mobile                Mobile viewport (390x844)
  --tablet                Tablet viewport (768x1024)
  --all                   All viewports
  --focus <area>          Focus analysis on: layout|colors|typography|spacing|cta|accessibility
  --brand <query>         Custom brand guidelines query for Grey Matter
  --model <name>          Vision model (default: qwen3-vl:235b-cloud)
  --save <path>           Save report to file
  --no-screenshot         Skip screenshot (use with --check for existing images)
  --delay <ms>            Wait before screenshot in ms (default: 3000)
"""

import sys
import os
import json
import base64
import time
import asyncio
from pathlib import Path
from datetime import datetime
from openclaw_env import env as _env

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("ERROR: playwright not installed. Run: pip3 install playwright && playwright install chromium")
    sys.exit(1)

try:
    import httpx
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
    import httpx


# --- Config ---
OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434")
VISION_MODEL = _env("VISION_MODEL", "qwen3-vl:235b-cloud")
LIGHTRAG_HOST = _env("LIGHTRAG_HOST", "http://75.43.156.117:9621")
SCREENSHOTS_DIR = os.path.expanduser("~/.openclaw/workspace/screenshots/design_reviews")
REPORTS_DIR = os.path.expanduser("~/.openclaw/workspace/design_reports")

VIEWPORTS = {
    "desktop": {"width": 1920, "height": 1080, "scale": 1, "ua": None},
    "tablet": {"width": 768, "height": 1024, "scale": 2, "ua": "Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) AppleWebKit/605.1.15"},
    "mobile": {"width": 390, "height": 844, "scale": 3, "ua": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15"},
}

DEFAULT_BRAND_QUERY = "brand guidelines colors fonts typography logo usage design standards style guide"

REVIEW_PROMPT = """You are a senior UI/UX designer and brand consultant. Analyze this website screenshot and provide a detailed design review.

{brand_context}

Evaluate the following aspects:
1. **Layout & Structure** — Grid alignment, visual hierarchy, content flow, whitespace usage
2. **Color Palette** — Color harmony, contrast ratios, brand color consistency, accessibility
3. **Typography** — Font choices, sizing hierarchy, readability, line spacing
4. **Visual Elements** — Images, icons, illustrations quality and consistency
5. **CTAs & Buttons** — Visibility, placement, color contrast, action clarity
6. **Responsiveness** — How well the layout adapts (if multiple viewports provided)
7. **Brand Consistency** — Alignment with brand guidelines (if provided)
8. **Accessibility** — Text contrast, touch targets, alt text indicators

For each aspect, provide:
- Current state (what you see)
- Issues found (if any)
- Specific recommendations

End with a **Design Score** (1-10) and **Top 3 Priority Fixes**.

{focus_instruction}"""

COMPARE_PROMPT = """You are a senior UI/UX designer. Compare these two website screenshots and provide a detailed comparison.

{brand_context}

For each screenshot, analyze:
1. Layout & structure differences
2. Color and typography choices
3. Visual hierarchy effectiveness
4. CTA placement and visibility
5. Overall design quality

Provide:
- **Winner** for each category with reasoning
- **Overall recommendation** on which design is stronger
- **Specific improvements** each could adopt from the other"""

FOCUS_INSTRUCTIONS = {
    "layout": "Focus primarily on layout structure, grid alignment, spacing, and content flow.",
    "colors": "Focus primarily on color palette, harmony, contrast ratios, and brand color usage.",
    "typography": "Focus primarily on font choices, sizing hierarchy, readability, and text styling.",
    "spacing": "Focus primarily on whitespace, padding, margins, and element spacing consistency.",
    "cta": "Focus primarily on call-to-action buttons, their visibility, placement, and effectiveness.",
    "accessibility": "Focus primarily on accessibility: contrast ratios, text sizes, touch targets, and WCAG compliance.",
}


def ensure_dirs():
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def take_screenshot(url, viewport="desktop", delay=3000):
    """Take a screenshot and return the file path."""
    ensure_dirs()
    vp = VIEWPORTS[viewport]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    domain = url.replace("https://", "").replace("http://", "").split("/")[0].replace(".", "_")
    output = os.path.join(SCREENSHOTS_DIR, f"{domain}_{viewport}_{ts}.png")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx_opts = {"viewport": {"width": vp["width"], "height": vp["height"]}}
        if vp["ua"]:
            ctx_opts["user_agent"] = vp["ua"]
        if vp["scale"] > 1:
            ctx_opts["device_scale_factor"] = vp["scale"]

        context = browser.new_context(**ctx_opts)
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(delay)
        page.screenshot(path=output, full_page=True)
        context.close()
        browser.close()

    return output


def image_to_base64(image_path):
    """Read an image file and return base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_lightrag(query_text):
    """Query LightRAG for brand guidelines context."""
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                f"{LIGHTRAG_HOST}/query",
                json={"query": query_text, "mode": "hybrid", "only_need_context": True},
            )
            if resp.status_code == 200:
                data = resp.json()
                context = data.get("response", "")
                if context and len(context.strip()) > 20:
                    return context[:4000]
        return ""
    except Exception:
        return ""


def vision_analyze(images_b64, prompt, model=None):
    """Send images to Ollama vision model for analysis."""
    model = model or VISION_MODEL

    with httpx.Client(timeout=300) as client:
        resp = client.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": images_b64,
                    }
                ],
                "stream": False,
            },
        )

        if resp.status_code != 200:
            return {"error": f"Ollama returned {resp.status_code}: {resp.text}"}

        data = resp.json()
        return {
            "analysis": data.get("message", {}).get("content", ""),
            "model": model,
            "eval_duration_ms": data.get("eval_duration", 0) // 1_000_000,
        }


def build_prompt(template, brand_query=None, focus=None):
    """Build the analysis prompt with brand context and focus."""
    brand_context = ""
    brand_query = brand_query or DEFAULT_BRAND_QUERY

    guidelines = query_lightrag(brand_query)
    if guidelines:
        brand_context = f"## Brand Guidelines (from knowledge base):\n{guidelines}\n\nUse these guidelines to evaluate brand consistency."
    else:
        brand_context = "(No brand guidelines found in knowledge base. Provide general design feedback.)"

    focus_instruction = ""
    if focus and focus in FOCUS_INSTRUCTIONS:
        focus_instruction = f"\n**FOCUS AREA:** {FOCUS_INSTRUCTIONS[focus]}"

    return template.format(brand_context=brand_context, focus_instruction=focus_instruction)


def parse_opts(args):
    """Parse global options from args."""
    opts = {
        "model": None,
        "focus": None,
        "brand": None,
        "save": None,
        "delay": 3000,
        "mobile": False,
        "tablet": False,
        "all": False,
    }
    remaining = []
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            opts["model"] = args[i + 1]; i += 2
        elif args[i] == "--focus" and i + 1 < len(args):
            opts["focus"] = args[i + 1]; i += 2
        elif args[i] == "--brand" and i + 1 < len(args):
            opts["brand"] = args[i + 1]; i += 2
        elif args[i] == "--save" and i + 1 < len(args):
            opts["save"] = args[i + 1]; i += 2
        elif args[i] == "--delay" and i + 1 < len(args):
            opts["delay"] = int(args[i + 1]); i += 2
        elif args[i] == "--mobile":
            opts["mobile"] = True; i += 1
        elif args[i] == "--tablet":
            opts["tablet"] = True; i += 1
        elif args[i] == "--all":
            opts["all"] = True; i += 1
        else:
            remaining.append(args[i]); i += 1
    return remaining, opts


def get_viewports(opts):
    """Determine which viewports to screenshot."""
    if opts["all"]:
        return ["desktop", "tablet", "mobile"]
    viewports = []
    if opts["mobile"]:
        viewports.append("mobile")
    if opts["tablet"]:
        viewports.append("tablet")
    if not viewports:
        viewports.append("desktop")
    return viewports


def save_report(result, opts):
    """Save report to file if --save specified."""
    if opts.get("save"):
        path = opts["save"]
    else:
        ensure_dirs()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(REPORTS_DIR, f"review_{ts}.json")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    result["report_saved"] = path


# --- Commands ---

def cmd_review(args, opts):
    """Review a URL's design."""
    if not args:
        print("Usage: review <url> [--mobile|--tablet|--all] [--focus area]"); sys.exit(1)

    url = args[0]
    viewports = get_viewports(opts)
    prompt = build_prompt(REVIEW_PROMPT, brand_query=opts["brand"], focus=opts["focus"])

    screenshots = []
    images_b64 = []

    for vp in viewports:
        print(f"Capturing {vp} screenshot of {url}...", file=sys.stderr)
        path = take_screenshot(url, viewport=vp, delay=opts["delay"])
        screenshots.append({"viewport": vp, "path": path, "size_kb": round(os.path.getsize(path) / 1024, 1)})
        images_b64.append(image_to_base64(path))

    viewport_note = ", ".join(viewports)
    full_prompt = f"Viewport(s): {viewport_note}\nURL: {url}\n\n{prompt}"

    print(f"Analyzing with {opts['model'] or VISION_MODEL}...", file=sys.stderr)
    result = vision_analyze(images_b64, full_prompt, model=opts["model"])

    output = {
        "status": "ok",
        "url": url,
        "viewports": viewports,
        "screenshots": screenshots,
        "model": result.get("model", VISION_MODEL),
        "analysis": result.get("analysis", ""),
        "eval_duration_ms": result.get("eval_duration_ms", 0),
    }

    if result.get("error"):
        output["status"] = "error"
        output["error"] = result["error"]

    save_report(output, opts)
    print(json.dumps(output, indent=2))


def cmd_compare(args, opts):
    """Compare two URLs side-by-side."""
    if len(args) < 2:
        print("Usage: compare <url1> <url2> [--focus area]"); sys.exit(1)

    url1, url2 = args[0], args[1]
    viewport = get_viewports(opts)[0]
    prompt = build_prompt(COMPARE_PROMPT, brand_query=opts["brand"], focus=opts["focus"])

    print(f"Capturing {viewport} screenshots...", file=sys.stderr)
    path1 = take_screenshot(url1, viewport=viewport, delay=opts["delay"])
    path2 = take_screenshot(url2, viewport=viewport, delay=opts["delay"])

    images_b64 = [image_to_base64(path1), image_to_base64(path2)]
    full_prompt = f"Image 1 (A): {url1}\nImage 2 (B): {url2}\nViewport: {viewport}\n\n{prompt}"

    print(f"Comparing with {opts['model'] or VISION_MODEL}...", file=sys.stderr)
    result = vision_analyze(images_b64, full_prompt, model=opts["model"])

    output = {
        "status": "ok",
        "url_a": url1,
        "url_b": url2,
        "viewport": viewport,
        "screenshots": [
            {"label": "A", "url": url1, "path": path1},
            {"label": "B", "url": url2, "path": path2},
        ],
        "model": result.get("model", VISION_MODEL),
        "analysis": result.get("analysis", ""),
    }

    if result.get("error"):
        output["status"] = "error"
        output["error"] = result["error"]

    save_report(output, opts)
    print(json.dumps(output, indent=2))


def cmd_audit(args, opts):
    """Full brand audit — all viewports + section analysis."""
    if not args:
        print("Usage: audit <url>"); sys.exit(1)

    url = args[0]
    opts["all"] = True
    viewports = get_viewports(opts)
    prompt = build_prompt(REVIEW_PROMPT, brand_query=opts["brand"], focus=None)

    screenshots = []
    images_b64 = []

    for vp in viewports:
        print(f"Capturing {vp} screenshot...", file=sys.stderr)
        path = take_screenshot(url, viewport=vp, delay=opts["delay"])
        screenshots.append({"viewport": vp, "path": path, "size_kb": round(os.path.getsize(path) / 1024, 1)})
        images_b64.append(image_to_base64(path))

    audit_prompt = f"""URL: {url}
Viewports provided: desktop, tablet, mobile (in that order)

{prompt}

Additionally for this full audit, evaluate:
- **Cross-viewport consistency** — Do design elements adapt well across all screen sizes?
- **Mobile-first assessment** — Is the mobile experience prioritized?
- **Critical rendering path** — What does the user see first on each viewport?

Provide a **Brand Compliance Score** (1-10) and a **Responsive Design Score** (1-10) alongside the overall Design Score."""

    print(f"Running full brand audit with {opts['model'] or VISION_MODEL}...", file=sys.stderr)
    result = vision_analyze(images_b64, audit_prompt, model=opts["model"])

    output = {
        "status": "ok",
        "type": "full_audit",
        "url": url,
        "viewports": viewports,
        "screenshots": screenshots,
        "model": result.get("model", VISION_MODEL),
        "analysis": result.get("analysis", ""),
        "eval_duration_ms": result.get("eval_duration_ms", 0),
    }

    if result.get("error"):
        output["status"] = "error"
        output["error"] = result["error"]

    save_report(output, opts)
    print(json.dumps(output, indent=2))


def cmd_check(args, opts):
    """Analyze an existing image file."""
    if not args:
        print("Usage: check <image_path> [--focus area]"); sys.exit(1)

    image_path = args[0]
    if not os.path.exists(image_path):
        print(f"ERROR: File not found: {image_path}"); sys.exit(1)

    prompt = build_prompt(REVIEW_PROMPT, brand_query=opts["brand"], focus=opts["focus"])
    full_prompt = f"Image: {os.path.basename(image_path)}\n\n{prompt}"

    images_b64 = [image_to_base64(image_path)]

    print(f"Analyzing with {opts['model'] or VISION_MODEL}...", file=sys.stderr)
    result = vision_analyze(images_b64, full_prompt, model=opts["model"])

    output = {
        "status": "ok",
        "image": image_path,
        "model": result.get("model", VISION_MODEL),
        "analysis": result.get("analysis", ""),
    }

    if result.get("error"):
        output["status"] = "error"
        output["error"] = result["error"]

    save_report(output, opts)
    print(json.dumps(output, indent=2))


def cmd_guidelines(args, opts):
    """Show current brand guidelines from Grey Matter."""
    query = opts["brand"] or DEFAULT_BRAND_QUERY
    print(f"Querying Grey Matter for brand guidelines...", file=sys.stderr)

    guidelines = query_lightrag(query)

    if guidelines:
        output = {"status": "ok", "query": query, "guidelines": guidelines}
    else:
        output = {"status": "empty", "query": query, "message": "No brand guidelines found in Grey Matter. Upload brand docs to LightRAG first."}

    print(json.dumps(output, indent=2))


# --- Dispatcher ---

COMMANDS = {
    "review": cmd_review,
    "compare": cmd_compare,
    "audit": cmd_audit,
    "check": cmd_check,
    "guidelines": cmd_guidelines,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"ERROR: Unknown command: {cmd}")
        print(f"Available: {', '.join(sorted(COMMANDS.keys()))}")
        sys.exit(1)

    remaining, opts = parse_opts(sys.argv[2:])
    COMMANDS[cmd](remaining, opts)


if __name__ == "__main__":
    main()
