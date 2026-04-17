"""List Kokoro + Qwen3 voices grouped by provider + rotation/reserved tier."""
from __future__ import annotations

import json
import sys

from scripts.providers.tts.kokoro import KokoroTts
from scripts.providers.tts.qwen3 import Qwen3Tts

# Mirror of theme/providers.ts rotation pools — keep in sync manually.
ROTATION = {
    "kokoro": ["am_adam", "am_michael", "am_eric", "af_sarah", "af_sky", "af_alloy"],
    "qwen3":  ["Barbra_Gordon", "Brenda_Walker", "JAYDEE", "Louis_Lane"],
}
RESERVED = {
    "qwen3": ["MJ"],
}


def annotate(provider: str, voice: str) -> str:
    if voice in RESERVED.get(provider, []): return "[RESERVED]"
    if voice in ROTATION.get(provider, []): return "[rotation]"
    return ""


def main(json_output: bool = False):
    all_voices = {
        "kokoro": list(KokoroTts().list_voices()),
        "qwen3":  list(Qwen3Tts().list_voices()),
    }
    if json_output:
        print(json.dumps(all_voices, indent=2))
        return

    for provider, voices in all_voices.items():
        print(f"\n=== {provider.upper()} ({len(voices)} voices) ===")
        for v in sorted(voices):
            tag = annotate(provider, v)
            print(f"  {v:20} {tag}")


if __name__ == "__main__":
    main(json_output="--json" in sys.argv)
