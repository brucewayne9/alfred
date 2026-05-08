# Lucius — Hermes Agent Test on Server 111

Companion code for the Hermes-Agent-on-111 evaluation against Oracle/OpenClaw on 117. See:
- Spec: `docs/superpowers/specs/2026-05-08-lucius-hermes-on-111-design.md`
- Plan: `docs/superpowers/plans/2026-05-08-lucius-hermes-on-111.md`

## Layout
- `mcp-claw-tools/` — MCP server wrapping 24 day-one integration scripts
- `scripts/` — Promote-queue digest/apply + identity guard
- `skills/` — Hermes skills authored for Lucius (e.g., `propose_memory`)
- `systemd/` — `hermes-gateway.service` unit (deployed to 111)
- `deploy/` — `source_manifest.txt`, `sync_lucius_to_111.sh`
- `logs/` — recon / install / deploy timestamp logs (override of root `.gitignore`'s `logs/`)

## Coordinates
- Telegram: `@Luciuslabsbot` (ID 8750983299, "Lucius Fox")
- Server: 111 (CasaOS dev, brucewayne9 home)
- Brain: `kimi-k2.6:cloud` via 105's Ollama bridge
- Memory layers: `~/.hermes/memories/` native + `~/.lucius/promote_queue.jsonl` for graduation candidates

## Status
- Test window: 2026-05-09 → 2026-05-23 (target)
- Test bar: 3 strikes (fall-back to Oracle/Alfred to finish a task) = fail
- Day-one tool count: 24 (down from 25; website_designer.py deferred — see deploy/source_manifest.txt header for rationale)
