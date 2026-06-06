# LoovaCast Broadcast-Clock Fix (2026-06-06)

Reference copy of the fix that made the custom AzuraCast **Broadcast Clocks** feature
actually air. Canonical home: GitHub `brucewayne9/loova-ai-dj-breaaks`, file
`backend/src/Radio/AutoDJ/ClockQueueSubscriber.php` (commit a5520ff). This copy is
kept in the Alfred repo for record/backup only — it is **not** wired into the Alfred app.

## The bugs (all confirmed with runtime instrumentation on a test station)

1. **Default builder overrode the clock.** `QueueBuilder::calculateNextSong` (BuildQueue
   priority 0) ran after the clock (priority 3) and replaced the clock's song every time.
   **Fix:** `$event->stopPropagation()` after each successful serve.

2. **Status flush was a silent no-op.** `$item->setStatus(Sent); $em->flush();` emitted no
   UPDATE (entity managed, but Doctrine tracked no change) → items never consumed, clock
   re-served the same item forever. **Fix:** `persistStatus()` helper does a direct SQL
   `UPDATE station_pregenerated_queue ...`.

3. **Queue rows had no media link.** `new StationQueue($station, $media)` only copies song
   metadata, not the media relationship → `media_id` NULL → unplayable. **Fix:** use the
   `StationQueue::fromMedia()` factory.

4. **AI DJ break slot aborted the build.** The break slot called `generateAndPlay` =
   inline TTS during queue-building, which threw and killed the clock after item 0.
   **Fix:** break slot is a no-op marker; DJ talk is handled by `ProcessAiDjBreaksTask`
   on the break's own schedule.

## Deploy
- Dev/source: server 98, `~/azuracast-custom` → `./update.sh --build`.
- Production: server **100** (LoovaCast). Pull GitHub `loova-ai-dj-breaaks` main + build.
- Also bundled in that commit: AI DJ voice-break loudness −16 → −13 LUFS (`TtsService.php`).

## Still open
- Clock-driven break *timing* (talk fires from the clock at exact positions) needs the
  per-break dead-RSS feeds cleaned + `has_pending_trigger` flagging re-enabled.
- MJ RuckTalk 2pm intro-over-bed: needs an intro music asset.
