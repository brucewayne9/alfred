<?php

declare(strict_types=1);

namespace App\Radio\AutoDJ;

use App\Entity\Enums\ClockElementType;
use App\Entity\Enums\PregeneratedQueueStatus;
use App\Entity\Podcast;
use App\Entity\PodcastEpisode;
use App\Entity\StationClock;
use App\Entity\StationClockElement;
use App\Entity\StationPregeneratedQueue;
use App\Entity\StationQueue;
use App\Event\Radio\BuildQueue;
use App\Radio\Adapters;
use App\Radio\Backend\Liquidsoap;
use App\Radio\Enums\LiquidsoapQueues;
use App\Service\AiDjBreaks\AiDjBreakService;
use Doctrine\ORM\EntityManagerInterface;
use Psr\Log\LoggerInterface;
use Symfony\Component\EventDispatcher\EventSubscriberInterface;

final class ClockQueueSubscriber implements EventSubscriberInterface
{
    public function __construct(
        private readonly EntityManagerInterface $em,
        private readonly LoggerInterface $logger,
        private readonly AiDjBreakService $aiDjBreakService,
        private readonly Adapters $adapters,
    ) {
    }

    public static function getSubscribedEvents(): array
    {
        return [
            BuildQueue::class => [
                ['getNextSongFromClock', 3],
            ],
        ];
    }

    public function getNextSongFromClock(BuildQueue $event): void
    {
        $station = $event->getStation();

        // Check if station has clocks enabled
        $clockCount = (int)$this->em->createQueryBuilder()
            ->select('COUNT(c.id)')
            ->from(StationClock::class, 'c')
            ->where('c.station = :station')
            ->setParameter('station', $station)
            ->getQuery()
            ->getSingleScalarResult();

        if ($clockCount === 0) {
            return; // Fall through to default AutoDJ
        }

        // Find next pending item within a reasonable time window
        $expectedPlayTime = $event->getExpectedCueTime();
        $windowEnd = (clone $expectedPlayTime)->modify('+10 minutes');

        /** @var StationPregeneratedQueue|null $nextItem */
        $nextItem = $this->em->createQueryBuilder()
            ->select('q')
            ->from(StationPregeneratedQueue::class, 'q')
            ->where('q.station = :station')
            ->andWhere('q.status = :status')
            ->andWhere('q.scheduled_play_at <= :windowEnd')
            ->setParameter('station', $station)
            ->setParameter('status', PregeneratedQueueStatus::Pending)
            ->setParameter('windowEnd', $windowEnd)
            ->orderBy('q.scheduled_play_at', 'ASC')
            ->setMaxResults(1)
            ->getQuery()
            ->getOneOrNullResult();

        if ($nextItem === null) {
            $this->logger->debug('No pending clock queue items, falling back to default AutoDJ');
            return; // Fall through to default AutoDJ
        }

        $elementType = $nextItem->getElementType();

        switch ($elementType) {
            case ClockElementType::Music:
            case ClockElementType::Sweeper:
            case ClockElementType::TopOfHour:
                $media = $nextItem->getMedia();
                if ($media === null) {
                    $this->logger->warning('Clock queue item has no media, marking as failed', [
                        'item_id' => $nextItem->getId(),
                    ]);
                    $this->persistStatus($nextItem, PregeneratedQueueStatus::Failed);
                    return;
                }

                $queueRow = StationQueue::fromMedia($station, $media);
                $queueRow->playlist = $nextItem->getPlaylist();
                $queueRow->timestamp_cued = $event->getExpectedCueTime();

                $this->persistStatus($nextItem, PregeneratedQueueStatus::Sent);

                $this->logger->info('Clock system providing next song', [
                    'artist' => $nextItem->getArtist(),
                    'title' => $nextItem->getTitle(),
                    'element_type' => $elementType->value,
                ]);

                $event->setNextSongs($queueRow);
                $event->stopPropagation();
                break;

            case ClockElementType::AiDjBreak:
                // Look up the clock element to get the specific AI DJ break config
                $specificBreakId = null;
                $clockElementId = $nextItem->getClockElementId();
                if ($clockElementId !== null) {
                    $clockElement = $this->em->getRepository(StationClockElement::class)->find($clockElementId);
                    $specificBreakId = $clockElement?->getAiDjBreakId();
                }

                // NEVER generate break TTS inline here — it blocks/throws inside the queue
                // builder and aborts the clock. The DJ talk is driven by the existing
                // ProcessAiDjBreaksTask on the break's own schedule (which interrupts over the
                // clock's music). This slot is a no-op marker; the default builder fills it
                // with music. (Clock-DRIVEN break timing via has_pending_trigger is a future
                // enhancement, blocked on per-break RSS/generation reliability.)
                $this->logger->debug('Clock AI DJ break slot (talk handled by break schedule)', [
                    'station' => $station->name,
                    'break_id' => $specificBreakId,
                ]);
                $this->persistStatus($nextItem, PregeneratedQueueStatus::Sent);
                break;

            case ClockElementType::VoiceTrack:
                $audioPath = $nextItem->getAudioFilePath();
                if ($audioPath !== null && file_exists($audioPath)) {
                    // Voice track recording exists - push to Liquidsoap as jingle
                    $duration = $nextItem->getDurationSeconds() ?? 0;
                    if ($duration <= 0) {
                        $probeCmd = sprintf(
                            'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 %s 2>/dev/null',
                            escapeshellarg($audioPath)
                        );
                        $duration = (float) trim(shell_exec($probeCmd) ?? '0');
                    }

                    $title = $nextItem->getTitle() ?? 'Voice Track';
                    $annotation = sprintf(
                        'annotate:title="%s",artist="Voice Track",jingle_mode="true",liq_cue_in="0.0",liq_cue_out="%.3f",liq_cross_duration="0.0":%s',
                        addslashes($title),
                        $duration,
                        $audioPath
                    );

                    $backend = $this->adapters->getBackendAdapter($station);
                    if ($backend instanceof Liquidsoap) {
                        $backend->enqueue($station, LiquidsoapQueues::Requests, $annotation);

                        $this->logger->info('Clock voice track pushed to Liquidsoap', [
                            'item_id' => $nextItem->getId(),
                            'title' => $title,
                            'duration' => $duration,
                        ]);

                        $this->persistStatus($nextItem, PregeneratedQueueStatus::Sent);
                    } else {
                        $this->persistStatus($nextItem, PregeneratedQueueStatus::Failed);
                    }
                } elseif ($nextItem->getMedia() !== null) {
                    // Fallback: media-based voice track
                    $queueRow = StationQueue::fromMedia($station, $nextItem->getMedia());
                    $queueRow->timestamp_cued = $event->getExpectedCueTime();

                    $this->persistStatus($nextItem, PregeneratedQueueStatus::Sent);

                    $event->setNextSongs($queueRow);
                    $event->stopPropagation();
                } else {
                    // No recording - skip slot
                    $this->logger->debug('Voice track slot skipped (no recording)', [
                        'item_id' => $nextItem->getId(),
                    ]);
                    $this->persistStatus($nextItem, PregeneratedQueueStatus::Skipped);
                }
                break;

            case ClockElementType::Podcast:
                // Look up clock element to find the podcast
                $clockElementId = $nextItem->getClockElementId();
                $podcastMedia = null;

                if ($clockElementId !== null) {
                    $clockElement = $this->em->getRepository(StationClockElement::class)->find($clockElementId);
                    $podcastId = $clockElement?->getPodcastId();

                    if ($podcastId !== null) {
                        // Find the next published episode with playlist_media that hasn't been recently played
                        $episode = $this->em->createQueryBuilder()
                            ->select('e')
                            ->from(PodcastEpisode::class, 'e')
                            ->where('e.podcast = :podcast')
                            ->andWhere('e.playlist_media IS NOT NULL')
                            ->setParameter('podcast', $podcastId)
                            ->orderBy('e.created_at', 'DESC')
                            ->setMaxResults(1)
                            ->getQuery()
                            ->getOneOrNullResult();

                        if ($episode instanceof PodcastEpisode) {
                            $podcastMedia = $episode->playlist_media;
                        }
                    }
                }

                if ($podcastMedia !== null) {
                    $queueRow = StationQueue::fromMedia($station, $podcastMedia);
                    $queueRow->timestamp_cued = $event->getExpectedCueTime();

                    $this->persistStatus($nextItem, PregeneratedQueueStatus::Sent);

                    $this->logger->info('Clock podcast slot: queuing episode', [
                        'station' => $station->name,
                        'media_title' => $podcastMedia->title ?? 'unknown',
                    ]);

                    $event->setNextSongs($queueRow);
                    $event->stopPropagation();
                } else {
                    $this->logger->warning('Clock podcast slot: no episode found, skipping', [
                        'station' => $station->name,
                    ]);
                    $this->persistStatus($nextItem, PregeneratedQueueStatus::Skipped);
                }
                break;
        }
    }

    /**
     * Persist a pregenerated-queue item's status via a direct DB write.
     * The ORM flush() silently no-ops for this entity in the BuildQueue dispatch
     * path (managed entity, but no UPDATE emitted), which caused items to never
     * be consumed and the clock to re-serve the same item forever.
     */
    private function persistStatus(StationPregeneratedQueue $item, PregeneratedQueueStatus $status): void
    {
        $item->setStatus($status);

        $sentAt = null;
        if (PregeneratedQueueStatus::Sent === $status) {
            $now = new \DateTime('now');
            $item->setSentToAutodjAt($now);
            $sentAt = $now->format('Y-m-d H:i:s');
        }

        $this->em->getConnection()->executeStatement(
            'UPDATE station_pregenerated_queue SET status = ?, sent_to_autodj_at = COALESCE(?, sent_to_autodj_at) WHERE id = ?',
            [$status->value, $sentAt, $item->getId()]
        );
    }
}
