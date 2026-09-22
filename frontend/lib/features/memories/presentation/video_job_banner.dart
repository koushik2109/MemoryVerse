import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/video_job_controller.dart';
import 'package:memory_verse/features/memories/presentation/video_player_screen.dart';

/// Floating banner showing background video generation progress, ETA, and actions.
class VideoJobBanner extends ConsumerWidget {
  const VideoJobBanner({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    try {
      final uiState = ref.watch(videoJobControllerProvider);
      final job = uiState.job;

      final isDirectorActive = ref.watch(aiDirectorActiveProvider);
      if (job == null || uiState.isSheetOpen == true || uiState.isDirectorOpen == true || isDirectorActive) {
        return const SizedBox.shrink();
      }

      final c = context.colors;
      final isProcessing = job.isProcessing == true;
      final isFailed = job.isFailed == true;
      final isCompleted = job.isCompleted == true;

      // Determine banner accent color and icon
      Color accentColor = c.primary;
      IconData icon = Icons.movie_creation_outlined;

      if (isFailed) {
        accentColor = c.error;
        icon = Icons.error_outline_rounded;
      } else if (isCompleted) {
        accentColor = c.success;
        icon = Icons.check_circle_rounded;
      }

      final progressVal = (job.overallProgress) / 100.0;

      return Container(
        margin: const EdgeInsets.symmetric(horizontal: AppSpacing.s12, vertical: AppSpacing.s6),
        decoration: BoxDecoration(
          color: c.surfaceElevated.withValues(alpha: 0.95),
          borderRadius: BorderRadius.circular(AppRadii.lg),
          border: Border.all(
            color: isProcessing ? accentColor.withValues(alpha: 0.4) : c.border,
            width: 1.0,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.25),
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            borderRadius: BorderRadius.circular(AppRadii.lg),
            onTap: () {
              if (isCompleted && job.resultUrl != null) {
                _openCompletedVideo(context, job);
              }
            },
            child: Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: AppSpacing.s12,
                vertical: AppSpacing.s10,
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Row(
                    children: [
                      // Status icon
                      SizedBox(
                        width: 28,
                        height: 28,
                        child: isProcessing
                            ? Icon(Icons.auto_awesome, size: 20, color: accentColor)
                            : Icon(icon, color: accentColor, size: 24),
                      ),
                      const SizedBox(width: AppSpacing.s12),

                      // Text descriptions
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Text(
                                  isCompleted
                                      ? 'Memory Video Ready'
                                      : isFailed
                                          ? 'Creation Paused'
                                          : 'Creating Video (${job.overallProgress}%)',
                                  style: const TextStyle(
                                    fontSize: 13,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                                if (isProcessing && job.formattedEta.isNotEmpty) ...[
                                  const SizedBox(width: AppSpacing.s8),
                                  Text(
                                    job.formattedEta,
                                    style: TextStyle(
                                      fontSize: 11,
                                      color: c.textMuted,
                                      fontWeight: FontWeight.w400,
                                    ),
                                  ),
                                ],
                              ],
                            ),
                            const SizedBox(height: 2),
                            Text(
                              isFailed
                                  ? (job.errorMessage ?? 'Upload or processing failed.')
                                  : job.currentTask,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: TextStyle(fontSize: 11, color: c.textMuted),
                            ),
                          ],
                        ),
                      ),

                      // Action buttons
                      if (isFailed) ...[
                        TextButton.icon(
                          style: TextButton.styleFrom(
                            visualDensity: VisualDensity.compact,
                            padding: const EdgeInsets.symmetric(horizontal: 8),
                            foregroundColor: c.primary,
                          ),
                          icon: const Icon(Icons.refresh_rounded, size: 16),
                          label: const Text('Retry', style: TextStyle(fontSize: 12)),
                          onPressed: () {
                            ref.read(videoJobControllerProvider.notifier).retry();
                          },
                        ),
                      ] else if (isCompleted) ...[
                        FilledButton.icon(
                          style: FilledButton.styleFrom(
                            visualDensity: VisualDensity.compact,
                            padding: const EdgeInsets.symmetric(horizontal: 10),
                            backgroundColor: c.primary,
                          ),
                          icon: const Icon(Icons.play_arrow_rounded, size: 16),
                          label: const Text('Watch', style: TextStyle(fontSize: 12)),
                          onPressed: () => _openCompletedVideo(context, job),
                        ),
                      ] else if (isProcessing) ...[
                        IconButton(
                          icon: const Icon(Icons.close_rounded, size: 18),
                          color: c.textMuted,
                          tooltip: 'Cancel job',
                          onPressed: () {
                            _confirmCancel(context, ref);
                          },
                        ),
                      ],

                      // Close banner for completed/failed
                      if (!isProcessing)
                        IconButton(
                          icon: const Icon(Icons.close_rounded, size: 18),
                          color: c.textMuted,
                          tooltip: 'Dismiss',
                          onPressed: () {
                            ref.read(videoJobControllerProvider.notifier).dismiss();
                          },
                        ),
                    ],
                  ),

                  // Linear progress indicator for active jobs
                  if (isProcessing) ...[
                    const SizedBox(height: 8),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(2),
                      child: LinearProgressIndicator(
                        value: progressVal > 0 ? progressVal : null,
                        minHeight: 3,
                        color: accentColor,
                        backgroundColor: c.border.withValues(alpha: 0.5),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      );
    } catch (e, stack) {
      debugPrint('VideoJobBanner build error: $e\n$stack');
      return const SizedBox.shrink();
    }
  }

  void _openCompletedVideo(BuildContext context, VideoJobModel job) {
    if (job.resultUrl != null) {
      final dummyMedia = MediaModel(
        id: job.resultMediaId ?? job.id,
        memoryId: job.memoryId ?? '',
        ownerId: job.userId ?? '',
        filename: 'memory_reel_${job.id}.mp4',
        storagePath: 'reels/${job.id}.mp4',
        url: job.resultUrl!,
        thumbnailUrl: null,
        mediaType: 'video',
        fileSize: 0,
        createdAt: DateTime.now(),
      );
      VideoPlayerScreen.open(context, dummyMedia);
    }
  }

  void _confirmCancel(BuildContext context, WidgetRef ref) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Cancel Video Creation?'),
        content: const Text('Are you sure you want to cancel video generation?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Keep Generating'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: Theme.of(context).colorScheme.error),
            onPressed: () {
              Navigator.pop(ctx);
              ref.read(videoJobControllerProvider.notifier).cancel();
            },
            child: const Text('Cancel Job'),
          ),
        ],
      ),
    );
  }
}
