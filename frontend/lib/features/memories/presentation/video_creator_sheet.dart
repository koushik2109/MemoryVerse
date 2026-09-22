import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/providers/video_job_controller.dart';
import 'package:memory_verse/features/memories/presentation/video_player_screen.dart';

class VideoCreatorSheet extends ConsumerStatefulWidget {
  final MemoryModel memory;

  const VideoCreatorSheet({super.key, required this.memory});

  static void show(BuildContext context, MemoryModel memory) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => VideoCreatorSheet(memory: memory),
    );
  }

  @override
  ConsumerState<VideoCreatorSheet> createState() => _VideoCreatorSheetState();
}

class _VideoCreatorSheetState extends ConsumerState<VideoCreatorSheet> {
  String _dimension = '9:16'; // '9:16' or '16:9'
  String _mood = 'calm'; // 'calm', 'nostalgic', 'energetic'
  late Set<String> _selectedMediaIds;
  bool _jobStartedLocally = false;

  int get _photoCount =>
      widget.memory.media.where((m) => m.mediaType == 'image').length;
  int get _videoCount =>
      widget.memory.media.where((m) => m.mediaType == 'video').length;
  int get _totalCount => widget.memory.media.length;

  bool get _isEligible => _totalCount >= 1;

  String get _ineligibilityReason {
    if (_totalCount == 0) return 'Add at least one photo or video to create a Memory Video.';
    return '';
  }

  @override
  void initState() {
    super.initState();
    _selectedMediaIds = widget.memory.media.map((m) => m.id).toSet();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        ref.read(videoJobControllerProvider.notifier).setSheetOpen(true);
      }
    });
  }

  @override
  void dispose() {
    ref.read(videoJobControllerProvider.notifier).setSheetOpen(false);
    super.dispose();
  }

  Future<void> _startGeneration() async {
    if (!_isEligible) return;

    setState(() {
      _jobStartedLocally = true;
    });

    final jobId = await ref.read(videoJobControllerProvider.notifier).startJob(
          memoryId: widget.memory.id,
          dimension: _dimension,
          mood: _mood,
          mediaIds: _selectedMediaIds.toList(),
        );

    if (jobId == null && mounted) {
      setState(() {
        _jobStartedLocally = false;
      });
    }
  }

  void _watchCompletedVideo(VideoJobModel job) {
    Navigator.pop(context);
    ref.invalidate(memoryDetailProvider(widget.memory.id));

    if (job.resultUrl != null) {
      final mediaItem = MediaModel(
        id: job.resultMediaId ?? job.id,
        memoryId: widget.memory.id,
        ownerId: job.userId ?? '',
        filename: 'memory_reel_${job.id}.mp4',
        storagePath: 'reels/${job.id}.mp4',
        url: job.resultUrl!,
        thumbnailUrl: null,
        mediaType: 'video',
        fileSize: 0,
        createdAt: DateTime.now(),
      );
      VideoPlayerScreen.open(context, mediaItem);
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Video creation complete! Refreshing memory...')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final uiState = ref.watch(videoJobControllerProvider);
    final activeJob = uiState.job;

    // Check if this sheet is associated with an ongoing job for this memory
    final hasOngoingJob = activeJob != null &&
        (activeJob.memoryId == widget.memory.id || _jobStartedLocally);

    return Container(
      padding: EdgeInsets.only(
        left: AppSpacing.s24,
        right: AppSpacing.s24,
        top: AppSpacing.s24,
        bottom: MediaQuery.of(context).viewInsets.bottom + AppSpacing.s32,
      ),
      decoration: BoxDecoration(
        color: c.bg,
        borderRadius: const BorderRadius.vertical(
          top: Radius.circular(AppRadii.xl),
        ),
      ),
      child: SafeArea(
        top: false,
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: c.border,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(height: AppSpacing.s24),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.auto_awesome, color: c.primary, size: 24),
                  const SizedBox(width: AppSpacing.s8),
                  Text(
                    'AI Memory Video',
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.w700,
                        ),
                  ),
                ],
              ),
              const SizedBox(height: AppSpacing.s8),
              Text(
                'Directing Ken Burns camera motions, evidence subtitles, and ambient 4-chord soundtrack.',
                textAlign: TextAlign.center,
                style: TextStyle(color: c.textMuted, fontSize: 13),
              ),
              const SizedBox(height: AppSpacing.s24),

              if (!hasOngoingJob) ...[
                // Creation Configuration View
                if (!_isEligible) ...[
                  Container(
                    padding: const EdgeInsets.all(AppSpacing.s16),
                    decoration: BoxDecoration(
                      color: c.surfaceElevated,
                      borderRadius: BorderRadius.circular(AppRadii.lg),
                      border: Border.all(color: c.border),
                    ),
                    child: Row(
                      children: [
                        Icon(Icons.info_outline, color: c.primary, size: 24),
                        const SizedBox(width: AppSpacing.s12),
                        Expanded(
                          child: Text(
                            _ineligibilityReason,
                            style: TextStyle(color: c.textMuted, fontSize: 13),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: AppSpacing.s24),
                ],

                // Overview stats
                Container(
                  padding: const EdgeInsets.all(AppSpacing.s16),
                  decoration: BoxDecoration(
                    color: c.surfaceElevated,
                    borderRadius: BorderRadius.circular(AppRadii.lg),
                    border: Border.all(color: c.border),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      _StatCount(count: _photoCount, label: 'Photos', icon: Icons.photo_outlined, c: c),
                      Container(width: 1, height: 40, color: c.border),
                      _StatCount(count: _videoCount, label: 'Videos', icon: Icons.videocam_outlined, c: c),
                      Container(width: 1, height: 40, color: c.border),
                      _StatCount(count: _selectedMediaIds.length, label: 'Selected', icon: Icons.check_circle_outlined, c: c),
                    ],
                  ),
                ),
                const SizedBox(height: AppSpacing.s24),

                // Aspect Ratio Selector
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    'Aspect Ratio',
                    style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: c.text),
                  ),
                ),
                const SizedBox(height: AppSpacing.s8),
                Row(
                  children: [
                    Expanded(
                      child: _ChoiceTile(
                        title: 'Portrait (9:16)',
                        subtitle: 'Reels / Stories',
                        icon: Icons.stay_current_portrait_rounded,
                        selected: _dimension == '9:16',
                        onTap: () => setState(() => _dimension = '9:16'),
                        c: c,
                      ),
                    ),
                    const SizedBox(width: AppSpacing.s12),
                    Expanded(
                      child: _ChoiceTile(
                        title: 'Landscape (16:9)',
                        subtitle: 'Cinema / TV',
                        icon: Icons.stay_current_landscape_rounded,
                        selected: _dimension == '16:9',
                        onTap: () => setState(() => _dimension = '16:9'),
                        c: c,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: AppSpacing.s20),

                // Music Mood Selector
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    'Soundtrack Mood',
                    style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: c.text),
                  ),
                ),
                const SizedBox(height: AppSpacing.s8),
                Row(
                  children: [
                    Expanded(
                      child: _ChoiceChip(
                        label: 'Calm Piano',
                        icon: Icons.spa_outlined,
                        selected: _mood == 'calm',
                        onTap: () => setState(() => _mood = 'calm'),
                        c: c,
                      ),
                    ),
                    const SizedBox(width: AppSpacing.s8),
                    Expanded(
                      child: _ChoiceChip(
                        label: 'Nostalgic',
                        icon: Icons.auto_awesome,
                        selected: _mood == 'nostalgic',
                        onTap: () => setState(() => _mood = 'nostalgic'),
                        c: c,
                      ),
                    ),
                    const SizedBox(width: AppSpacing.s8),
                    Expanded(
                      child: _ChoiceChip(
                        label: 'Energetic',
                        icon: Icons.bolt,
                        selected: _mood == 'energetic',
                        onTap: () => setState(() => _mood = 'energetic'),
                        c: c,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: AppSpacing.s24),

                if (uiState.clientError != null) ...[
                  Text(
                    'Error: ${uiState.clientError}',
                    style: TextStyle(color: c.error, fontSize: 12),
                  ),
                  const SizedBox(height: AppSpacing.s16),
                ],

                SizedBox(
                  width: double.infinity,
                  child: FilledButton.icon(
                    onPressed: _isEligible ? _startGeneration : null,
                    icon: const Icon(Icons.movie_creation_outlined),
                    label: const Text('Create AI Video'),
                    style: FilledButton.styleFrom(
                      backgroundColor: c.primary,
                      foregroundColor: c.primaryInverse,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
              ] else ...[
                // Active Job Live Progress / Terminal State View
                _buildActiveJobView(context, activeJob, c),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildActiveJobView(BuildContext context, VideoJobModel job, AppColors c) {
    final isFailed = job.isFailed;
    final isCompleted = job.isCompleted;
    final progressVal = job.overallProgress / 100.0;

    if (isCompleted) {
      return Column(
        children: [
          const SizedBox(height: AppSpacing.s16),
          Icon(Icons.check_circle_rounded, size: 56, color: c.success),
          const SizedBox(height: AppSpacing.s16),
          Text(
            'Memory Video Ready!',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: c.text),
          ),
          const SizedBox(height: AppSpacing.s8),
          Text(
            'Your multi-scene memory video has been rendered and saved.',
            textAlign: TextAlign.center,
            style: TextStyle(color: c.textMuted, fontSize: 13),
          ),
          const SizedBox(height: AppSpacing.s24),
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed: () => _watchCompletedVideo(job),
              icon: const Icon(Icons.play_arrow_rounded),
              label: const Text('Watch Memory Video'),
              style: FilledButton.styleFrom(
                backgroundColor: c.primary,
                foregroundColor: c.primaryInverse,
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
            ),
          ),
          const SizedBox(height: AppSpacing.s8),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              ref.read(videoJobControllerProvider.notifier).dismiss();
            },
            child: const Text('Done'),
          ),
        ],
      );
    }

    if (isFailed) {
      return Column(
        children: [
          const SizedBox(height: AppSpacing.s16),
          Icon(Icons.error_outline_rounded, size: 56, color: c.error),
          const SizedBox(height: AppSpacing.s16),
          Text(
            'Creation Paused',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: c.text),
          ),
          const SizedBox(height: AppSpacing.s8),
          Container(
            padding: const EdgeInsets.all(AppSpacing.s12),
            decoration: BoxDecoration(
              color: c.error.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(AppRadii.md),
              border: Border.all(color: c.error.withValues(alpha: 0.3)),
            ),
            child: Text(
              job.errorMessage ?? 'An error occurred during video creation.',
              textAlign: TextAlign.center,
              style: TextStyle(color: c.error, fontSize: 12),
            ),
          ),
          const SizedBox(height: AppSpacing.s24),
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed: () {
                ref.read(videoJobControllerProvider.notifier).retry();
              },
              icon: const Icon(Icons.refresh_rounded),
              label: const Text('Resume / Retry Creation'),
              style: FilledButton.styleFrom(
                backgroundColor: c.primary,
                foregroundColor: c.primaryInverse,
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
            ),
          ),
          const SizedBox(height: AppSpacing.s8),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              ref.read(videoJobControllerProvider.notifier).dismiss();
            },
            child: const Text('Dismiss'),
          ),
        ],
      );
    }

    // Active Processing State
    return Column(
      children: [
        const SizedBox(height: AppSpacing.s16),
        Stack(
          alignment: Alignment.center,
          children: [
            SizedBox(
              width: 76,
              height: 76,
              child: CircularProgressIndicator(
                value: progressVal > 0 ? progressVal : null,
                strokeWidth: 4,
                color: c.primary,
                backgroundColor: c.border,
              ),
            ),
            Text(
              '${job.overallProgress}%',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 16,
                color: c.text,
              ),
            ),
          ],
        ),
        const SizedBox(height: AppSpacing.s24),

        // Stage Title & Indicator
        if (job.stageIndex > 0)
          Text(
            'Stage ${job.stageIndex} of ${job.totalStages}',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: c.primary,
              letterSpacing: 0.5,
            ),
          ),
        const SizedBox(height: AppSpacing.s4),
        Text(
          job.currentTask,
          textAlign: TextAlign.center,
          style: TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 14,
            color: c.text,
          ),
        ),

        const SizedBox(height: AppSpacing.s12),
        // Progress Details: ETA & Upload Speed
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.schedule_rounded, size: 14, color: c.textMuted),
            const SizedBox(width: 4),
            Text(
              job.formattedEta,
              style: TextStyle(fontSize: 12, color: c.textMuted),
            ),
            if (job.formattedUploadSpeed != null) ...[
              const SizedBox(width: AppSpacing.s16),
              Icon(Icons.cloud_upload_outlined, size: 14, color: c.primary),
              const SizedBox(width: 4),
              Text(
                job.formattedUploadSpeed!,
                style: TextStyle(fontSize: 12, color: c.primary, fontWeight: FontWeight.w500),
              ),
            ],
          ],
        ),

        const SizedBox(height: AppSpacing.s24),

        // Non-blocking Action Buttons
        SizedBox(
          width: double.infinity,
          child: FilledButton(
            onPressed: () => Navigator.pop(context),
            style: FilledButton.styleFrom(
              backgroundColor: c.surfaceElevated,
              foregroundColor: c.text,
              padding: const EdgeInsets.symmetric(vertical: 14),
            ),
            child: const Text('Hide in background'),
          ),
        ),
        const SizedBox(height: AppSpacing.s8),
        TextButton(
          onPressed: () {
            ref.read(videoJobControllerProvider.notifier).cancel();
          },
          style: TextButton.styleFrom(foregroundColor: c.error),
          child: const Text('Cancel Video Creation'),
        ),
      ],
    );
  }
}

class _ChoiceTile extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;
  final bool selected;
  final VoidCallback onTap;
  final AppColors c;

  const _ChoiceTile({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.selected,
    required this.onTap,
    required this.c,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(AppRadii.md),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 12),
        decoration: BoxDecoration(
          color: selected ? c.primary.withValues(alpha: 0.12) : c.bg,
          borderRadius: BorderRadius.circular(AppRadii.md),
          border: Border.all(
            color: selected ? c.primary : c.border,
            width: selected ? 1.8 : 1.0,
          ),
        ),
        child: Column(
          children: [
            Icon(icon, color: selected ? c.primary : c.textMuted, size: 24),
            const SizedBox(height: 6),
            Text(
              title,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: selected ? c.primary : c.text,
              ),
            ),
            Text(
              subtitle,
              style: TextStyle(fontSize: 10, color: c.textMuted),
            ),
          ],
        ),
      ),
    );
  }
}

class _ChoiceChip extends StatelessWidget {
  final String label;
  final IconData icon;
  final bool selected;
  final VoidCallback onTap;
  final AppColors c;

  const _ChoiceChip({
    required this.label,
    required this.icon,
    required this.selected,
    required this.onTap,
    required this.c,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(AppRadii.md),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 8),
        decoration: BoxDecoration(
          color: selected ? c.primary.withValues(alpha: 0.12) : c.bg,
          borderRadius: BorderRadius.circular(AppRadii.md),
          border: Border.all(
            color: selected ? c.primary : c.border,
            width: selected ? 1.8 : 1.0,
          ),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: selected ? c.primary : c.textMuted, size: 16),
            const SizedBox(width: 4),
            Flexible(
              child: Text(
                label,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: selected ? c.primary : c.text,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _StatCount extends StatelessWidget {
  final int count;
  final String label;
  final IconData icon;
  final AppColors c;

  const _StatCount({
    required this.count,
    required this.label,
    required this.icon,
    required this.c,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Icon(icon, color: c.primary, size: 28),
        const SizedBox(height: 4),
        Text(
          '$count $label',
          style: TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 16,
            color: c.text,
          ),
        ),
      ],
    );
  }
}
