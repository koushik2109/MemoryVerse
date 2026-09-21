import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
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
  String? _jobId;
  String _status = 'ready'; // ready, queued, processing, completed, failed
  String? _errorMsg;
  Timer? _timer;

  String _dimension = '9:16'; // '9:16' or '16:9'
  String _mood = 'calm'; // 'calm', 'nostalgic', 'energetic'
  late Set<String> _selectedMediaIds;

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
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _startGeneration() async {
    if (!_isEligible) return;

    setState(() {
      _status = 'queued';
      _errorMsg = null;
    });

    try {
      final repo = ref.read(mediaRepositoryProvider);
      final jobId = await repo.generateVideo(
        widget.memory.id,
        dimension: _dimension,
        mood: _mood,
        mediaIds: _selectedMediaIds.toList(),
      );

      setState(() {
        _jobId = jobId;
      });

      _startPolling();
    } catch (e) {
      setState(() {
        _status = 'failed';
        _errorMsg = e.toString();
      });
    }
  }

  void _startPolling() {
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(seconds: 2), (timer) async {
      if (_jobId == null) return;
      try {
        final repo = ref.read(mediaRepositoryProvider);
        final job = await repo.getJobStatus(_jobId!);

        if (!mounted) return;

        setState(() {
          _status = job.status;
          if (_status == 'failed') {
            _errorMsg = job.errorMessage ?? 'Unknown error occurred.';
            timer.cancel();
          }
        });

        if (_status == 'completed') {
          timer.cancel();
          ref.invalidate(memoryDetailProvider(widget.memory.id));
        }
      } catch (e) {
        // ignore occasional network errors
      }
    });
  }

  void _watchCompletedVideo() {
    Navigator.pop(context);
    final mediaList = widget.memory.media;
    MediaModel? targetMedia;
    // Look for video media item
    final videos = mediaList.where((m) => m.mediaType == 'video').toList();
    if (videos.isNotEmpty) {
      targetMedia = videos.last;
    } else if (mediaList.isNotEmpty) {
      targetMedia = mediaList.first;
    }

    if (targetMedia != null) {
      VideoPlayerScreen.open(context, targetMedia);
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Video creation complete! Refreshing memory...')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;

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

              if (!_isEligible) ...[
                Container(
                  padding: const EdgeInsets.all(AppSpacing.s16),
                  decoration: BoxDecoration(
                    color: c.surfaceElevated,
                    borderRadius: BorderRadius.circular(AppRadii.lg),
                  ),
                  child: Row(
                    children: [
                      Icon(Icons.info_outline, color: c.textMuted),
                      const SizedBox(width: AppSpacing.s12),
                      Expanded(
                        child: Text(
                          _ineligibilityReason,
                          style: TextStyle(color: c.text, fontSize: 13),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: AppSpacing.s24),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton(
                    onPressed: () => Navigator.pop(context),
                    style: FilledButton.styleFrom(
                      backgroundColor: c.surfaceElevated,
                      foregroundColor: c.text,
                    ),
                    child: const Text('Close'),
                  ),
                ),
              ] else if (_status == 'completed') ...[
                // Completed State with Direct Playback
                Container(
                  padding: const EdgeInsets.all(AppSpacing.s20),
                  decoration: BoxDecoration(
                    color: c.surfaceElevated,
                    borderRadius: BorderRadius.circular(AppRadii.lg),
                    border: Border.all(color: c.primary.withValues(alpha: 0.3)),
                  ),
                  child: Column(
                    children: [
                      Icon(Icons.check_circle_outline, color: c.primary, size: 48),
                      const SizedBox(height: AppSpacing.s12),
                      Text(
                        'Your AI Video is Ready!',
                        style: TextStyle(
                          fontWeight: FontWeight.w700,
                          fontSize: 18,
                          color: c.text,
                        ),
                      ),
                      const SizedBox(height: AppSpacing.s6),
                      Text(
                        'Created with ${_selectedMediaIds.length} pictures, Ken Burns motion, and $_mood soundtrack.',
                        textAlign: TextAlign.center,
                        style: TextStyle(color: c.textMuted, fontSize: 13),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: AppSpacing.s24),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton.icon(
                    onPressed: _watchCompletedVideo,
                    icon: const Icon(Icons.play_arrow_rounded, size: 26),
                    label: const Text('Watch Video Now'),
                    style: FilledButton.styleFrom(
                      backgroundColor: c.primary,
                      foregroundColor: c.primaryInverse,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
              ] else if (_status == 'ready' || _status == 'failed') ...[
                // Media Stats Card
                Container(
                  padding: const EdgeInsets.all(AppSpacing.s16),
                  decoration: BoxDecoration(
                    color: c.surfaceElevated,
                    borderRadius: BorderRadius.circular(AppRadii.lg),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            'Source Media',
                            style: TextStyle(
                              fontWeight: FontWeight.w600,
                              color: c.text,
                            ),
                          ),
                          Text(
                            '${_selectedMediaIds.length} selected',
                            style: TextStyle(color: c.primary, fontSize: 12, fontWeight: FontWeight.w600),
                          ),
                        ],
                      ),
                      const SizedBox(height: AppSpacing.s12),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceAround,
                        children: [
                          _StatCount(
                            count: _photoCount,
                            label: 'Photos',
                            icon: Icons.photo_outlined,
                            c: c,
                          ),
                          _StatCount(
                            count: _videoCount,
                            label: 'Videos',
                            icon: Icons.videocam_outlined,
                            c: c,
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: AppSpacing.s16),

                // Aspect Ratio Selector
                Container(
                  padding: const EdgeInsets.all(AppSpacing.s16),
                  decoration: BoxDecoration(
                    color: c.surfaceElevated,
                    borderRadius: BorderRadius.circular(AppRadii.lg),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Aspect Ratio',
                        style: TextStyle(fontWeight: FontWeight.w600, color: c.text, fontSize: 13),
                      ),
                      const SizedBox(height: AppSpacing.s10),
                      Row(
                        children: [
                          Expanded(
                            child: _ChoiceTile(
                              title: '9:16 Portrait',
                              subtitle: 'Reels & Shorts',
                              icon: Icons.stay_current_portrait_rounded,
                              selected: _dimension == '9:16',
                              onTap: () => setState(() => _dimension = '9:16'),
                              c: c,
                            ),
                          ),
                          const SizedBox(width: AppSpacing.s12),
                          Expanded(
                            child: _ChoiceTile(
                              title: '16:9 Cinema',
                              subtitle: 'Widescreen / TV',
                              icon: Icons.stay_current_landscape_rounded,
                              selected: _dimension == '16:9',
                              onTap: () => setState(() => _dimension = '16:9'),
                              c: c,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: AppSpacing.s16),

                // Soundtrack Mood Selector
                Container(
                  padding: const EdgeInsets.all(AppSpacing.s16),
                  decoration: BoxDecoration(
                    color: c.surfaceElevated,
                    borderRadius: BorderRadius.circular(AppRadii.lg),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Soundtrack Mood (4-Chord AI Pad)',
                        style: TextStyle(fontWeight: FontWeight.w600, color: c.text, fontSize: 13),
                      ),
                      const SizedBox(height: AppSpacing.s10),
                      Row(
                        children: [
                          Expanded(
                            child: _ChoiceChip(
                              label: 'Calm',
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
                              icon: Icons.favorite_border,
                              selected: _mood == 'nostalgic',
                              onTap: () => setState(() => _mood = 'nostalgic'),
                              c: c,
                            ),
                          ),
                          const SizedBox(width: AppSpacing.s8),
                          Expanded(
                            child: _ChoiceChip(
                              label: 'Energetic',
                              icon: Icons.bolt_outlined,
                              selected: _mood == 'energetic',
                              onTap: () => setState(() => _mood = 'energetic'),
                              c: c,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),

                if (_status == 'failed') ...[
                  const SizedBox(height: AppSpacing.s16),
                  Text(
                    'Failed: $_errorMsg',
                    style: TextStyle(color: c.error, fontSize: 12),
                  ),
                ],
                const SizedBox(height: AppSpacing.s24),

                SizedBox(
                  width: double.infinity,
                  child: FilledButton.icon(
                    onPressed: _startGeneration,
                    icon: const Icon(Icons.movie_creation_outlined),
                    label: Text(
                      _status == 'failed' ? 'Retry Creation' : 'Create AI Video',
                    ),
                    style: FilledButton.styleFrom(
                      backgroundColor: c.primary,
                      foregroundColor: c.primaryInverse,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
              ] else ...[
                // Processing state
                const SizedBox(height: AppSpacing.s16),
                CircularProgressIndicator(color: c.primary),
                const SizedBox(height: AppSpacing.s24),
                Text(
                  _status == 'queued'
                      ? 'AI Director preparing narrative arc...'
                      : _status == 'processing'
                          ? 'Composing Ken Burns motion & synthesized audio...'
                          : 'Finalizing Memory Reel...',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontWeight: FontWeight.w600, color: c.text),
                ),
                const SizedBox(height: AppSpacing.s8),
                Text(
                  'You can keep this open or hide in background; your video will appear in this memory.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: c.textMuted, fontSize: 12),
                ),
                const SizedBox(height: AppSpacing.s24),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton(
                    onPressed: () => Navigator.pop(context),
                    style: FilledButton.styleFrom(
                      backgroundColor: c.surfaceElevated,
                      foregroundColor: c.text,
                    ),
                    child: const Text('Hide in background'),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
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
