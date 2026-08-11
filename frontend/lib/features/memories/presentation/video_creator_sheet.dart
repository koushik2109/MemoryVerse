import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';

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

  int get _photoCount => widget.memory.media.where((m) => m.mediaType == 'image').length;
  int get _videoCount => widget.memory.media.where((m) => m.mediaType == 'video').length;
  int get _totalCount => widget.memory.media.length;

  bool get _isEligible {
    if (_totalCount == 0) return false;
    if (_photoCount == 1 && _videoCount == 0) return false;
    // 1 video only -> handled in UI (can export original, but not stitch)
    if (_videoCount == 1 && _photoCount == 0) return false;
    return true;
  }

  String get _ineligibilityReason {
    if (_totalCount == 0) return 'Add media to create a video.';
    if (_photoCount == 1 && _videoCount == 0) return 'Add more media to stitch a video.';
    if (_videoCount == 1 && _photoCount == 0) return 'Your memory has one video. Add more photos or video clips to create a stitched Memory Video.';
    return '';
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
      final jobId = await repo.generateVideo(widget.memory.id);
      
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
          // Wait a second then refresh and open video
          await Future.delayed(const Duration(seconds: 1));
          if (mounted) {
            ref.invalidate(memoryDetailProvider(widget.memory.id));
            Navigator.pop(context); // Close sheet
            
            // Navigate to video player if possible.
            // But we need the MediaModel. We can fetch it or just tell the user it's done.
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('Video creation complete!')),
            );
          }
        }
      } catch (e) {
        // ignore occasional network errors
      }
    });
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
        borderRadius: const BorderRadius.vertical(top: Radius.circular(AppRadii.xl)),
      ),
      child: SafeArea(
        top: false,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(width: 40, height: 4, decoration: BoxDecoration(color: c.border, borderRadius: BorderRadius.circular(2))),
            const SizedBox(height: AppSpacing.s24),
            Text('Create Memory Video', style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: AppSpacing.s8),
            Text('Automatically stitch your memory into a cinematic video.', textAlign: TextAlign.center, style: TextStyle(color: c.textMuted)),
            const SizedBox(height: AppSpacing.s32),

            if (!_isEligible) ...[
              Container(
                padding: const EdgeInsets.all(AppSpacing.s16),
                decoration: BoxDecoration(color: c.surfaceElevated, borderRadius: BorderRadius.circular(AppRadii.lg)),
                child: Row(
                  children: [
                    Icon(Icons.info_outline, color: c.textMuted),
                    const SizedBox(width: AppSpacing.s12),
                    Expanded(
                      child: Text(_ineligibilityReason, style: TextStyle(color: c.text, fontSize: 13)),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: AppSpacing.s24),
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: () => Navigator.pop(context),
                  style: FilledButton.styleFrom(backgroundColor: c.surfaceElevated, foregroundColor: c.text),
                  child: const Text('Close'),
                ),
              ),
            ] else if (_status == 'ready' || _status == 'failed') ...[
              Container(
                padding: const EdgeInsets.all(AppSpacing.s16),
                decoration: BoxDecoration(color: c.surfaceElevated, borderRadius: BorderRadius.circular(AppRadii.lg)),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Media included', style: TextStyle(fontWeight: FontWeight.w600, color: c.text)),
                    const SizedBox(height: AppSpacing.s8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _StatCount(count: _photoCount, label: 'Photos', icon: Icons.photo_outlined, c: c),
                        _StatCount(count: _videoCount, label: 'Videos', icon: Icons.videocam_outlined, c: c),
                      ],
                    ),
                  ],
                ),
              ),
              if (_status == 'failed') ...[
                const SizedBox(height: AppSpacing.s16),
                Text('Failed: $_errorMsg', style: TextStyle(color: c.error, fontSize: 12)),
              ],
              const SizedBox(height: AppSpacing.s32),
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  onPressed: _startGeneration,
                  icon: const Icon(Icons.movie_creation_outlined),
                  label: Text(_status == 'failed' ? 'Retry Creation' : 'Create Video'),
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
                _status == 'queued' ? 'Preparing...' : 
                _status == 'processing' ? 'Stitching video (this may take a minute)...' : 
                'Finalizing...',
                style: TextStyle(fontWeight: FontWeight.w600, color: c.text),
              ),
              const SizedBox(height: AppSpacing.s8),
              Text('You can close this sheet, the video will appear when ready.', textAlign: TextAlign.center, style: TextStyle(color: c.textMuted, fontSize: 12)),
              const SizedBox(height: AppSpacing.s24),
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: () => Navigator.pop(context),
                  style: FilledButton.styleFrom(backgroundColor: c.surfaceElevated, foregroundColor: c.text),
                  child: const Text('Hide in background'),
                ),
              ),
            ],
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

  const _StatCount({required this.count, required this.label, required this.icon, required this.c});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Icon(icon, color: c.primary, size: 28),
        const SizedBox(height: 4),
        Text('$count $label', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16, color: c.text)),
      ],
    );
  }
}
