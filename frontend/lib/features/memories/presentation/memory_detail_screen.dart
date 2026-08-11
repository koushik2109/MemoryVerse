import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/core/widgets/states.dart';
import 'package:memory_verse/features/memories/presentation/multi_media_picker_screen.dart';
import 'package:memory_verse/features/memories/presentation/video_creator_sheet.dart';
import 'package:memory_verse/features/memories/presentation/memory_media_viewer.dart';
import 'package:memory_verse/features/memories/presentation/edit_memory_screen.dart';
import 'package:share_plus/share_plus.dart';

class MemoryDetailScreen extends ConsumerStatefulWidget {
  const MemoryDetailScreen({super.key, required this.memory});
  final MemoryModel memory;

  static void open(BuildContext context, MemoryModel memory) {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => MemoryDetailScreen(memory: memory)),
    );
  }

  @override
  ConsumerState<MemoryDetailScreen> createState() => _MemoryDetailScreenState();
}

class _MemoryDetailScreenState extends ConsumerState<MemoryDetailScreen> {


  void _shareMemory() {
    Share.share('Check out my memory: ${widget.memory.title}');
  }

  Future<void> _deleteMemory() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: context.colors.surface,
        title: Text('Delete Memory?', style: TextStyle(color: context.colors.text)),
        content: Text('This memory and all its media will be permanently deleted.', style: TextStyle(color: context.colors.textMuted)),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: Text('Cancel', style: TextStyle(color: context.colors.textMuted))),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text('Delete', style: TextStyle(color: context.colors.error)),
          ),
        ],
      ),
    );

    if (confirmed == true && mounted) {
      try {
        await ref.read(memoryRepositoryProvider).deleteMemory(widget.memory.id);
        ref.invalidate(memoriesListProvider);
        if (mounted) Navigator.pop(context);
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Failed to delete memory')));
        }
      }
    }
  }

  MediaModel? _getCoverMedia(MemoryModel memory) {
    if (memory.media.isEmpty) return null;
    if (memory.coverMediaId != null) {
      try {
        return memory.media.firstWhere((m) => m.id == memory.coverMediaId);
      } catch (_) {}
    }
    return memory.media.first;
  }

  Widget _buildContextualBanner(MemoryModel memory, AppColors c) {
    final total = memory.media.length;
    final photos = memory.media.where((m) => m.mediaType == 'image').length;
    final videos = memory.media.where((m) => m.mediaType == 'video').length;

    if (total == 0) {
      return Container(
        margin: const EdgeInsets.symmetric(horizontal: AppSpacing.s24, vertical: AppSpacing.s16),
        padding: const EdgeInsets.all(AppSpacing.s24),
        decoration: BoxDecoration(
          color: c.surfaceElevated,
          borderRadius: BorderRadius.circular(AppRadii.lg),
          border: Border.all(color: c.border.withValues(alpha: 0.5)),
        ),
        child: Column(
          children: [
            Icon(Icons.photo_library_outlined, size: 48, color: c.primary.withValues(alpha: 0.5)),
            const SizedBox(height: AppSpacing.s16),
            Text('Your story starts here', style: TextStyle(color: c.text, fontSize: 18, fontWeight: FontWeight.w600)),
            const SizedBox(height: AppSpacing.s8),
            Text('This memory is waiting for its first moment.', style: TextStyle(color: c.textMuted, fontSize: 14)),
            const SizedBox(height: AppSpacing.s16),
            FilledButton.icon(
              icon: const Icon(Icons.add_photo_alternate_outlined),
              label: const Text('Add Photos & Videos'),
              onPressed: () => MultiMediaPickerScreen.open(context, memoryId: memory.id),
              style: FilledButton.styleFrom(
                backgroundColor: c.primary,
                foregroundColor: c.primaryInverse,
              ),
            ),
          ],
        ),
      );
    }

    if (total == 1 && videos == 1) {
      return Container(
        margin: const EdgeInsets.symmetric(horizontal: AppSpacing.s24, vertical: AppSpacing.s16),
        padding: const EdgeInsets.all(AppSpacing.s16),
        decoration: BoxDecoration(
          color: c.surfaceElevated,
          borderRadius: BorderRadius.circular(AppRadii.lg),
          border: Border.all(color: c.border.withValues(alpha: 0.5)),
        ),
        child: Row(
          children: [
            Icon(Icons.info_outline, color: c.textMuted),
            const SizedBox(width: AppSpacing.s12),
            Expanded(
              child: Text(
                'Add more photos or video clips to stitch a Memory Video.',
                style: TextStyle(color: c.textMuted, fontSize: 13, height: 1.4),
              ),
            ),
            const SizedBox(width: AppSpacing.s12),
            TextButton(
              onPressed: () => MultiMediaPickerScreen.open(context, memoryId: memory.id),
              child: const Text('Add Media'),
            ),
          ],
        ),
      );
    }

    // Eligible for video creation
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: AppSpacing.s24, vertical: AppSpacing.s16),
      padding: const EdgeInsets.all(AppSpacing.s20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [c.primary.withValues(alpha: 0.1), c.primary.withValues(alpha: 0.02)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(AppRadii.lg),
        border: Border.all(color: c.primary.withValues(alpha: 0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.auto_awesome, color: c.primary, size: 20),
              const SizedBox(width: AppSpacing.s8),
              Text('$total Moments', style: TextStyle(color: c.text, fontSize: 16, fontWeight: FontWeight.w600)),
            ],
          ),
          const SizedBox(height: AppSpacing.s4),
          Text(
            '${photos > 0 ? '$photos photos' : ''}${photos > 0 && videos > 0 ? ' · ' : ''}${videos > 0 ? '$videos videos' : ''}',
            style: TextStyle(color: c.textMuted, fontSize: 13),
          ),
          const SizedBox(height: AppSpacing.s16),
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              icon: const Icon(Icons.movie_creation_outlined),
              label: const Text('Create Memory Video'),
              onPressed: () => VideoCreatorSheet.show(context, memory),
              style: FilledButton.styleFrom(
                backgroundColor: c.primary,
                foregroundColor: c.primaryInverse,
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final memoryAsync = ref.watch(memoryDetailProvider(widget.memory.id));

    return Scaffold(
      backgroundColor: c.bg,
      body: memoryAsync.when(
        loading: () => const LoadingState(),
        error: (err, _) => ErrorState(message: 'Failed to load memory', onRetry: () => ref.invalidate(memoryDetailProvider(widget.memory.id))),
        data: (memory) {
          final coverMedia = _getCoverMedia(memory);

          return CustomScrollView(
            physics: const BouncingScrollPhysics(parent: AlwaysScrollableScrollPhysics()),
            slivers: [
              // Hero Header
              SliverAppBar(
                expandedHeight: 350.0,
                pinned: true,
                backgroundColor: c.bg,
                elevation: 0,
                leading: IconButton(
                  icon: const Icon(Icons.arrow_back_rounded, color: Colors.white, shadows: [Shadow(color: Colors.black45, blurRadius: 4)]),
                  onPressed: () => Navigator.pop(context),
                ),
                actions: [
                  PopupMenuButton<String>(
                    icon: const Icon(Icons.more_horiz, color: Colors.white, shadows: [Shadow(color: Colors.black45, blurRadius: 4)]),
                    color: c.surfaceElevated,
                    onSelected: (val) {
                      if (val == 'delete') _deleteMemory();
                    },
                    itemBuilder: (_) => [
                      PopupMenuItem(value: 'delete', child: Text('Delete Memory', style: TextStyle(color: c.error))),
                    ],
                  ),
                ],
                flexibleSpace: FlexibleSpaceBar(
                  background: Stack(
                    fit: StackFit.expand,
                    children: [
                      // Cover Image or Gradient
                      if (coverMedia != null)
                        Image.network(
                          coverMedia.url,
                          fit: BoxFit.cover,
                          errorBuilder: (_, __, ___) => Container(color: c.surfaceElevated),
                        )
                      else
                        Container(
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [c.primary.withValues(alpha: 0.2), c.bg],
                              begin: Alignment.topCenter,
                              end: Alignment.bottomCenter,
                            ),
                          ),
                        ),
                      // Gradient Overlay for Text Readability
                      Container(
                        decoration: const BoxDecoration(
                          gradient: LinearGradient(
                            colors: [Colors.transparent, Colors.black87],
                            begin: Alignment.topCenter,
                            end: Alignment.bottomCenter,
                            stops: [0.4, 1.0],
                          ),
                        ),
                      ),
                      // Hero Content
                      Positioned(
                        left: AppSpacing.s24,
                        right: AppSpacing.s24,
                        bottom: AppSpacing.s24,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              memory.title,
                              style: const TextStyle(fontFamily: 'Inter', fontSize: 36, fontWeight: FontWeight.w700, letterSpacing: -1.0, color: Colors.white, height: 1.1),
                            ),
                            const SizedBox(height: AppSpacing.s8),
                            Row(
                              children: [
                                const Icon(Icons.calendar_today_rounded, size: 14, color: Colors.white70),
                                const SizedBox(width: AppSpacing.s6),
                                Text(
                                  DateFormat('MMMM d, yyyy').format(memory.memoryDate),
                                  style: const TextStyle(fontFamily: 'Inter', fontSize: 13, color: Colors.white70, fontWeight: FontWeight.w500),
                                ),
                                if (memory.locationName != null) ...[
                                  const SizedBox(width: AppSpacing.s12),
                                  const Icon(Icons.location_on, size: 14, color: Colors.white70),
                                  const SizedBox(width: AppSpacing.s4),
                                  Text(
                                    memory.locationName!,
                                    style: const TextStyle(fontFamily: 'Inter', fontSize: 13, color: Colors.white70, fontWeight: FontWeight.w500),
                                  ),
                                ],
                              ],
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              // Action Bar
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24, vertical: AppSpacing.s16),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      _ActionButton(
                        icon: Icons.add_photo_alternate_outlined,
                        label: 'Add Media',
                        c: c,
                        onTap: () => MultiMediaPickerScreen.open(context, memoryId: memory.id),
                      ),
                      _ActionButton(
                        icon: Icons.ios_share,
                        label: 'Share',
                        c: c,
                        onTap: _shareMemory,
                      ),
                      _ActionButton(
                        icon: Icons.edit_outlined,
                        label: 'Edit',
                        c: c,
                        onTap: () => EditMemoryScreen.open(context, memory),
                      ),
                    ],
                  ),
                ),
              ),

              // Description
              if (memory.description != null && memory.description!.isNotEmpty)
                SliverToBoxAdapter(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24, vertical: AppSpacing.s8),
                    child: Text(
                      memory.description!,
                      style: TextStyle(fontFamily: 'Inter', fontSize: 15, color: c.text, height: 1.5),
                    ),
                  ),
                ),

              // Contextual Banner
              SliverToBoxAdapter(
                child: _buildContextualBanner(memory, c),
              ),

              // Gallery Title
              if (memory.media.isNotEmpty)
                SliverToBoxAdapter(
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(AppSpacing.s24, AppSpacing.s16, AppSpacing.s24, AppSpacing.s16),
                    child: Text('Gallery', style: TextStyle(color: c.text, fontSize: 20, fontWeight: FontWeight.w700)),
                  ),
                ),

              // Gallery Grid
              if (memory.media.isNotEmpty)
                SliverPadding(
                  padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s16),
                  sliver: SliverGrid(
                    gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 3,
                      crossAxisSpacing: 8,
                      mainAxisSpacing: 8,
                      childAspectRatio: 0.8, // Slightly taller than wide for a modern look
                    ),
                    delegate: SliverChildBuilderDelegate(
                      (context, index) {
                        final mediaItem = memory.media[index];
                        return GestureDetector(
                          onTap: () {
                            MemoryMediaViewer.open(context, memory, index);
                          },
                          child: Container(
                            decoration: BoxDecoration(
                              color: c.surfaceElevated,
                              borderRadius: BorderRadius.circular(AppRadii.sm),
                            ),
                            clipBehavior: Clip.antiAlias,
                            child: Stack(
                              fit: StackFit.expand,
                              children: [
                                Image.network(
                                  mediaItem.thumbnailUrl ?? mediaItem.url,
                                  fit: BoxFit.cover,
                                  errorBuilder: (_, __, ___) => Center(child: Icon(Icons.image_outlined, color: c.textMuted)),
                                ),
                                if (mediaItem.isVideo)
                                  Positioned(
                                    top: 8,
                                    right: 8,
                                    child: Container(
                                      padding: const EdgeInsets.all(4),
                                      decoration: BoxDecoration(
                                        color: Colors.black.withValues(alpha: 0.6),
                                        shape: BoxShape.circle,
                                      ),
                                      child: const Icon(Icons.play_arrow_rounded, color: Colors.white, size: 16),
                                    ),
                                  ),
                              ],
                            ),
                          ),
                        );
                      },
                      childCount: memory.media.length,
                    ),
                  ),
                ),
                
              if (memory.media.isEmpty)
                const SliverPadding(
                  padding: EdgeInsets.symmetric(horizontal: AppSpacing.s24),
                  sliver: SliverToBoxAdapter(
                    child: Padding(
                      padding: EdgeInsets.symmetric(vertical: AppSpacing.s32),
                      child: EmptyState(
                        imageAsset: 'assets/images/empty_memory.png',
                        title: 'No media yet.',
                        subtitle: 'Upload photos or videos to enrich this memory.',
                      ),
                    ),
                  ),
                ),
                
              const SliverToBoxAdapter(child: SizedBox(height: 100)),
            ],
          );
        },
      ),
    );
  }
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final AppColors c;
  final VoidCallback onTap;

  const _ActionButton({required this.icon, required this.label, required this.c, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: c.surfaceElevated,
              shape: BoxShape.circle,
              border: Border.all(color: c.border.withValues(alpha: 0.5)),
            ),
            child: Icon(icon, color: c.text, size: 24),
          ),
          const SizedBox(height: AppSpacing.s8),
          Text(label, style: TextStyle(color: c.text, fontSize: 12, fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }
}
