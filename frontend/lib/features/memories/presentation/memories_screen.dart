import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/widgets/media.dart';
import 'package:memory_verse/core/widgets/states.dart';
import 'package:memory_verse/features/memories/presentation/create_memory_sheet.dart';
import 'package:memory_verse/features/memories/presentation/memory_detail_screen.dart';
import 'package:memory_verse/features/memories/presentation/widgets/movie_card.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;

class MemoriesScreen extends ConsumerStatefulWidget {
  const MemoriesScreen({super.key});

  @override
  ConsumerState<MemoriesScreen> createState() => _MemoriesScreenState();
}

class _MemoriesScreenState extends ConsumerState<MemoriesScreen> {
  int _selectedTabIndex = 0; // 0: Memories, 1: Movies

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final memoriesAsync = ref.watch(memoriesListProvider);

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Header ──────────────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppSpacing.s20,
                AppSpacing.s12,
                AppSpacing.s20,
                0,
              ),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      'Memories',
                      style: Theme.of(context).textTheme.displaySmall
                          ?.copyWith(color: c.text),
                    ),
                  ),
                  IconButton(
                    icon: Icon(
                      Icons.add_circle_outline_rounded,
                      color: c.primary,
                    ),
                    onPressed: () => CreateMemorySheet.show(context),
                  ),
                ],
              ),
            ),

            // Month label
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppSpacing.s20,
                AppSpacing.s4,
                AppSpacing.s20,
                AppSpacing.s8,
              ),
              child: Text(
                DateFormat('MMMM yyyy').format(DateTime.now()),
                style: Theme.of(context).textTheme.bodySmall
                    ?.copyWith(color: c.textMuted),
              ),
            ),

            // ── Filter Segment (Memories vs Movies) ───────
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s20, vertical: AppSpacing.s8),
              child: memoriesAsync.maybeWhen(
                data: (memories) {
                  int movieCount = 0;
                  for (final m in memories) {
                    movieCount += m.media.where((x) => x.isGeneratedMovie).length;
                  }

                  return Row(
                    children: [
                      _SegmentTab(
                        label: 'All Memories (${memories.length})',
                        icon: Icons.photo_library_outlined,
                        isSelected: _selectedTabIndex == 0,
                        onTap: () => setState(() => _selectedTabIndex = 0),
                      ),
                      const SizedBox(width: 8),
                      _SegmentTab(
                        label: 'Movies ($movieCount)',
                        icon: Icons.movie_creation_outlined,
                        isSelected: _selectedTabIndex == 1,
                        onTap: () => setState(() => _selectedTabIndex = 1),
                      ),
                    ],
                  );
                },
                orElse: () => const SizedBox.shrink(),
              ),
            ),

            const SizedBox(height: AppSpacing.s8),

            // ── Content (Memories or Movies) ─────────────
            Expanded(
              child: memoriesAsync.when(
                loading: () => const LoadingState(),
                error: (e, _) => ErrorState(
                  message: e.toString(),
                  onRetry: () => ref.invalidate(memoriesListProvider),
                ),
                data: (memories) {
                  if (memories.isEmpty) {
                    return EmptyState(
                      icon: Icons.photo_library_outlined,
                      title: 'No memories yet',
                      subtitle: 'Capture your first moment and let MemoryVerse bring it to life.',
                      buttonText: 'Create Memory',
                      onTap: () => CreateMemorySheet.show(context),
                    );
                  }

                  // ── Movies Tab Content ───────────────────
                  if (_selectedTabIndex == 1) {
                    final allMovies = <(MemoryModel, MediaModel)>[];
                    for (final m in memories) {
                      for (final media in m.media) {
                        if (media.isGeneratedMovie) {
                          allMovies.add((m, media));
                        }
                      }
                    }

                    if (allMovies.isEmpty) {
                      return const EmptyState(
                        icon: Icons.movie_filter_rounded,
                        title: 'No movies yet',
                        subtitle: 'Tap into any memory and select "Create with AI" to generate a cinematic movie.',
                      );
                    }

                    return RefreshIndicator(
                      color: c.primary,
                      onRefresh: () async => ref.invalidate(memoriesListProvider),
                      child: ListView.builder(
                        padding: const EdgeInsets.symmetric(
                          horizontal: AppSpacing.s20,
                          vertical: AppSpacing.s8,
                        ),
                        itemCount: allMovies.length,
                        itemBuilder: (context, idx) {
                          final item = allMovies[idx];
                          return MovieCard(
                            movie: item.$2,
                            memoryTitle: item.$1.title,
                          );
                        },
                      ),
                    );
                  }

                  // ── Memories Tab Content ─────────────────
                  // Group by date
                  final today = DateTime.now();
                  final groups = <String, List<MemoryModel>>{};
                  for (final m in memories) {
                    final diff = today.difference(m.memoryDate).inDays;
                    final label = diff == 0
                        ? 'Today'
                        : diff == 1
                        ? 'Yesterday'
                        : DateFormat('MMMM yyyy').format(m.memoryDate);
                    groups.putIfAbsent(label, () => []).add(m);
                  }

                  return RefreshIndicator(
                    color: c.primary,
                    onRefresh: () async => ref.invalidate(memoriesListProvider),
                    child: ListView.builder(
                      padding: const EdgeInsets.symmetric(
                        horizontal: AppSpacing.s20,
                      ),
                      itemCount: groups.length,
                      itemBuilder: (_, gi) {
                        final label = groups.keys.elementAt(gi);
                        final items = groups[label]!;
                        return Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            if (gi > 0) const SizedBox(height: AppSpacing.s24),
                            Text(
                              label,
                              style: Theme.of(context).textTheme.titleSmall
                                  ?.copyWith(
                                    color: c.textMuted,
                                    fontWeight: FontWeight.w600,
                                  ),
                            ),
                            const SizedBox(height: AppSpacing.s12),
                            GridView.builder(
                              shrinkWrap: true,
                              physics: const NeverScrollableScrollPhysics(),
                              gridDelegate:
                                  const SliverGridDelegateWithFixedCrossAxisCount(
                                    crossAxisCount: 2,
                                    crossAxisSpacing: AppSpacing.s12,
                                    mainAxisSpacing: AppSpacing.s12,
                                    childAspectRatio: 0.8,
                                  ),
                              itemCount: items.length,
                              itemBuilder: (_, i) =>
                                  _MemoryGridItem(memory: items[i]),
                            ),
                          ],
                        );
                      },
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SegmentTab extends StatelessWidget {
  final String label;
  final IconData icon;
  final bool isSelected;
  final VoidCallback onTap;

  const _SegmentTab({
    required this.label,
    required this.icon,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: isSelected
              ? c.primary.withValues(alpha: 0.2)
              : c.surfaceElevated.withValues(alpha: 0.4),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: isSelected
                ? c.primary.withValues(alpha: 0.5)
                : adt.AppColors.onDarkPrimary.withValues(alpha: 0.1),
            width: 1,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 16,
              color: isSelected ? c.primary : c.textMuted,
            ),
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                fontSize: 13,
                fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
                color: isSelected ? adt.AppColors.onDarkPrimary : c.textMuted,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _MemoryGridItem extends StatelessWidget {
  const _MemoryGridItem({required this.memory});
  final MemoryModel memory;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;

    MediaModel? coverMedia;
    if (memory.coverMediaId != null) {
      coverMedia = memory.media.where((m) => m.id == memory.coverMediaId).firstOrNull;
    }
    // "use the coverpage with the intro if not given else give it the picture the person has given , dont make any picture default choosen"
    final String? imageUrl = coverMedia?.thumbnailUrl ?? coverMedia?.url;

    return GestureDetector(
      onTap: () => MemoryDetailScreen.open(context, memory),
      child: Container(
        decoration: BoxDecoration(
          color: c.surfaceElevated,
          borderRadius: BorderRadius.circular(AppRadii.xl),
        ),
        clipBehavior: Clip.antiAlias,
        child: Stack(
          fit: StackFit.expand,
          children: [
            if (imageUrl != null)
              AppNetworkImage(
                imageUrl: imageUrl,
                fit: BoxFit.cover,
                memCacheWidth: 400,
              )
            else
              Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      Color(0xFF220F30),
                      Color(0xFF361448),
                      Color(0xFF190924),
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.auto_awesome_rounded,
                        color: c.primary.withValues(alpha: 0.7),
                        size: 30,
                      ),
                      const SizedBox(height: 6),
                      Text(
                        '✦ Intro ✦',
                        style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.w600,
                          color: c.primary.withValues(alpha: 0.8),
                          letterSpacing: 1.0,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.transparent,
                    adt.AppColors.plum900.withValues(alpha: 0.7),
                  ],
                  stops: const [0.5, 1.0],
                ),
              ),
            ),
            Positioned(
              bottom: AppSpacing.s12,
              left: AppSpacing.s12,
              right: AppSpacing.s12,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    memory.title,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: adt.AppColors.onDarkPrimary,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    '${memory.media.length} items',
                    style: TextStyle(
                      fontSize: 11,
                      color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.8),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
