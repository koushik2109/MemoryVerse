import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/core/widgets/states.dart';
import 'package:memory_verse/features/memories/presentation/memory_detail_screen.dart';
import 'package:memory_verse/features/memories/presentation/edit_memory_screen.dart';
import 'package:share_plus/share_plus.dart';

class TimelineScreen extends ConsumerStatefulWidget {
  const TimelineScreen({super.key});

  @override
  ConsumerState<TimelineScreen> createState() => _TimelineScreenState();
}

class _TimelineScreenState extends ConsumerState<TimelineScreen> {
  String? _selectedYear;
  String? _selectedMonth;
  String _searchQuery = '';

  void _shareMemory(MemoryModel memory) {
    Share.share('Check out my memory: ${memory.title}');
  }

  Future<void> _deleteMemory(MemoryModel memory) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: context.colors.surface,
        title: Text(
          'Delete Memory?',
          style: TextStyle(color: context.colors.text),
        ),
        content: Text(
          'This memory and all its media will be permanently deleted.',
          style: TextStyle(color: context.colors.textMuted),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text(
              'Cancel',
              style: TextStyle(color: context.colors.textMuted),
            ),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text(
              'Delete',
              style: TextStyle(color: context.colors.error),
            ),
          ),
        ],
      ),
    );

    if (confirmed == true && mounted) {
      try {
        await ref.read(memoryRepositoryProvider).deleteMemory(memory.id);
        ref.invalidate(timelineProvider);
        ref.invalidate(memoriesListProvider);
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Failed to delete memory')),
          );
        }
      }
    }
  }

  void _showMemoryOptions(BuildContext context, MemoryModel memory) {
    final c = context.colors;
    showModalBottomSheet(
      context: context,
      backgroundColor: c.surfaceElevated,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(AppRadii.xl)),
      ),
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(height: AppSpacing.s8),
            Container(
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: c.border,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: AppSpacing.s16),
            ListTile(
              leading: Icon(Icons.edit_outlined, color: c.text),
              title: Text('Edit Memory', style: TextStyle(color: c.text)),
              onTap: () {
                Navigator.pop(ctx);
                EditMemoryScreen.open(context, memory);
              },
            ),
            ListTile(
              leading: Icon(Icons.share_outlined, color: c.text),
              title: Text('Share', style: TextStyle(color: c.text)),
              onTap: () {
                Navigator.pop(ctx);
                _shareMemory(memory);
              },
            ),
            ListTile(
              leading: Icon(Icons.delete_outline, color: c.error),
              title: Text('Delete', style: TextStyle(color: c.error)),
              onTap: () {
                Navigator.pop(ctx);
                _deleteMemory(memory);
              },
            ),
            const SizedBox(height: AppSpacing.s16),
          ],
        ),
      ),
    );
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

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final timelineAsync = ref.watch(timelineProvider);

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: SafeArea(
        child: timelineAsync.when(
          loading: () => const LoadingState(),
          error: (e, _) => ErrorState(
            message: e.toString(),
            onRetry: () => ref.invalidate(timelineProvider),
          ),
          data: (timeline) {
            // Filter logic
            var filteredGroups = timeline.groups;

            if (_selectedYear != null) {
              filteredGroups = filteredGroups
                  .where((g) => g.year == _selectedYear)
                  .toList();
            }

            if (_selectedMonth != null) {
              filteredGroups = filteredGroups
                  .map((g) {
                    final filteredMonths = g.months
                        .where((m) => m.month == _selectedMonth)
                        .toList();
                    return TimelineYearGroup(
                      year: g.year,
                      months: filteredMonths,
                    );
                  })
                  .where((g) => g.months.isNotEmpty)
                  .toList();
            }

            if (_searchQuery.isNotEmpty) {
              final q = _searchQuery.toLowerCase();
              filteredGroups = filteredGroups
                  .map((g) {
                    final filteredMonths = g.months
                        .map((m) {
                          final filteredDays = m.days
                              .map((d) {
                                final filteredMemories = d.memories.where((
                                  mem,
                                ) {
                                  return mem.title.toLowerCase().contains(q) ||
                                      (mem.locationName?.toLowerCase().contains(
                                            q,
                                          ) ??
                                          false) ||
                                      (mem.description?.toLowerCase().contains(
                                            q,
                                          ) ??
                                          false);
                                }).toList();
                                return TimelineDayGroup(
                                  dateLabel: d.dateLabel,
                                  date: d.date,
                                  memories: filteredMemories,
                                );
                              })
                              .where((d) => d.memories.isNotEmpty)
                              .toList();
                          return TimelineMonthGroup(
                            month: m.month,
                            days: filteredDays,
                          );
                        })
                        .where((m) => m.days.isNotEmpty)
                        .toList();
                    return TimelineYearGroup(
                      year: g.year,
                      months: filteredMonths,
                    );
                  })
                  .where((g) => g.months.isNotEmpty)
                  .toList();
            }

            final hasData = timeline.groups.isNotEmpty;
            final hasFilteredData = filteredGroups.isNotEmpty;

            final availableYears = timeline.groups
                .map((g) => g.year)
                .toSet()
                .toList();
            final availableMonths = {
              'January',
              'February',
              'March',
              'April',
              'May',
              'June',
              'July',
              'August',
              'September',
              'October',
              'November',
              'December',
            };

            return CustomScrollView(
              physics: const BouncingScrollPhysics(
                parent: AlwaysScrollableScrollPhysics(),
              ),
              slivers: [
                // Header & Search
                SliverToBoxAdapter(
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(
                      AppSpacing.s24,
                      AppSpacing.s16,
                      AppSpacing.s24,
                      AppSpacing.s16,
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Timeline',
                          style: TextStyle(
                            fontSize: 32,
                            fontWeight: FontWeight.w700,
                            color: c.text,
                            letterSpacing: -1.0,
                          ),
                        ),
                        const SizedBox(height: AppSpacing.s16),
                        if (hasData)
                          Container(
                            decoration: BoxDecoration(
                              color: c.surfaceElevated,
                              borderRadius: BorderRadius.circular(AppRadii.lg),
                            ),
                            child: TextField(
                              style: TextStyle(color: c.text),
                              decoration: InputDecoration(
                                hintText: 'Search memories...',
                                hintStyle: TextStyle(color: c.textMuted),
                                prefixIcon: Icon(
                                  Icons.search,
                                  color: c.textMuted,
                                ),
                                border: InputBorder.none,
                                contentPadding: const EdgeInsets.symmetric(
                                  horizontal: 16,
                                  vertical: 14,
                                ),
                              ),
                              onChanged: (val) =>
                                  setState(() => _searchQuery = val),
                            ),
                          ),
                      ],
                    ),
                  ),
                ),

                // Filters (Horizontal Scroll)
                if (hasData)
                  SliverToBoxAdapter(
                    child: SizedBox(
                      height: 40,
                      child: ListView(
                        scrollDirection: Axis.horizontal,
                        padding: const EdgeInsets.symmetric(
                          horizontal: AppSpacing.s24,
                        ),
                        children: [
                          _FilterChip(
                            label: _selectedYear ?? 'All Years',
                            isActive: _selectedYear != null,
                            onTap: () {
                              showModalBottomSheet(
                                context: context,
                                backgroundColor: c.surfaceElevated,
                                builder: (ctx) => ListView(
                                  children: [
                                    ListTile(
                                      title: Text(
                                        'All Years',
                                        style: TextStyle(color: c.text),
                                      ),
                                      onTap: () {
                                        setState(() => _selectedYear = null);
                                        Navigator.pop(ctx);
                                      },
                                    ),
                                    ...availableYears.map(
                                      (y) => ListTile(
                                        title: Text(
                                          y,
                                          style: TextStyle(color: c.text),
                                        ),
                                        onTap: () {
                                          setState(() => _selectedYear = y);
                                          Navigator.pop(ctx);
                                        },
                                      ),
                                    ),
                                  ],
                                ),
                              );
                            },
                          ),
                          const SizedBox(width: AppSpacing.s8),
                          _FilterChip(
                            label: _selectedMonth ?? 'All Months',
                            isActive: _selectedMonth != null,
                            onTap: () {
                              showModalBottomSheet(
                                context: context,
                                backgroundColor: c.surfaceElevated,
                                builder: (ctx) => ListView(
                                  children: [
                                    ListTile(
                                      title: Text(
                                        'All Months',
                                        style: TextStyle(color: c.text),
                                      ),
                                      onTap: () {
                                        setState(() => _selectedMonth = null);
                                        Navigator.pop(ctx);
                                      },
                                    ),
                                    ...availableMonths.map(
                                      (m) => ListTile(
                                        title: Text(
                                          m,
                                          style: TextStyle(color: c.text),
                                        ),
                                        onTap: () {
                                          setState(() => _selectedMonth = m);
                                          Navigator.pop(ctx);
                                        },
                                      ),
                                    ),
                                  ],
                                ),
                              );
                            },
                          ),
                        ],
                      ),
                    ),
                  ),

                const SliverToBoxAdapter(
                  child: SizedBox(height: AppSpacing.s16),
                ),

                if (!hasData)
                  SliverFillRemaining(
                    hasScrollBody: false,
                    child: EmptyState(
                      title: 'No memories yet.',
                      subtitle: 'Start building your digital story today.',
                      icon: Icons.timeline,
                      buttonText: 'Create Memory',
                      onTap: () {
                        // Open create memory
                      },
                    ),
                  )
                else if (!hasFilteredData)
                  SliverFillRemaining(
                    hasScrollBody: false,
                    child: Center(
                      child: Text(
                        'No memories match your filters.',
                        style: TextStyle(color: c.textMuted),
                      ),
                    ),
                  )
                else
                  ...filteredGroups.expand(
                    (yearGroup) => [
                      // Year Header
                      SliverToBoxAdapter(
                        child: Padding(
                          padding: const EdgeInsets.fromLTRB(
                            AppSpacing.s24,
                            AppSpacing.s32,
                            AppSpacing.s24,
                            AppSpacing.s16,
                          ),
                          child: Text(
                            yearGroup.year,
                            style: TextStyle(
                              fontSize: 28,
                              fontWeight: FontWeight.w800,
                              color: c.text,
                              letterSpacing: -0.5,
                            ),
                          ),
                        ),
                      ),
                      ...yearGroup.months.expand(
                        (monthGroup) => [
                          // Month Header
                          SliverToBoxAdapter(
                            child: Padding(
                              padding: const EdgeInsets.fromLTRB(
                                AppSpacing.s24,
                                AppSpacing.s16,
                                AppSpacing.s24,
                                AppSpacing.s12,
                              ),
                              child: Text(
                                monthGroup.month.toUpperCase(),
                                style: TextStyle(
                                  fontSize: 13,
                                  fontWeight: FontWeight.w700,
                                  color: c.primary,
                                  letterSpacing: 1.2,
                                ),
                              ),
                            ),
                          ),
                          ...monthGroup.days.expand(
                            (dayGroup) => [
                              // Day Label (Aug 10) + Memories List
                              SliverToBoxAdapter(
                                child: Padding(
                                  padding: const EdgeInsets.fromLTRB(
                                    AppSpacing.s24,
                                    AppSpacing.s8,
                                    AppSpacing.s24,
                                    0,
                                  ),
                                  child: Row(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      // Date Column
                                      SizedBox(
                                        width: 60,
                                        child: Text(
                                          dayGroup.dateLabel,
                                          style: TextStyle(
                                            fontSize: 14,
                                            fontWeight: FontWeight.w600,
                                            color: c.textMuted,
                                            height: 1.3,
                                          ),
                                        ),
                                      ),
                                      // Memories Column
                                      Expanded(
                                        child: Column(
                                          children: dayGroup.memories.map((
                                            memory,
                                          ) {
                                            final cover = _getCoverMedia(
                                              memory,
                                            );
                                            final photos = memory.media
                                                .where(
                                                  (m) => m.mediaType == 'image',
                                                )
                                                .length;
                                            final videos = memory.media
                                                .where(
                                                  (m) => m.mediaType == 'video',
                                                )
                                                .length;

                                            return Padding(
                                              padding: const EdgeInsets.only(
                                                bottom: AppSpacing.s20,
                                              ),
                                              child: GestureDetector(
                                                onTap: () =>
                                                    MemoryDetailScreen.open(
                                                      context,
                                                      memory,
                                                    ),
                                                onLongPress: () =>
                                                    _showMemoryOptions(
                                                      context,
                                                      memory,
                                                    ),
                                                child: Container(
                                                  decoration: BoxDecoration(
                                                    color: c.surfaceElevated,
                                                    borderRadius:
                                                        BorderRadius.circular(
                                                          AppRadii.lg,
                                                        ),
                                                    border: Border.all(
                                                      color: c.border
                                                          .withValues(
                                                            alpha: 0.5,
                                                          ),
                                                    ),
                                                  ),
                                                  clipBehavior: Clip.antiAlias,
                                                  child: Column(
                                                    crossAxisAlignment:
                                                        CrossAxisAlignment
                                                            .start,
                                                    mainAxisSize:
                                                        MainAxisSize.min,
                                                    children: [
                                                      // Cover Image
                                                      if (cover != null)
                                                        Stack(
                                                          children: [
                                                            SizedBox(
                                                              height: 140,
                                                              width: double
                                                                  .infinity,
                                                              child: Image.network(
                                                                cover.thumbnailUrl ??
                                                                    cover.url,
                                                                fit: BoxFit
                                                                    .cover,
                                                                errorBuilder:
                                                                    (
                                                                      _,
                                                                      __,
                                                                      ___,
                                                                    ) => Container(
                                                                      color: c
                                                                          .surface,
                                                                    ),
                                                              ),
                                                            ),
                                                            if (cover.isVideo)
                                                              Positioned(
                                                                top: 8,
                                                                right: 8,
                                                                child: Container(
                                                                  padding:
                                                                      const EdgeInsets.all(
                                                                        4,
                                                                      ),
                                                                  decoration: BoxDecoration(
                                                                    color: Colors
                                                                        .black
                                                                        .withValues(
                                                                          alpha:
                                                                              0.6,
                                                                        ),
                                                                    shape: BoxShape
                                                                        .circle,
                                                                  ),
                                                                  child: const Icon(
                                                                    Icons
                                                                        .play_arrow_rounded,
                                                                    color: Colors
                                                                        .white,
                                                                    size: 16,
                                                                  ),
                                                                ),
                                                              ),
                                                          ],
                                                        )
                                                      else
                                                        Container(
                                                          height: 100,
                                                          width:
                                                              double.infinity,
                                                          decoration: BoxDecoration(
                                                            gradient: LinearGradient(
                                                              colors: [
                                                                c.primary
                                                                    .withValues(
                                                                      alpha:
                                                                          0.2,
                                                                    ),
                                                                c.bg,
                                                              ],
                                                              begin: Alignment
                                                                  .topLeft,
                                                              end: Alignment
                                                                  .bottomRight,
                                                            ),
                                                          ),
                                                        ),

                                                      // Info
                                                      Padding(
                                                        padding:
                                                            const EdgeInsets.all(
                                                              AppSpacing.s16,
                                                            ),
                                                        child: Column(
                                                          crossAxisAlignment:
                                                              CrossAxisAlignment
                                                                  .start,
                                                          mainAxisSize:
                                                              MainAxisSize.min,
                                                          children: [
                                                            Text(
                                                              memory.title,
                                                              style: TextStyle(
                                                                fontSize: 16,
                                                                fontWeight:
                                                                    FontWeight
                                                                        .w700,
                                                                color: c.text,
                                                              ),
                                                              maxLines: 1,
                                                              overflow:
                                                                  TextOverflow
                                                                      .ellipsis,
                                                            ),
                                                            const SizedBox(
                                                              height:
                                                                  AppSpacing.s4,
                                                            ),
                                                            Row(
                                                              children: [
                                                                if (memory
                                                                        .locationName !=
                                                                    null) ...[
                                                                  Icon(
                                                                    Icons
                                                                        .location_on,
                                                                    size: 12,
                                                                    color: c
                                                                        .textMuted,
                                                                  ),
                                                                  const SizedBox(
                                                                    width: 4,
                                                                  ),
                                                                  Expanded(
                                                                    child: Text(
                                                                      memory
                                                                          .locationName!,
                                                                      style: TextStyle(
                                                                        fontSize:
                                                                            12,
                                                                        color: c
                                                                            .textMuted,
                                                                      ),
                                                                      maxLines:
                                                                          1,
                                                                      overflow:
                                                                          TextOverflow
                                                                              .ellipsis,
                                                                    ),
                                                                  ),
                                                                ],
                                                                Text(
                                                                  '${photos > 0 ? '$photos photos' : ''}${photos > 0 && videos > 0 ? ' · ' : ''}${videos > 0 ? '$videos videos' : ''}${photos == 0 && videos == 0 ? '0 media' : ''}',
                                                                  style: TextStyle(
                                                                    fontSize:
                                                                        12,
                                                                    color: c
                                                                        .textMuted,
                                                                    fontWeight:
                                                                        FontWeight
                                                                            .w500,
                                                                  ),
                                                                ),
                                                              ],
                                                            ),
                                                          ],
                                                        ),
                                                      ),
                                                    ],
                                                  ),
                                                ),
                                              ),
                                            );
                                          }).toList(),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ],
                  ),

                const SliverToBoxAdapter(child: SizedBox(height: 100)),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _FilterChip extends StatelessWidget {
  final String label;
  final bool isActive;
  final VoidCallback onTap;

  const _FilterChip({
    required this.label,
    required this.isActive,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color: isActive ? c.primary : c.surfaceElevated,
          borderRadius: BorderRadius.circular(20),
          border: isActive ? null : Border.all(color: c.border),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              label,
              style: TextStyle(
                color: isActive ? c.primaryInverse : c.text,
                fontSize: 13,
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(width: 4),
            Icon(
              Icons.keyboard_arrow_down,
              size: 16,
              color: isActive ? c.primaryInverse : c.textMuted,
            ),
          ],
        ),
      ),
    );
  }
}
