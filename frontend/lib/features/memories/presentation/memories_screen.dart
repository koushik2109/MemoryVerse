import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/widgets/states.dart';
import 'package:memory_verse/features/memories/presentation/create_memory_sheet.dart';
import 'package:memory_verse/features/memories/presentation/memory_detail_screen.dart';

class MemoriesScreen extends ConsumerStatefulWidget {
  const MemoriesScreen({super.key});

  @override
  ConsumerState<MemoriesScreen> createState() => _MemoriesScreenState();
}

class _MemoriesScreenState extends ConsumerState<MemoriesScreen> {
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
              padding: const EdgeInsets.fromLTRB(AppSpacing.s20, AppSpacing.s12, AppSpacing.s20, 0),
              child: Row(
                children: [
                  Expanded(
                    child: Text('Memories', style: Theme.of(context).textTheme.displaySmall?.copyWith(color: c.text)),
                  ),
                  IconButton(
                    icon: Icon(Icons.add_circle_outline_rounded, color: c.primary),
                    onPressed: () => CreateMemorySheet.show(context),
                  ),
                ],
              ),
            ),

            // Month label
            Padding(
              padding: const EdgeInsets.fromLTRB(AppSpacing.s20, AppSpacing.s4, AppSpacing.s20, AppSpacing.s12),
              child: Text(
                DateFormat('MMMM yyyy').format(DateTime.now()),
                style: Theme.of(context).textTheme.bodySmall?.copyWith(color: c.textMuted),
              ),
            ),

            const SizedBox(height: AppSpacing.s16),

            // ── Memory list ─────────────────────────────
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

                  // Group by date
                  final today = DateTime.now();
                  final groups = <String, List<MemoryModel>>{};
                  for (final m in memories) {
                    final diff = today.difference(m.memoryDate).inDays;
                    final label = diff == 0 ? 'Today' : diff == 1 ? 'Yesterday' : DateFormat('MMMM yyyy').format(m.memoryDate);
                    groups.putIfAbsent(label, () => []).add(m);
                  }

                  return RefreshIndicator(
                    color: c.primary,
                    onRefresh: () async => ref.invalidate(memoriesListProvider),
                    child: ListView.builder(
                      padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s20),
                      itemCount: groups.length,
                      itemBuilder: (_, gi) {
                        final label = groups.keys.elementAt(gi);
                        final items = groups[label]!;
                        return Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            if (gi > 0) const SizedBox(height: AppSpacing.s24),
                            Text(label, style: Theme.of(context).textTheme.titleSmall?.copyWith(color: c.textMuted, fontWeight: FontWeight.w600)),
                            const SizedBox(height: AppSpacing.s12),
                            GridView.builder(
                              shrinkWrap: true,
                              physics: const NeverScrollableScrollPhysics(),
                              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                                crossAxisCount: 2,
                                crossAxisSpacing: AppSpacing.s12,
                                mainAxisSpacing: AppSpacing.s12,
                                childAspectRatio: 0.8,
                              ),
                              itemCount: items.length,
                              itemBuilder: (_, i) => _MemoryGridItem(memory: items[i]),
                            )
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

class _MemoryGridItem extends StatelessWidget {
  const _MemoryGridItem({required this.memory});
  final MemoryModel memory;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    
    final String? imageUrl = memory.media.isNotEmpty 
        ? (memory.media.first.thumbnailUrl ?? memory.media.first.url)
        : null;
        
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
              Image.network(
                imageUrl,
                fit: BoxFit.cover,
                errorBuilder: (_, __, ___) => Container(color: c.surfaceElevated),
              )
            else
              Container(
                color: c.primary.withValues(alpha: 0.1),
                child: Center(
                  child: Icon(Icons.auto_awesome_rounded, color: c.primary.withValues(alpha: 0.5), size: 32),
                ),
              ),
            Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [Colors.transparent, Colors.black.withValues(alpha: 0.7)],
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
                    style: const TextStyle(fontFamily: 'Inter', fontSize: 13, fontWeight: FontWeight.w600, color: Colors.white),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    '${memory.media.length} items',
                    style: TextStyle(fontFamily: 'Inter', fontSize: 11, color: Colors.white.withValues(alpha: 0.8)),
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
