import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/widgets/media.dart';

class SearchScreen extends ConsumerStatefulWidget {
  const SearchScreen({super.key});

  @override
  ConsumerState<SearchScreen> createState() => _SearchScreenState();
}

class _SearchScreenState extends ConsumerState<SearchScreen> {
  final _controller = TextEditingController();
  String _query = '';

  static const List<String> _suggestedQueries = [
    'birthday photos',
    'photos from the beach',
    'people celebrating',
    'sunset',
    'family',
    'trip photos',
    'nature',
    'party',
  ];

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _applyQuery(String q) {
    _controller.text = q;
    _controller.selection = TextSelection.fromPosition(
      TextPosition(offset: q.length),
    );
    setState(() => _query = q.trim());
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final searchAsync = _query.isNotEmpty
        ? ref.watch(searchResultsProvider(_query))
        : null;

    return Scaffold(
      backgroundColor: c.bg,
      appBar: AppBar(
        backgroundColor: c.bg,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded),
          onPressed: () => context.pop(),
        ),
        title: TextField(
          controller: _controller,
          autofocus: true,
          style: Theme.of(context).textTheme.bodyMedium,
          decoration: InputDecoration(
            hintText: 'Search memories, vaults, AI tags...',
            hintStyle: TextStyle(color: c.textMuted),
            border: InputBorder.none,
            enabledBorder: InputBorder.none,
            focusedBorder: InputBorder.none,
            filled: false,
            contentPadding: EdgeInsets.zero,
          ),
          onChanged: (v) => setState(() => _query = v.trim()),
          onSubmitted: (v) => setState(() => _query = v.trim()),
        ),
        actions: [
          if (_query.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.close_rounded),
              onPressed: () {
                _controller.clear();
                setState(() => _query = '');
              },
            ),
        ],
      ),
      body: Column(
        children: [
          // Quick suggestions carousel
          Container(
            height: 48,
            padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s16),
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              itemCount: _suggestedQueries.length,
              separatorBuilder: (_, __) => const SizedBox(width: AppSpacing.s8),
              itemBuilder: (context, i) {
                final q = _suggestedQueries[i];
                final isSelected = _query.toLowerCase() == q.toLowerCase();
                return Center(
                  child: ActionChip(
                    label: Text(q),
                    labelStyle: TextStyle(
                      fontSize: 12,
                      fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
                      color: isSelected ? Colors.white : c.textMuted,
                    ),
                    backgroundColor: isSelected ? c.primary : c.surfaceElevated,
                    side: BorderSide(
                      color: isSelected ? c.primary : c.border,
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(AppRadii.pill),
                    ),
                    onPressed: () => _applyQuery(q),
                  ),
                );
              },
            ),
          ),
          const Divider(height: 1),

          // Search results or empty state
          Expanded(
            child: _query.isEmpty
                ? _EmptySearch(c: c, onSelect: _applyQuery)
                : searchAsync?.when(
                        loading: () => Center(
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              CircularProgressIndicator(
                                color: c.primary,
                                strokeWidth: 2.5,
                              ),
                              const SizedBox(height: AppSpacing.s16),
                              Text(
                                'Searching with multimodal AI...',
                                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                      color: c.textMuted,
                                    ),
                              ),
                            ],
                          ),
                        ),
                        error: (e, _) => Center(
                          child: Padding(
                            padding: const EdgeInsets.all(AppSpacing.s24),
                            child: Text(
                              'Search error: $e',
                              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                    color: c.error,
                                  ),
                              textAlign: TextAlign.center,
                            ),
                          ),
                        ),
                        data: (results) {
                          final rawVaults = results['vaults'] as List? ?? [];
                          final rawMedia = results['media'] as List? ?? [];

                          final vaults = rawVaults
                              .map((v) => VaultModel.fromJson(Map<String, dynamic>.from(v)))
                              .toList();
                          final media = rawMedia
                              .map((m) => MediaModel.fromJson(Map<String, dynamic>.from(m)))
                              .toList();

                          if (vaults.isEmpty && media.isEmpty) {
                            return Center(
                              child: Column(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Icon(
                                    Icons.search_off_rounded,
                                    size: 56,
                                    color: c.textMuted,
                                  ),
                                  const SizedBox(height: AppSpacing.s16),
                                  Text(
                                    'No matching memories found',
                                    style: Theme.of(context).textTheme.headlineSmall,
                                  ),
                                  const SizedBox(height: AppSpacing.s4),
                                  Text(
                                    'Try searching for "beach", "sunset", "party", or "birthday"',
                                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                          color: c.textMuted,
                                        ),
                                  ),
                                ],
                              ),
                            );
                          }

                          return CustomScrollView(
                            slivers: [
                              if (vaults.isNotEmpty) ...[
                                SliverToBoxAdapter(
                                  child: Padding(
                                    padding: const EdgeInsets.fromLTRB(
                                      AppSpacing.s16,
                                      AppSpacing.s16,
                                      AppSpacing.s16,
                                      AppSpacing.s8,
                                    ),
                                    child: Row(
                                      children: [
                                        Icon(Icons.folder_special_rounded, size: 18, color: c.primary),
                                        const SizedBox(width: AppSpacing.s8),
                                        Text(
                                          'Matching Vaults (${vaults.length})',
                                          style: Theme.of(context).textTheme.titleSmall?.copyWith(
                                                fontWeight: FontWeight.bold,
                                              ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                                SliverToBoxAdapter(
                                  child: SizedBox(
                                    height: 110,
                                    child: ListView.separated(
                                      scrollDirection: Axis.horizontal,
                                      padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s16),
                                      itemCount: vaults.length,
                                      separatorBuilder: (_, __) => const SizedBox(width: AppSpacing.s12),
                                      itemBuilder: (context, i) {
                                        final v = vaults[i];
                                        return _VaultSearchCard(vault: v, c: c);
                                      },
                                    ),
                                  ),
                                ),
                              ],
                              if (media.isNotEmpty) ...[
                                SliverToBoxAdapter(
                                  child: Padding(
                                    padding: const EdgeInsets.fromLTRB(
                                      AppSpacing.s16,
                                      AppSpacing.s20,
                                      AppSpacing.s16,
                                      AppSpacing.s12,
                                    ),
                                    child: Row(
                                      children: [
                                        Icon(Icons.photo_library_rounded, size: 18, color: c.primary),
                                        const SizedBox(width: AppSpacing.s8),
                                        Text(
                                          'Media Results (${media.length})',
                                          style: Theme.of(context).textTheme.titleSmall?.copyWith(
                                                fontWeight: FontWeight.bold,
                                              ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                                SliverPadding(
                                  padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s16),
                                  sliver: SliverGrid(
                                    gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                                      crossAxisCount: 2,
                                      crossAxisSpacing: 12,
                                      mainAxisSpacing: 12,
                                      childAspectRatio: 0.72,
                                    ),
                                    delegate: SliverChildBuilderDelegate(
                                      (context, i) {
                                        final m = media[i];
                                        return _MediaSearchCard(media: m, c: c);
                                      },
                                      childCount: media.length,
                                    ),
                                  ),
                                ),
                                const SliverToBoxAdapter(
                                  child: SizedBox(height: AppSpacing.s32),
                                ),
                              ],
                            ],
                          );
                        },
                      ) ??
                    const SizedBox.shrink(),
          ),
        ],
      ),
    );
  }
}

class _EmptySearch extends StatelessWidget {
  const _EmptySearch({required this.c, required this.onSelect});
  final AppColors c;
  final ValueChanged<String> onSelect;

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(AppSpacing.s24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Center(
            child: Column(
              children: [
                const SizedBox(height: AppSpacing.s20),
                Icon(Icons.travel_explore_rounded, size: 56, color: c.primary),
                const SizedBox(height: AppSpacing.s16),
                Text(
                  'Multimodal Semantic Search',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                ),
                const SizedBox(height: AppSpacing.s8),
                Text(
                  'Search across visual AI tags, locations, and CLIP vector embeddings naturally.',
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: c.textMuted,
                      ),
                ),
                const SizedBox(height: AppSpacing.s24),
              ],
            ),
          ),
          Text(
            'Try Searching For',
            style: Theme.of(context).textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
          ),
          const SizedBox(height: AppSpacing.s12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _SearchSuggestionPill(
                icon: Icons.cake_outlined,
                label: 'birthday photos',
                c: c,
                onTap: () => onSelect('birthday photos'),
              ),
              _SearchSuggestionPill(
                icon: Icons.beach_access_outlined,
                label: 'photos from the beach',
                c: c,
                onTap: () => onSelect('photos from the beach'),
              ),
              _SearchSuggestionPill(
                icon: Icons.celebration_outlined,
                label: 'people celebrating',
                c: c,
                onTap: () => onSelect('people celebrating'),
              ),
              _SearchSuggestionPill(
                icon: Icons.wb_twilight_rounded,
                label: 'sunset',
                c: c,
                onTap: () => onSelect('sunset'),
              ),
              _SearchSuggestionPill(
                icon: Icons.groups_outlined,
                label: 'family',
                c: c,
                onTap: () => onSelect('family'),
              ),
              _SearchSuggestionPill(
                icon: Icons.directions_car_outlined,
                label: 'trip photos',
                c: c,
                onTap: () => onSelect('trip photos'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _SearchSuggestionPill extends StatelessWidget {
  const _SearchSuggestionPill({
    required this.icon,
    required this.label,
    required this.c,
    required this.onTap,
  });

  final IconData icon;
  final String label;
  final AppColors c;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(AppRadii.pill),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: c.surface,
          borderRadius: BorderRadius.circular(AppRadii.pill),
          border: Border.all(color: c.border),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 16, color: c.primary),
            const SizedBox(width: 8),
            Text(
              label,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    fontWeight: FontWeight.w500,
                  ),
            ),
          ],
        ),
      ),
    );
  }
}

class _VaultSearchCard extends StatelessWidget {
  const _VaultSearchCard({required this.vault, required this.c});
  final VaultModel vault;
  final AppColors c;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 200,
      padding: const EdgeInsets.all(AppSpacing.s12),
      decoration: BoxDecoration(
        color: c.surface,
        borderRadius: BorderRadius.circular(AppRadii.md),
        border: Border.all(color: c.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Row(
            children: [
              Container(
                width: 36,
                height: 36,
                decoration: BoxDecoration(
                  color: c.primary.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(AppRadii.sm),
                ),
                child: Icon(Icons.folder_rounded, color: c.primary, size: 20),
              ),
              const SizedBox(width: AppSpacing.s10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      vault.name,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            fontWeight: FontWeight.w600,
                          ),
                    ),
                    Text(
                      '${vault.mediaCount} items • ${vault.memberCount} members',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: c.textMuted,
                            fontSize: 11,
                          ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          if (vault.description != null && vault.description!.isNotEmpty) ...[
            const SizedBox(height: AppSpacing.s8),
            Text(
              vault.description!,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: c.textMuted,
                    fontSize: 11,
                  ),
            ),
          ],
        ],
      ),
    );
  }
}

class _MediaSearchCard extends StatelessWidget {
  const _MediaSearchCard({required this.media, required this.c});
  final MediaModel media;
  final AppColors c;

  @override
  Widget build(BuildContext context) {
    final tags = media.aiTags;
    final isAlt = media.isAlternate;
    final altCount = media.alternateCount;

    return Container(
      decoration: BoxDecoration(
        color: c.surface,
        borderRadius: BorderRadius.circular(AppRadii.md),
        border: Border.all(
          color: isAlt ? Colors.amber.withValues(alpha: 0.4) : c.border,
        ),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Media preview image
          Expanded(
            child: Stack(
              fit: StackFit.expand,
              children: [
                AppNetworkImage(
                  imageUrl: media.thumbnailUrl ?? media.url,
                  fit: BoxFit.cover,
                ),
                // Video indicator overlay
                if (media.isVideo)
                  Positioned(
                    top: 8,
                    left: 8,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.65),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: const Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.videocam_rounded, size: 12, color: Colors.white),
                          SizedBox(width: 4),
                          Text(
                            'VIDEO',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 9,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                // Collaborative Duplicate badge
                if (isAlt)
                  Positioned(
                    top: 8,
                    right: 8,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
                      decoration: BoxDecoration(
                        color: Colors.amber.shade800,
                        borderRadius: BorderRadius.circular(AppRadii.pill),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withValues(alpha: 0.3),
                            blurRadius: 4,
                          ),
                        ],
                      ),
                      child: const Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.copy_rounded, size: 10, color: Colors.white),
                          SizedBox(width: 4),
                          Text(
                            'Alternate',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 10,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ),
                  )
                else if (altCount > 0)
                  Positioned(
                    top: 8,
                    right: 8,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
                      decoration: BoxDecoration(
                        color: c.primary,
                        borderRadius: BorderRadius.circular(AppRadii.pill),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withValues(alpha: 0.3),
                            blurRadius: 4,
                          ),
                        ],
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.auto_awesome_rounded, size: 10, color: Colors.white),
                          const SizedBox(width: 4),
                          Text(
                            '+$altCount Alt',
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 10,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ),

          // Metadata footer
          Padding(
            padding: const EdgeInsets.all(AppSpacing.s8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  media.filename,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        fontWeight: FontWeight.w600,
                        fontSize: 12,
                      ),
                ),
                const SizedBox(height: 2),
                if (media.locationName != null && media.locationName!.isNotEmpty)
                  Row(
                    children: [
                      Icon(Icons.location_on_rounded, size: 11, color: c.textMuted),
                      const SizedBox(width: 2),
                      Expanded(
                        child: Text(
                          media.locationName!,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(color: c.textMuted, fontSize: 10),
                        ),
                      ),
                    ],
                  ),
                if (tags.isNotEmpty) ...[
                  const SizedBox(height: 4),
                  Wrap(
                    spacing: 4,
                    runSpacing: 2,
                    children: tags.take(3).map((tag) {
                      return Container(
                        padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1.5),
                        decoration: BoxDecoration(
                          color: c.surfaceElevated,
                          borderRadius: BorderRadius.circular(3),
                          border: Border.all(color: c.border, width: 0.5),
                        ),
                        child: Text(
                          '#$tag',
                          style: TextStyle(
                            fontSize: 9,
                            color: c.textMuted,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}
