import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';

class SearchScreen extends ConsumerStatefulWidget {
  const SearchScreen({super.key});

  @override
  ConsumerState<SearchScreen> createState() => _SearchScreenState();
}

class _SearchScreenState extends ConsumerState<SearchScreen> {
  final _controller = TextEditingController();
  String _query = '';

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
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
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded),
          onPressed: () => context.pop(),
        ),
        title: TextField(
          controller: _controller,
          autofocus: true,
          style: Theme.of(context).textTheme.bodyMedium,
          decoration: const InputDecoration(
            hintText: 'Search memories, vaults...',
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
      body: _query.isEmpty
          ? _EmptySearch(c: c)
          : searchAsync?.when(
                  loading: () => Center(
                    child: CircularProgressIndicator(
                      color: c.primary,
                      strokeWidth: 2,
                    ),
                  ),
                  error: (e, _) => Center(
                    child: Text(
                      'Error: $e',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ),
                  data: (results) {
                    final vaults = results['vaults'] as List? ?? [];
                    final media = results['media'] as List? ?? [];

                    if (vaults.isEmpty && media.isEmpty) {
                      return Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              Icons.search_off_rounded,
                              size: 48,
                              color: c.textMuted,
                            ),
                            const SizedBox(height: AppSpacing.s16),
                            Text(
                              'No results found',
                              style: Theme.of(context).textTheme.headlineSmall,
                            ),
                            const SizedBox(height: AppSpacing.s4),
                            Text(
                              'Try a different search term',
                              style: Theme.of(context).textTheme.bodySmall
                                  ?.copyWith(color: c.textMuted),
                            ),
                          ],
                        ),
                      );
                    }

                    return ListView(
                      padding: const EdgeInsets.all(AppSpacing.s20),
                      children: [
                        if (media.isNotEmpty) ...[
                          Text(
                            'Media',
                            style: Theme.of(context).textTheme.titleSmall
                                ?.copyWith(color: c.textMuted),
                          ),
                          const SizedBox(height: AppSpacing.s10),
                          ...media.map(
                            (m) => Container(
                              margin: const EdgeInsets.only(
                                bottom: AppSpacing.s8,
                              ),
                              padding: const EdgeInsets.all(AppSpacing.s12),
                              decoration: BoxDecoration(
                                color: c.surface,
                                borderRadius: BorderRadius.circular(
                                  AppRadii.md,
                                ),
                                border: Border.all(color: c.border),
                              ),
                              child: Text(
                                m.toString(),
                                style: Theme.of(context).textTheme.bodySmall,
                              ),
                            ),
                          ),
                        ],
                        if (vaults.isNotEmpty) ...[
                          if (media.isNotEmpty)
                            const SizedBox(height: AppSpacing.s20),
                          Text(
                            'Vaults',
                            style: Theme.of(context).textTheme.titleSmall
                                ?.copyWith(color: c.textMuted),
                          ),
                          const SizedBox(height: AppSpacing.s10),
                          ...vaults.map(
                            (v) => Container(
                              margin: const EdgeInsets.only(
                                bottom: AppSpacing.s8,
                              ),
                              padding: const EdgeInsets.all(AppSpacing.s12),
                              decoration: BoxDecoration(
                                color: c.surface,
                                borderRadius: BorderRadius.circular(
                                  AppRadii.md,
                                ),
                                border: Border.all(color: c.border),
                              ),
                              child: Text(
                                v.toString(),
                                style: Theme.of(context).textTheme.bodySmall,
                              ),
                            ),
                          ),
                        ],
                      ],
                    );
                  },
                ) ??
                const SizedBox.shrink(),
    );
  }
}

class _EmptySearch extends StatelessWidget {
  const _EmptySearch({required this.c});
  final AppColors c;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.search_rounded, size: 48, color: c.textMuted),
          const SizedBox(height: AppSpacing.s16),
          Text(
            'Search your memories',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: AppSpacing.s4),
          Text(
            'Find photos, videos, vaults, and more',
            style: Theme.of(context).textTheme.bodySmall
                ?.copyWith(color: c.textMuted),
          ),
        ],
      ),
    );
  }
}
