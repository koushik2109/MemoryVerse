import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/navigation/router.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/widgets/media.dart';
import 'package:memory_verse/core/widgets/states.dart';
import 'package:memory_verse/features/memories/presentation/create_memory_sheet.dart';
import 'package:memory_verse/features/vaults/presentation/create_vault_dialog.dart';
import 'package:memory_verse/features/vaults/presentation/join_vault_dialog.dart';
import 'package:memory_verse/features/vaults/presentation/vault_detail_screen.dart';
import 'package:memory_verse/core/presentation/widgets/stacked_carousel.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as design;

class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    const c = AppColors.dark;
    final profileAsync = ref.watch(userProfileProvider);
    final vaultsAsync = ref.watch(vaultsListProvider);
    final memoriesAsync = ref.watch(memoriesListProvider);
    final timelineAsync = ref.watch(timelineProvider);

    return Container(
      decoration: const BoxDecoration(gradient: design.AppGradients.dark),
      child: Scaffold(
        backgroundColor: Colors.transparent,
      body: RefreshIndicator(
        color: c.primary,
        backgroundColor: c.surface,
        onRefresh: () async {
          ref.invalidate(userProfileProvider);
          ref.invalidate(vaultsListProvider);
          ref.invalidate(memoriesListProvider);
          ref.invalidate(timelineProvider);
        },
        child: CustomScrollView(
          physics: const BouncingScrollPhysics(
            parent: AlwaysScrollableScrollPhysics(),
          ),
          slivers: [
            // Header
            _buildHeader(context, c, profileAsync),
            const SliverToBoxAdapter(child: SizedBox(height: AppSpacing.s24)),

            // Hero / CTA
            _buildHeroCTA(context, c, profileAsync),
            const SliverToBoxAdapter(child: SizedBox(height: AppSpacing.s32)),

            // Quick Actions
            _buildQuickActions(context, c),
            const SliverToBoxAdapter(child: SizedBox(height: AppSpacing.s40)),

            // Recent Memories
            _buildSectionHeader(
              context,
              c,
              'Recent Memories',
              'View all',
              () => context.go(Routes.memories),
            ),
            const SliverToBoxAdapter(child: SizedBox(height: AppSpacing.s16)),
            _buildRecentMemories(context, c, memoriesAsync),
            const SliverToBoxAdapter(child: SizedBox(height: AppSpacing.s40)),

            // Timeline Preview
            _buildSectionHeader(
              context,
              c,
              'Your Timeline',
              'Explore',
              () => context.go(Routes.timeline),
            ),
            const SliverToBoxAdapter(child: SizedBox(height: AppSpacing.s16)),
            _buildTimelinePreview(context, c, timelineAsync),
            const SliverToBoxAdapter(child: SizedBox(height: AppSpacing.s40)),

            // Collaborative Rooms (Vaults)
            _buildSectionHeader(
              context,
              c,
              'Rooms',
              'Manage',
              () => context.go(Routes.memories),
            ),
            const SliverToBoxAdapter(child: SizedBox(height: AppSpacing.s16)),
            _buildRooms(context, c, vaultsAsync),

            const SliverToBoxAdapter(child: SizedBox(height: AppSpacing.s80)),
          ],
        ),
      ),
      ),
    );
  }

  Widget _buildHeader(
    BuildContext context,
    AppColors c,
    AsyncValue<UserModel> profileAsync,
  ) {
    final hour = DateTime.now().hour;
    final greeting = hour < 12
        ? 'Good morning'
        : hour < 17
        ? 'Good afternoon'
        : 'Good evening';

    return SliverSafeArea(
      bottom: false,
      sliver: SliverToBoxAdapter(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(
            AppSpacing.s24,
            AppSpacing.s12,
            AppSpacing.s24,
            0,
          ),
          child: Row(
            children: [
              profileAsync.when(
                data: (p) =>
                    Avatar(url: p.avatarUrl, name: p.fullName, size: 44),
                loading: () => const Avatar(size: 44),
                error: (_, __) => const Avatar(size: 44),
              ),
              const SizedBox(width: AppSpacing.s16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    profileAsync.when(
                      data: (p) => Text(
                        '$greeting, ${p.fullName?.split(' ').first ?? 'there'}',
                        style: TextStyle(
                          fontSize: 13,
                          color: c.textMuted,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      loading: () => Text(
                        greeting,
                        style: TextStyle(
                          fontSize: 13,
                          color: c.textMuted,
                        ),
                      ),
                      error: (_, __) => Text(
                        greeting,
                        style: TextStyle(
                          fontSize: 13,
                          color: c.textMuted,
                        ),
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      'MemoryVerse',
                      style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.w700,
                        letterSpacing: -0.5,
                        color: c.text,
                      ),
                    ),
                  ],
                ),
              ),
              _IconButton(
                icon: Icons.search_rounded,
                onTap: () => context.push(Routes.search),
                colors: c,
              ),
              const SizedBox(width: AppSpacing.s8),
              _IconButton(
                icon: Icons.notifications_none_rounded,
                onTap: () => context.push(Routes.notifications),
                colors: c,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeroCTA(
    BuildContext context,
    AppColors c,
    AsyncValue<UserModel> profileAsync,
  ) {
    return SliverToBoxAdapter(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: AppSpacing.s16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Your story.',
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.w700,
                  letterSpacing: -1.2,
                  color: c.text,
                  height: 1.1,
                ),
              ),
              const SizedBox(height: AppSpacing.s8),
              Text(
                'Capture today\'s moments before they fade.',
                style: TextStyle(
                  fontSize: 15,
                  color: c.textMuted,
                  height: 1.4,
                ),
              ),
              const SizedBox(height: 48),
              Center(
                child: ElevatedButton(
                  onPressed: () => CreateMemorySheet.show(context),
                  style: design.AppTheme.primaryButtonOnDark.copyWith(
                    padding: WidgetStateProperty.all(
                      const EdgeInsets.symmetric(horizontal: 32, vertical: 20),
                    ),
                    textStyle: WidgetStateProperty.all(
                      design.AppTextStyles.body.copyWith(fontWeight: FontWeight.w600, fontSize: 16),
                    ),
                  ),
                  child: const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text('Create Memory'),
                      SizedBox(width: 8),
                      Icon(Icons.arrow_forward, size: 20),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildQuickActions(BuildContext context, AppColors c) {
    return SliverToBoxAdapter(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            _QuickActionButton(
              icon: Icons.create_new_folder_outlined,
              label: 'Create Room',
              onTap: () => CreateVaultDialog.show(context),
              colors: c,
            ),
            _QuickActionButton(
              icon: Icons.group_add_outlined,
              label: 'Join Room',
              onTap: () => JoinVaultDialog.show(context),
              colors: c,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionHeader(
    BuildContext context,
    AppColors c,
    String title,
    String action,
    VoidCallback onTap,
  ) {
    return SliverToBoxAdapter(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              title,
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w600,
                letterSpacing: -0.5,
                color: c.text,
              ),
            ),
            GestureDetector(
              onTap: onTap,
              child: Text(
                action,
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                  color: c.primary,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRecentMemories(
    BuildContext context,
    AppColors c,
    AsyncValue<List<MemoryModel>> memoriesAsync,
  ) {
    return SliverToBoxAdapter(
      child: memoriesAsync.when(
        data: (memories) {
          if (memories.isEmpty) {
            const mockItems = [
              CarouselItem(
                title: "Mountain Trek",
                subtitle: "Scale new heights and embrace the hiker's journey.",
                imageUrl: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600&q=80",
                badge: "Adventure",
              ),
              CarouselItem(
                title: "River Rafting",
                subtitle: "Feel the adrenaline rush as you navigate the wild rapids.",
                imageUrl: "https://images.unsplash.com/photo-1504280390367-361c6d9f38f4?w=600&q=80",
                badge: "Extreme",
              ),
              CarouselItem(
                title: "Forest Walk",
                subtitle: "Deep dive into the silence of the ancient woods.",
                imageUrl: "https://images.unsplash.com/photo-1448375240586-882707db888b?w=600&q=80",
                badge: "Nature",
              ),
              CarouselItem(
                title: "Azure Beach",
                subtitle: "Unwind on the crystal clear shores of a tropical paradise.",
                imageUrl: "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=600&q=80",
                badge: "Paradise",
              ),
            ];
            return const StackedCarousel(items: mockItems, showIndicators: false);
          }
          final items = memories.map((m) {
            String imageUrl = 'https://images.unsplash.com/photo-1551632811-561732d1e306?q=80&w=2070'; // fallback
            if (m.media.isNotEmpty) {
              imageUrl = m.media.first.url;
            }
            return CarouselItem(
              title: m.title,
              subtitle: m.locationName ?? m.description,
              imageUrl: imageUrl,
            );
          }).toList();
          return StackedCarousel(items: items, showIndicators: false);
        },
        loading: () => const SizedBox(
          height: 240,
          child: Center(child: CircularProgressIndicator()),
        ),
        error: (_, __) => const SizedBox(height: 240),
      ),
    );
  }

  Widget _buildTimelinePreview(
    BuildContext context,
    AppColors c,
    AsyncValue<TimelineResponse> timelineAsync,
  ) {
    return SliverToBoxAdapter(
      child: timelineAsync.when(
        loading: () => SizedBox(
          height: 120,
          child: ListView.separated(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
            itemCount: 4,
            separatorBuilder: (_, __) => const SizedBox(width: AppSpacing.s12),
            itemBuilder: (_, __) =>
                _SkeletonCard(width: 100, height: 120, colors: c),
          ),
        ),
        error: (_, __) => const EmptyState(
          title: 'No timeline yet.',
          subtitle: 'Memories will appear here chronologically.',
          imageAsset: 'assets/images/empty_timeline.png',
          isSmall: true,
        ),
        data: (timeline) {
          final items = timeline.groups
              .expand((y) => y.months)
              .expand((m) => m.days)
              .expand((d) => d.memories)
              .take(10)
              .toList();
          if (items.isEmpty) {
            return const EmptyState(
              title: 'No timeline yet.',
              subtitle: 'Memories will appear here chronologically.',
              imageAsset: 'assets/images/empty_timeline.png',
              isSmall: true,
            );
          }
          return SizedBox(
            height: 120,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
              itemCount: items.length,
              separatorBuilder: (_, __) =>
                  const SizedBox(width: AppSpacing.s12),
              itemBuilder: (_, i) {
                final item = items[i];
                return _TimelinePreviewCard(item: item, colors: c);
              },
            ),
          );
        },
      ),
    );
  }

  Widget _buildRooms(
    BuildContext context,
    AppColors c,
    AsyncValue<List<VaultModel>> vaultsAsync,
  ) {
    return vaultsAsync.when(
      loading: () => SliverToBoxAdapter(
        child: SizedBox(
          height: 160,
          child: ListView.separated(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
            itemCount: 2,
            separatorBuilder: (_, __) => const SizedBox(width: AppSpacing.s16),
            itemBuilder: (_, __) =>
                _SkeletonCard(width: 240, height: 160, colors: c),
          ),
        ),
      ),
      error: (_, __) => SliverToBoxAdapter(
        child: EmptyState(
          title: 'No rooms yet',
          subtitle: 'Create a room to collaborate.',
          imageAsset: 'assets/images/empty_room.png',
          buttonText: 'Create Room',
          onTap: () => CreateVaultDialog.show(context),
        ),
      ),
      data: (vaults) {
        if (vaults.isEmpty) {
          return SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: AppSpacing.s24,
                vertical: AppSpacing.s16,
              ),
              child: Container(
                padding: const EdgeInsets.all(AppSpacing.s24),
                decoration: BoxDecoration(
                  color: c.surfaceElevated,
                  borderRadius: BorderRadius.circular(AppRadii.lg),
                  border: Border.all(color: c.border.withValues(alpha: 0.5)),
                ),
                child: Column(
                  children: [
                    Icon(
                      Icons.group_outlined,
                      size: 48,
                      color: c.primary.withValues(alpha: 0.5),
                    ),
                    const SizedBox(height: AppSpacing.s16),
                    Text(
                      'Share memories together.',
                      style: TextStyle(color: c.textMuted, fontSize: 14),
                    ),
                    const SizedBox(height: AppSpacing.s16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        TextButton(
                          onPressed: () => CreateVaultDialog.show(context),
                          child: Text(
                            'Create Room',
                            style: TextStyle(
                              color: c.primary,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                        const SizedBox(width: AppSpacing.s16),
                        TextButton(
                          onPressed: () => JoinVaultDialog.show(context),
                          child: Text(
                            'Join Room',
                            style: TextStyle(
                              color: c.primary,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          );
        }
        return SliverToBoxAdapter(
          child: SizedBox(
            height: 160,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
              itemCount: vaults.length,
              separatorBuilder: (_, __) =>
                  const SizedBox(width: AppSpacing.s16),
              itemBuilder: (_, i) {
                final v = vaults[i];
                return _RoomCard(
                  vault: v,
                  onTap: () => Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) => VaultDetailScreen(vaultId: v.id),
                    ),
                  ),
                  colors: c,
                );
              },
            ),
          ),
        );
      },
    );
  }
}

// --- Supporting Widgets ---

class _IconButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  final AppColors colors;
  const _IconButton({
    required this.icon,
    required this.onTap,
    required this.colors,
  });

  @override
  Widget build(BuildContext context) {
    return IconButton(
      onPressed: onTap,
      style: design.AppTheme.iconButtonStyle(onDark: true),
      icon: Icon(icon, size: 20),
    );
  }
}

class _QuickActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final AppColors colors;
  const _QuickActionButton({
    required this.icon,
    required this.label,
    required this.onTap,
    required this.colors,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width:
            (MediaQuery.sizeOf(context).width - 48 - 16) / 2, // evenly spaced for 2 items
        padding: const EdgeInsets.symmetric(vertical: AppSpacing.s8),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 32, color: colors.text),
            const SizedBox(height: AppSpacing.s8),
            Text(
              label,
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w500,
                color: colors.textMuted,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _TimelinePreviewCard extends StatelessWidget {
  final MemoryModel item;
  final AppColors colors;
  const _TimelinePreviewCard({required this.item, required this.colors});

  @override
  Widget build(BuildContext context) {
    final imageUrl = item.media.isNotEmpty
        ? (item.media.first.thumbnailUrl ?? item.media.first.url)
        : null;

    return Container(
      width: 100,
      height: 120,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(AppRadii.lg),
        color: colors.surfaceElevated,
      ),
      clipBehavior: Clip.antiAlias,
      child: imageUrl != null
          ? Image.network(
              imageUrl,
              fit: BoxFit.cover,
              errorBuilder: (_, __, ___) =>
                  Container(color: colors.surfaceElevated),
            )
          : Container(
              color: colors.surfaceElevated,
              child: Icon(
                Icons.photo_library_outlined,
                color: colors.primary.withValues(alpha: 0.3),
              ),
            ),
    );
  }
}

class _RoomCard extends StatelessWidget {
  final VaultModel vault;
  final VoidCallback onTap;
  final AppColors colors;
  const _RoomCard({
    required this.vault,
    required this.onTap,
    required this.colors,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 240,
        height: 160,
        padding: const EdgeInsets.all(AppSpacing.s20),
        decoration: BoxDecoration(
          color: colors.surface,
          borderRadius: BorderRadius.circular(AppRadii.xl),
          border: Border.all(color: colors.border),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: colors.primary.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(AppRadii.md),
                  ),
                  child: Icon(
                    Icons.meeting_room_rounded,
                    size: 20,
                    color: colors.primary,
                  ),
                ),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: colors.surfaceElevated,
                    borderRadius: BorderRadius.circular(100.0),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        Icons.people_alt_rounded,
                        size: 12,
                        color: colors.textMuted,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        '${vault.memberCount}',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: colors.text,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  vault.name,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: colors.text,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '${vault.mediaCount} memories',
                  style: TextStyle(
                    fontSize: 13,
                    color: colors.textMuted,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _SkeletonCard extends StatelessWidget {
  final double width;
  final double height;
  final AppColors colors;
  const _SkeletonCard({
    required this.width,
    required this.height,
    required this.colors,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: width,
      height: height,
      decoration: BoxDecoration(
        color: colors.surfaceElevated,
        borderRadius: BorderRadius.circular(AppRadii.xl),
      ),
    );
  }
}
