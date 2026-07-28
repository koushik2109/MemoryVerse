import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:supabase_flutter/supabase_flutter.dart' as sb;

import 'package:frontend/features/authentication/providers/auth_provider.dart';
import 'package:frontend/features/memory/models/memory.dart';
import 'package:frontend/features/memory/providers/memory_provider.dart';

// ─────────────────────────────────────────────────────────────────────────────
//  Cosmic Home Page
// ─────────────────────────────────────────────────────────────────────────────

class HomePage extends ConsumerWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final memories = ref.watch(memoriesProvider);
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      backgroundColor: cs.surface,
      body: CustomScrollView(
        physics: const BouncingScrollPhysics(),
        slivers: [
          // ── Hero Cosmic Header ─────────────────────────────────────────────
          SliverToBoxAdapter(
            child: _HomeHeroHeader(memoriesCount: memories.length),
          ),

          // ── Stats row ─────────────────────────────────────────────────────
          SliverPadding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            sliver: SliverToBoxAdapter(
              child: _StatsRow(memories: memories),
            ),
          ),

          const SliverToBoxAdapter(child: SizedBox(height: 28)),

          // ── Section title ─────────────────────────────────────────────────
          SliverPadding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            sliver: SliverToBoxAdapter(
              child: Row(
                children: [
                  const Icon(Icons.auto_awesome_rounded,
                      size: 20, color: Color(0xFFA78BFA)),
                  const SizedBox(width: 8),
                  Text(
                    'Cosmic Memories',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.w700,
                          letterSpacing: -0.3,
                        ),
                  ),
                  const Spacer(),
                  TextButton(
                    onPressed: () {},
                    child: Text(
                      'Explore all',
                      style: TextStyle(
                        color: cs.primary,
                        fontWeight: FontWeight.w600,
                        fontSize: 13,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),

          const SliverToBoxAdapter(child: SizedBox(height: 12)),

          // ── Memory grid ───────────────────────────────────────────────────
          memories.isEmpty
              ? SliverFillRemaining(
                  hasScrollBody: false,
                  child: _EmptyState(),
                )
              : SliverPadding(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  sliver: SliverGrid(
                    gridDelegate:
                        const SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 2,
                      mainAxisSpacing: 14,
                      crossAxisSpacing: 14,
                      childAspectRatio: 0.82,
                    ),
                    delegate: SliverChildBuilderDelegate(
                      (ctx, i) => _MemoryCard(memory: memories[i], index: i),
                      childCount: memories.length,
                    ),
                  ),
                ),

          const SliverToBoxAdapter(child: SizedBox(height: 110)),
        ],
      ),

      // ── Cosmic FAB ────────────────────────────────────────────────────────
      floatingActionButton: _AddMemoryFab(),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Hero Header with Space / Nebula Background
// ─────────────────────────────────────────────────────────────────────────────

class _HomeHeroHeader extends ConsumerWidget {
  const _HomeHeroHeader({required this.memoriesCount});
  final int memoriesCount;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final tt = Theme.of(context).textTheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    final sbUser = sb.Supabase.instance.client.auth.currentUser;
    final displayName =
        sbUser?.userMetadata?['display_name'] as String? ??
            sbUser?.email?.split('@').first ??
            'Stargazer';
    final avatarUrl = sbUser?.userMetadata?['avatar_url'] as String?;
    final initials = displayName.isNotEmpty
        ? displayName.trim().split(' ').map((p) => p[0]).take(2).join().toUpperCase()
        : 'MV';

    final hour = DateTime.now().hour;
    final greeting = hour < 12
        ? 'Good morning'
        : hour < 17
            ? 'Good afternoon'
            : 'Good evening';

    return Stack(
      children: [
        // Cosmic Nebula Hero Background
        Container(
          height: 250,
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: isDark
                  ? const [
                      Color(0xFF2E1065), // Nebula Violet
                      Color(0xFF0F172A), // Deep Space
                      Color(0xFF030712), // Void Black
                    ]
                  : const [
                      Color(0xFF4C1D95), // Deep Purple Nebula
                      Color(0xFF6D28D9), // Vibrant Cosmic Purple
                      Color(0xFF0284C7), // Galaxy Blue Accent
                    ],
            ),
          ),
        ),

        // Glowing Star / Planet Orbs
        Positioned(
          right: -20,
          top: -20,
          child: Container(
            width: 170,
            height: 170,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: RadialGradient(
                colors: [
                  const Color(0xFFA78BFA).withValues(alpha: 0.25),
                  Colors.transparent,
                ],
              ),
            ),
          ),
        ),
        Positioned(
          left: 40,
          top: 100,
          child: Container(
            width: 90,
            height: 90,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: RadialGradient(
                colors: [
                  const Color(0xFF38BDF8).withValues(alpha: 0.18),
                  Colors.transparent,
                ],
              ),
            ),
          ),
        ),
        Positioned(
          right: 80,
          bottom: 30,
          child: const Icon(
            Icons.auto_awesome_rounded,
            size: 16,
            color: Color(0xFFFDE047),
          ),
        ),

        // Content Overlay
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 56, 20, 28),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Text(
                              '$greeting, ',
                              style: tt.bodyMedium?.copyWith(
                                color: Colors.white.withValues(alpha: 0.75),
                                fontWeight: FontWeight.w400,
                              ),
                            ),
                            const Text('✨', style: TextStyle(fontSize: 14)),
                          ],
                        ),
                        const SizedBox(height: 2),
                        Text(
                          displayName,
                          style: tt.headlineSmall?.copyWith(
                            color: Colors.white,
                            fontWeight: FontWeight.w800,
                            letterSpacing: -0.5,
                          ),
                        ),
                      ],
                    ),
                  ),

                  // Profile avatar -> bottom sheet
                  GestureDetector(
                    onTap: () => _showProfileSheet(context, ref,
                        displayName: displayName,
                        email: sbUser?.email ?? '',
                        initials: initials,
                        avatarUrl: avatarUrl),
                    child: Hero(
                      tag: 'profile_avatar',
                      child: Container(
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: const Color(0xFFA78BFA).withValues(alpha: 0.6),
                            width: 2.5,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: const Color(0xFF7C3AED).withValues(alpha: 0.4),
                              blurRadius: 10,
                              spreadRadius: 1,
                            ),
                          ],
                        ),
                        child: CircleAvatar(
                          radius: 24,
                          backgroundColor: Colors.white.withValues(alpha: 0.15),
                          backgroundImage: avatarUrl != null
                              ? NetworkImage(avatarUrl)
                              : null,
                          child: avatarUrl == null
                              ? Text(
                                  initials,
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontWeight: FontWeight.w700,
                                    fontSize: 16,
                                  ),
                                )
                              : null,
                        ),
                      ),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 22),

              // Subline with space motif
              Row(
                children: [
                  const Icon(Icons.rocket_launch_rounded,
                      size: 16, color: Color(0xFF38BDF8)),
                  const SizedBox(width: 6),
                  Text(
                    memoriesCount == 0
                        ? 'Store your first verse in the cosmos ✨'
                        : '$memoriesCount ${memoriesCount == 1 ? 'memory' : 'memories'} preserved in orbit',
                    style: tt.bodySmall?.copyWith(
                      color: Colors.white.withValues(alpha: 0.8),
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }

  void _showProfileSheet(
    BuildContext context,
    WidgetRef ref, {
    required String displayName,
    required String email,
    required String initials,
    String? avatarUrl,
  }) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _ProfileSheet(
        ref: ref,
        displayName: displayName,
        email: email,
        initials: initials,
        avatarUrl: avatarUrl,
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Profile Sheet
// ─────────────────────────────────────────────────────────────────────────────

class _ProfileSheet extends StatelessWidget {
  const _ProfileSheet({
    required this.ref,
    required this.displayName,
    required this.email,
    required this.initials,
    this.avatarUrl,
  });

  final WidgetRef ref;
  final String displayName;
  final String email;
  final String initials;
  final String? avatarUrl;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return ClipRRect(
      borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
        child: Container(
          decoration: BoxDecoration(
            color: isDark
                ? const Color(0xFF0D1527).withValues(alpha: 0.96)
                : Colors.white.withValues(alpha: 0.96),
            borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
            border: Border(
              top: BorderSide(
                color: const Color(0xFF7C3AED).withValues(alpha: 0.3),
              ),
            ),
          ),
          child: SafeArea(
            top: false,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(24, 16, 24, 24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 40,
                    height: 4,
                    decoration: BoxDecoration(
                      color: cs.outlineVariant,
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),

                  const SizedBox(height: 24),

                  Hero(
                    tag: 'profile_avatar',
                    child: CircleAvatar(
                      radius: 40,
                      backgroundColor: cs.primaryContainer,
                      backgroundImage:
                          avatarUrl != null ? NetworkImage(avatarUrl!) : null,
                      child: avatarUrl == null
                          ? Text(
                              initials,
                              style: tt.headlineSmall?.copyWith(
                                color: cs.onPrimaryContainer,
                                fontWeight: FontWeight.w700,
                              ),
                            )
                          : null,
                    ),
                  ),

                  const SizedBox(height: 16),

                  Text(
                    displayName,
                    style: tt.titleLarge?.copyWith(
                      fontWeight: FontWeight.w700,
                      letterSpacing: -0.4,
                    ),
                  ),

                  const SizedBox(height: 4),

                  Text(
                    email,
                    style: tt.bodyMedium?.copyWith(
                      color: cs.onSurfaceVariant,
                    ),
                  ),

                  const SizedBox(height: 28),

                  Divider(color: cs.outlineVariant),

                  const SizedBox(height: 12),

                  _SheetTile(
                    icon: Icons.person_outline_rounded,
                    label: 'Edit Profile',
                    onTap: () {
                      Navigator.pop(context);
                    },
                  ),

                  _SheetTile(
                    icon: Icons.auto_awesome_outlined,
                    label: 'Cosmic Preferences',
                    onTap: () {
                      Navigator.pop(context);
                    },
                  ),

                  const SizedBox(height: 8),

                  Divider(color: cs.outlineVariant),

                  const SizedBox(height: 8),

                  _SheetTile(
                    icon: Icons.logout_rounded,
                    label: 'Sign Out',
                    labelColor: cs.error,
                    iconColor: cs.error,
                    onTap: () async {
                      Navigator.pop(context);
                      await ref.read(authProvider.notifier).logout();
                      if (context.mounted) context.go('/login');
                    },
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    )
        .animate()
        .slideY(begin: 0.2, end: 0, duration: 300.ms, curve: Curves.easeOutQuint)
        .fadeIn(duration: 250.ms);
  }
}

class _SheetTile extends StatelessWidget {
  const _SheetTile({
    required this.icon,
    required this.label,
    required this.onTap,
    this.labelColor,
    this.iconColor,
  });

  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final Color? labelColor;
  final Color? iconColor;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return ListTile(
      leading: Icon(icon, color: iconColor ?? cs.onSurface, size: 22),
      title: Text(
        label,
        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              color: labelColor ?? cs.onSurface,
              fontWeight: FontWeight.w500,
            ),
      ),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      onTap: onTap,
      contentPadding: const EdgeInsets.symmetric(horizontal: 4),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Stats Row
// ─────────────────────────────────────────────────────────────────────────────

class _StatsRow extends StatelessWidget {
  const _StatsRow({required this.memories});
  final List<Memory> memories;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    final thisMonth = memories
        .where((m) =>
            m.date.month == DateTime.now().month &&
            m.date.year == DateTime.now().year)
        .length;
    final streak = memories.isNotEmpty ? 7 : 0;

    return Container(
      margin: const EdgeInsets.only(top: 20),
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
      decoration: BoxDecoration(
        color: cs.surfaceContainerLow,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: cs.outline.withValues(alpha: 0.4)),
      ),
      child: Row(
        children: [
          _StatItem(value: memories.length.toString(), label: 'Memories 🌌'),
          _StatDivider(),
          _StatItem(value: thisMonth.toString(), label: 'This Cycle 🪐'),
          _StatDivider(),
          _StatItem(value: '$streak 🔥', label: 'Orbit Streak'),
        ],
      ),
    )
        .animate()
        .fadeIn(delay: 100.ms, duration: 400.ms)
        .slideY(begin: 0.15, end: 0, duration: 400.ms, curve: Curves.easeOut);
  }
}

class _StatItem extends StatelessWidget {
  const _StatItem({required this.value, required this.label});
  final String value;
  final String label;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Expanded(
      child: Column(
        children: [
          Text(
            value,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w700,
                  color: cs.primary,
                ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: Theme.of(context).textTheme.labelSmall?.copyWith(
                  color: cs.onSurfaceVariant,
                ),
          ),
        ],
      ),
    );
  }
}

class _StatDivider extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      width: 1,
      height: 32,
      color: cs.outlineVariant,
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Memory Card (Cosmic Accent Palette)
// ─────────────────────────────────────────────────────────────────────────────

const _kCosmicCardAccents = [
  Color(0xFF8B5CF6), // Cosmic Purple
  Color(0xFF06B6D4), // Starlight Cyan
  Color(0xFFEC4899), // Nebula Pink
  Color(0xFF3B82F6), // Deep Galaxy Blue
  Color(0xFFF59E0B), // Supernova Gold
  Color(0xFF10B981), // Emerald Aurora
];

class _MemoryCard extends StatelessWidget {
  const _MemoryCard({required this.memory, required this.index});
  final Memory memory;
  final int index;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final accent = _kCosmicCardAccents[index % _kCosmicCardAccents.length];

    return GestureDetector(
      onTap: () {},
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        decoration: BoxDecoration(
          color: cs.surfaceContainerLow,
          borderRadius: BorderRadius.circular(22),
          border: Border.all(
            color: accent.withValues(alpha: isDark ? 0.35 : 0.25),
          ),
          boxShadow: [
            if (isDark)
              BoxShadow(
                color: accent.withValues(alpha: 0.08),
                blurRadius: 12,
                spreadRadius: 0,
              ),
          ],
        ),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: accent.withValues(alpha: isDark ? 0.2 : 0.1),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(memory.icon, color: accent, size: 22),
              ),

              const Spacer(),

              Text(
                memory.title,
                style: tt.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                  height: 1.3,
                ),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),

              const SizedBox(height: 6),

              Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: accent.withValues(alpha: isDark ? 0.18 : 0.08),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _formatDate(memory.date),
                  style: tt.labelSmall?.copyWith(
                    color: accent,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    )
        .animate()
        .fadeIn(delay: (index * 60).ms, duration: 350.ms)
        .slideY(begin: 0.12, end: 0, duration: 350.ms, curve: Curves.easeOut);
  }

  String _formatDate(DateTime d) =>
      '${d.day.toString().padLeft(2, '0')}/${d.month.toString().padLeft(2, '0')}/${d.year}';
}

// ─────────────────────────────────────────────────────────────────────────────
//  Empty state
// ─────────────────────────────────────────────────────────────────────────────

class _EmptyState extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(40),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                color: const Color(0xFF7C3AED).withValues(alpha: 0.18),
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.auto_awesome_rounded,
                  size: 38, color: Color(0xFFA78BFA)),
            ),
            const SizedBox(height: 20),
            Text(
              'Your Cosmic Journal is Empty',
              style: tt.titleMedium?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 8),
            Text(
              'Tap the + button to launch\nyour first memory into orbit.',
              textAlign: TextAlign.center,
              style: tt.bodySmall?.copyWith(
                color: cs.onSurfaceVariant,
                height: 1.6,
              ),
            ),
          ],
        ),
      ).animate().fadeIn(delay: 150.ms),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  FAB
// ─────────────────────────────────────────────────────────────────────────────

class _AddMemoryFab extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return FloatingActionButton.extended(
      onPressed: () {
        final newMemory = Memory(
          id: DateTime.now().millisecondsSinceEpoch.toString(),
          title: 'Verse #${ref.read(memoriesProvider).length + 1}',
          description: 'A newly captured cosmic memory.',
          date: DateTime.now(),
          icon: Icons.auto_awesome_rounded,
        );
        ref.read(memoriesProvider.notifier).addMemory(newMemory);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: const Text('✨ Memory launched into the verse!'),
            backgroundColor: const Color(0xFF7C3AED),
            behavior: SnackBarBehavior.floating,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          ),
        );
      },
      icon: const Icon(Icons.add_rounded),
      label: const Text('New Verse'),
    );
  }
}
