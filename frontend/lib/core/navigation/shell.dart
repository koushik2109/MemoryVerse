import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/features/memories/presentation/create_memory_sheet.dart';
import 'package:memory_verse/features/memories/presentation/upload_progress_banner.dart';

class AppShell extends StatelessWidget {
  const AppShell({super.key, required this.shell});
  final StatefulNavigationShell shell;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: context.colors.bg,
      body: Column(
        children: [
          Expanded(child: shell),
          const UploadProgressBanner(),
        ],
      ),
      bottomNavigationBar: _MvBottomBar(shell: shell),
    );
  }
}

// ─────────────────────────────────────────────────────────

class _MvBottomBar extends StatelessWidget {
  const _MvBottomBar({required this.shell});
  final StatefulNavigationShell shell;

  void _onTap(BuildContext context, int index) {
    HapticFeedback.selectionClick();
    // Index 2 is the center "Create" action — open create memory sheet
    if (index == 2) {
      CreateMemorySheet.show(context);
      return;
    }
    
    // Map bottom bar index (0..4) to shell branch index (0..3)
    int branchIndex = index;
    if (index > 2) {
      branchIndex = index - 1;
    }

    if (shell.currentIndex == branchIndex) return;
    shell.goBranch(branchIndex, initialLocation: branchIndex == shell.currentIndex);
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final current = shell.currentIndex;

    return Container(
      decoration: BoxDecoration(
        color: c.surface,
        border: Border(top: BorderSide(color: c.border, width: 0.5)),
      ),
      child: SafeArea(
        top: false,
        child: SizedBox(
          height: 60,
          child: Row(
            children: [
              _NavItem(
                icon: Icons.home_outlined,
                activeIcon: Icons.home_rounded,
                label: 'Home',
                active: current == 0,
                onTap: () => _onTap(context, 0),
              ),
              _NavItem(
                icon: Icons.timeline_outlined,
                activeIcon: Icons.timeline_rounded,
                label: 'Timeline',
                active: current == 1,
                onTap: () => _onTap(context, 1),
              ),
              // Center Create button
              Expanded(
                child: GestureDetector(
                  onTap: () => _onTap(context, 2),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        width: 44,
                        height: 44,
                        decoration: BoxDecoration(
                          color: c.primary,
                          borderRadius: BorderRadius.circular(AppRadii.md),
                          boxShadow: AppShadows.elevated(c.primary),
                        ),
                        child: const Icon(Icons.add_rounded, color: Colors.white, size: 26),
                      ),
                    ],
                  ),
                ),
              ),
              _NavItem(
                icon: Icons.photo_library_outlined,
                activeIcon: Icons.photo_library_rounded,
                label: 'Memories',
                active: current == 2,
                onTap: () => _onTap(context, 3),
              ),
              _NavItem(
                icon: Icons.person_outlined,
                activeIcon: Icons.person_rounded,
                label: 'Profile',
                active: current == 3,
                onTap: () => _onTap(context, 4),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _NavItem extends StatelessWidget {
  const _NavItem({
    required this.icon,
    required this.activeIcon,
    required this.label,
    required this.active,
    required this.onTap,
  });

  final IconData icon;
  final IconData activeIcon;
  final String label;
  final bool active;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        behavior: HitTestBehavior.opaque,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              active ? activeIcon : icon,
              size: 22,
              color: active ? c.primary : c.textMuted,
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                fontSize: 11,
                fontWeight: active ? FontWeight.w600 : FontWeight.w400,
                color: active ? c.primary : c.textMuted,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
