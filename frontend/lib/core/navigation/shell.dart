import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/features/memories/presentation/create_memory_sheet.dart';
import 'package:memory_verse/features/memories/presentation/upload_progress_banner.dart';

import 'package:memory_verse/core/presentation/widgets/pixelated_mesh_background.dart';

class AppShell extends StatelessWidget {
  const AppShell({super.key, required this.shell});
  final StatefulNavigationShell shell;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          const PixelatedMeshBackground(),
          Column(
            children: [
              Expanded(child: shell),
              const UploadProgressBanner(),
            ],
          ),
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
    if (index == 2) {
      CreateMemorySheet.show(context);
      return;
    }
    int branchIndex = index;
    if (index > 2) {
      branchIndex = index - 1;
    }
    if (shell.currentIndex == branchIndex) return;
    shell.goBranch(
      branchIndex,
      initialLocation: branchIndex == shell.currentIndex,
    );
  }

  @override
  Widget build(BuildContext context) {
    final current = shell.currentIndex;

    return ClipRect(
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
        child: Container(
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.2),
            border: Border(
              top: BorderSide(color: Colors.white.withOpacity(0.1), width: 0.5),
            ),
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
                              color: Colors.white.withOpacity(0.15),
                              borderRadius: BorderRadius.circular(AppRadii.md),
                              border: Border.all(
                                color: Colors.white.withOpacity(0.3),
                              ),
                            ),
                            child: const Icon(
                              Icons.add_rounded,
                              color: Colors.white,
                              size: 26,
                            ),
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
              color: active ? Colors.white : Colors.white.withOpacity(0.5),
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                fontFamily: 'Inter',
                fontSize: 11,
                fontWeight: active ? FontWeight.w600 : FontWeight.w400,
                color: active ? Colors.white : Colors.white.withOpacity(0.5),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
