import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/navigation/router.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/providers/auth_provider.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/core/widgets/states.dart';
import 'package:memory_verse/core/widgets/media.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;
class ProfileScreen extends ConsumerWidget {
  const ProfileScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final c = context.colors;
    final profileAsync = ref.watch(userProfileProvider);

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s20),
          child: Column(
            children: [
              const SizedBox(height: AppSpacing.s20),

              // ── Profile Header ────────────────────────
              profileAsync.when(
                loading: () => const LoadingState(),
                error: (err, stack) {
                  debugPrint('Failed to load profileProvider: $err\n$stack');
                  return ErrorState(
                    message: 'Failed to load profile',
                    onRetry: () => ref.invalidate(userProfileProvider),
                  );
                },
                data: (user) => Column(
                  children: [
                    Avatar(url: user.avatarUrl, name: user.fullName, size: 80),
                    const SizedBox(height: AppSpacing.s16),
                    Text(
                      user.fullName ?? 'User',
                      style: Theme.of(context).textTheme.headlineMedium
                          ?.copyWith(
                            color: c.text,
                            fontWeight: FontWeight.w700,
                          ),
                    ),
                    if (user.username != null && user.username!.isNotEmpty) ...[
                      const SizedBox(height: AppSpacing.s4),
                      Text(
                        '@${user.username}',
                        style: Theme.of(context).textTheme.titleMedium
                            ?.copyWith(color: c.primary),
                      ),
                    ],
                    const SizedBox(height: AppSpacing.s4),
                    Text(
                      user.email,
                      style: Theme.of(context).textTheme.bodySmall
                          ?.copyWith(color: c.textMuted),
                    ),
                    if (user.bio != null && user.bio!.isNotEmpty) ...[
                      const SizedBox(height: AppSpacing.s8),
                      Text(
                        user.bio!,
                        textAlign: TextAlign.center,
                        style: Theme.of(context).textTheme.bodySmall
                            ?.copyWith(color: c.textMuted),
                      ),
                    ],

                    const SizedBox(height: AppSpacing.s24),

                    // Stats
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        _StatItem(
                          label: 'Memories',
                          value: '${user.mediaCount}',
                        ),
                        Container(
                          width: 1,
                          height: 32,
                          color: c.border,
                          margin: const EdgeInsets.symmetric(
                            horizontal: AppSpacing.s24,
                          ),
                        ),
                        _StatItem(label: 'Vaults', value: '${user.vaultCount}'),
                      ],
                    ),
                  ],
                ),
              ),

              const SizedBox(height: AppSpacing.s32),

              // ── Settings List ─────────────────────────
              _SettingsSection(
                children: [
                  _SettingsTile(
                    icon: Icons.person_outline_rounded,
                    title: 'Edit Profile',
                    onTap: () {
                      showDialog(
                        context: context,
                        builder: (ctx) => _EditProfileDialog(
                          currentName: profileAsync.value?.fullName ?? '',
                        ),
                      );
                    },
                  ),
                  _SettingsTile(
                    icon: Icons.notifications_outlined,
                    title: 'Notifications',
                    onTap: () => context.push(Routes.notifications),
                  ),
                  _SettingsTile(
                    icon: Icons.settings_outlined,
                    title: 'Settings',
                    onTap: () => context.push(Routes.settings),
                  ),
                ],
              ),

              const SizedBox(height: AppSpacing.s16),

              _SettingsSection(
                children: [
                  _SettingsTile(
                    icon: Icons.help_outline_rounded,
                    title: 'Help & Support',
                    onTap: () {
                      showDialog(
                        context: context,
                        builder: (ctx) => AlertDialog(
                          backgroundColor: ctx.colors.surface,
                          title: const Text('Help & Support'),
                          content: const Text(
                            'For assistance, please email memoryversekmit@gmail.com',
                          ),
                          actions: [
                            TextButton(
                              onPressed: () => Navigator.pop(ctx),
                              child: const Text('Close'),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                  _SettingsTile(
                    icon: Icons.info_outline_rounded,
                    title: 'About MemoryVerse',
                    onTap: () {
                      showDialog(
                        context: context,
                        builder: (ctx) => AlertDialog(
                          backgroundColor: ctx.colors.surface,
                          title: Row(
                            children: [
                              Container(
                                width: 32,
                                height: 32,
                                decoration: BoxDecoration(
                                  color: ctx.colors.primary,
                                  borderRadius: BorderRadius.circular(AppRadii.sm),
                                ),
                                child: const Icon(
                                  Icons.auto_awesome_rounded,
                                  color: adt.AppColors.onDarkPrimary,
                                  size: 16,
                                ),
                              ),
                              const SizedBox(width: AppSpacing.s12),
                              const Text('MemoryVerse'),
                            ],
                          ),
                          content: const Text(
                            'Version 1.0.0\n\n© 2026 Koushik. All rights reserved.',
                          ),
                          actions: [
                            TextButton(
                              onPressed: () => Navigator.pop(ctx),
                              child: const Text('Close'),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ],
              ),

              const SizedBox(height: AppSpacing.s16),

              _SettingsSection(
                children: [
                  _SettingsTile(
                    icon: Icons.logout_rounded,
                    title: 'Sign Out',
                    isDestructive: true,
                    onTap: () async {
                      await ref.read(authNotifierProvider.notifier).signOut();
                      if (context.mounted) context.go(Routes.signIn);
                    },
                  ),
                ],
              ),

              const SizedBox(height: AppSpacing.s48),

              // Version
              Text(
                'MemoryVerse v1.0.0',
                style: Theme.of(context).textTheme.labelSmall
                    ?.copyWith(color: c.textMuted),
              ),

              const SizedBox(height: AppSpacing.s32),
            ],
          ),
        ),
      ),
    );
  }
}

class _StatItem extends StatelessWidget {
  const _StatItem({required this.label, required this.value});
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          value,
          style: Theme.of(context).textTheme.headlineMedium
              ?.copyWith(fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 2),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall
              ?.copyWith(color: context.colors.textMuted),
        ),
      ],
    );
  }
}

class _SettingsSection extends StatelessWidget {
  const _SettingsSection({required this.children});
  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: children,
    );
  }
}

class _SettingsTile extends StatelessWidget {
  const _SettingsTile({
    required this.icon,
    required this.title,
    required this.onTap,
    this.isDestructive = false,
  }) : trailing = null;

  final IconData icon;
  final String title;
  final VoidCallback onTap;
  final Widget? trailing;
  final bool isDestructive;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final color = isDestructive ? c.error : c.text;

    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(AppRadii.md),
      child: Padding(
        padding: const EdgeInsets.symmetric(
          horizontal: AppSpacing.s16,
          vertical: AppSpacing.s14,
        ),
        child: Row(
          children: [
            Icon(icon, size: 22, color: isDestructive ? c.error : c.textMuted),
            const SizedBox(width: AppSpacing.s12),
            Expanded(
              child: Text(
                title,
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w500,
                  color: color,
                ),
              ),
            ),
            if (trailing != null) trailing!,
            if (!isDestructive)
              Icon(Icons.chevron_right_rounded, size: 20, color: c.textMuted),
          ],
        ),
      ),
    );
  }
}

class _EditProfileDialog extends ConsumerStatefulWidget {
  const _EditProfileDialog({required this.currentName});
  final String currentName;

  @override
  ConsumerState<_EditProfileDialog> createState() => _EditProfileDialogState();
}

class _EditProfileDialogState extends ConsumerState<_EditProfileDialog> {
  late final TextEditingController _nameController;
  bool _isSaving = false;

  @override
  void initState() {
    super.initState();
    _nameController = TextEditingController(text: widget.currentName);
  }

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final newName = _nameController.text.trim();
    if (newName.isEmpty) return;

    setState(() => _isSaving = true);
    try {
      final repo = ref.read(profileRepositoryProvider);
      await repo.updateProfile(fullName: newName);
      ref.invalidate(userProfileProvider);
      if (mounted) Navigator.pop(context);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to update: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isSaving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return AlertDialog(
      backgroundColor: c.surface,
      title: const Text('Edit Profile'),
      content: TextField(
        controller: _nameController,
        decoration: InputDecoration(
          labelText: 'Display Name',
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(AppRadii.md),
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: _isSaving ? null : () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed: _isSaving ? null : _save,
          style: FilledButton.styleFrom(backgroundColor: c.primary),
          child: _isSaving
              ? const SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: adt.AppColors.onDarkPrimary,
                  ),
                )
              : const Text('Save'),
        ),
      ],
    );
  }
}
