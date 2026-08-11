import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/theme_provider.dart';

class SettingsScreen extends ConsumerWidget {
  const SettingsScreen({super.key});


  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final c = context.colors;

    return Scaffold(
      backgroundColor: c.bg,
      appBar: AppBar(
        backgroundColor: c.bg,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded),
          onPressed: () => context.pop(),
        ),
        title: const Text('Settings'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(AppSpacing.s20),
        children: [
          _Section(
            title: 'General',
            children: [
              _Tile(
                icon: Icons.palette_outlined,
                title: 'Theme',
                subtitle: context.isDark ? 'Dark mode' : 'Light mode',
                onTap: () => ref.read(themeModeProvider.notifier).toggle(),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _Section extends StatelessWidget {
  const _Section({required this.title, required this.children});
  final String title;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: Theme.of(context).textTheme.titleSmall?.copyWith(color: c.textMuted)),
        const SizedBox(height: AppSpacing.s10),
        Container(
          decoration: BoxDecoration(
            color: c.surface,
            borderRadius: BorderRadius.circular(AppRadii.md),
            border: Border.all(color: c.border),
          ),
          child: Column(
            children: List.generate(children.length * 2 - 1, (i) {
              if (i.isOdd) return Divider(height: 0, color: c.border, indent: AppSpacing.s48);
              return children[i ~/ 2];
            }),
          ),
        ),
      ],
    );
  }
}

class _Tile extends StatelessWidget {
  const _Tile({required this.icon, required this.title, required this.subtitle, required this.onTap});
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(AppRadii.md),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s16, vertical: AppSpacing.s14),
        child: Row(
          children: [
            Icon(icon, size: 22, color: c.textMuted),
            const SizedBox(width: AppSpacing.s12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title, style: Theme.of(context).textTheme.bodyMedium),
                  Text(subtitle, style: Theme.of(context).textTheme.bodySmall?.copyWith(color: c.textMuted)),
                ],
              ),
            ),
            Icon(Icons.chevron_right_rounded, size: 20, color: c.textMuted),
          ],
        ),
      ),
    );
  }
}
