import 'package:flutter/material.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/widgets/buttons.dart';

class AppDialog extends StatelessWidget {
  final String title;
  final String content;
  final String primaryLabel;
  final VoidCallback onPrimary;
  final String? secondaryLabel;
  final VoidCallback? onSecondary;
  final bool isDestructive;

  const AppDialog({
    super.key,
    required this.title,
    required this.content,
    required this.primaryLabel,
    required this.onPrimary,
    this.secondaryLabel,
    this.onSecondary,
    this.isDestructive = false,
  });

  static Future<T?> show<T>({
    required BuildContext context,
    required String title,
    required String content,
    required String primaryLabel,
    required VoidCallback onPrimary,
    String? secondaryLabel,
    VoidCallback? onSecondary,
    bool isDestructive = false,
  }) {
    return showDialog<T>(
      context: context,
      builder: (context) => AppDialog(
        title: title,
        content: content,
        primaryLabel: primaryLabel,
        onPrimary: onPrimary,
        secondaryLabel: secondaryLabel,
        onSecondary: onSecondary,
        isDestructive: isDestructive,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Dialog(
      backgroundColor: c.surface,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppRadii.md),
        side: BorderSide(color: c.borderSubtle),
      ),
      elevation: 0,
      child: Padding(
        padding: const EdgeInsets.all(AppSpacing.s24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: context.text.headlineSmall),
            const SizedBox(height: AppSpacing.s12),
            Text(content, style: context.text.bodyMedium?.copyWith(color: c.textMuted)),
            const SizedBox(height: AppSpacing.s32),
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                if (secondaryLabel != null) ...[
                  Expanded(
                    child: SecondaryButton(
                      label: secondaryLabel!,
                      onPressed: onSecondary ?? () => Navigator.of(context).pop(),
                    ),
                  ),
                  const SizedBox(width: AppSpacing.s12),
                ],
                Expanded(
                  child: ElevatedButton(
                    onPressed: onPrimary,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: isDestructive ? c.error : c.primary,
                      foregroundColor: c.primaryInverse,
                      elevation: 0,
                      padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24, vertical: AppSpacing.s16),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(AppRadii.sm)),
                      minimumSize: const Size(0, 56),
                    ),
                    child: Text(primaryLabel, style: context.text.labelLarge?.copyWith(color: c.primaryInverse, fontWeight: FontWeight.w600)),
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

class BottomSheetContainer extends StatelessWidget {
  final Widget child;

  const BottomSheetContainer({super.key, required this.child});

  static Future<T?> show<T>(BuildContext context, Widget child) {
    return showModalBottomSheet<T>(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (context) => BottomSheetContainer(child: child),
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Container(
      margin: const EdgeInsets.only(top: AppSpacing.s64),
      decoration: BoxDecoration(
        color: c.bg,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(AppRadii.xl)),
        border: Border.all(color: c.borderSubtle),
      ),
      child: SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(height: AppSpacing.s12),
            Container(
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: c.border,
                borderRadius: BorderRadius.circular(AppRadii.pill),
              ),
            ),
            const SizedBox(height: AppSpacing.s24),
            Flexible(child: child),
          ],
        ),
      ),
    );
  }
}
