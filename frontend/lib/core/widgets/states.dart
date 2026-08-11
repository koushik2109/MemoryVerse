import 'package:flutter/material.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/widgets/buttons.dart';

class EmptyState extends StatelessWidget {
  final IconData? icon;
  final String? imageAsset;
  final String title;
  final String subtitle;
  final String? actionLabel;
  final VoidCallback? onAction;

  const EmptyState({
    super.key,
    this.icon,
    this.imageAsset,
    required this.title,
    required this.subtitle,
    this.actionLabel,
    this.onAction,
  });

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s32, vertical: AppSpacing.s64),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (imageAsset != null)
              Image.asset(imageAsset!, width: 120, height: 120)
            else if (icon != null)
              Icon(icon, size: 48, color: c.textMuted.withValues(alpha: 0.5)),
            const SizedBox(height: AppSpacing.s24),
            Text(
              title,
              style: context.text.headlineSmall?.copyWith(color: c.text),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: AppSpacing.s8),
            Text(
              subtitle,
              style: context.text.bodyMedium?.copyWith(color: c.textMuted),
              textAlign: TextAlign.center,
            ),
            if (actionLabel != null && onAction != null) ...[
              const SizedBox(height: AppSpacing.s32),
              SizedBox(
                width: 200,
                child: SecondaryButton(
                  label: actionLabel!,
                  onPressed: onAction,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class ErrorState extends StatelessWidget {
  final String message;
  final VoidCallback? onRetry;
  final String? imageAsset;

  const ErrorState({
    super.key,
    required this.message,
    this.onRetry,
    this.imageAsset = 'assets/images/error_visual.png',
  });

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(AppSpacing.s32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (imageAsset != null)
              Image.asset(imageAsset!, width: 120, height: 120)
            else
              Container(
                padding: const EdgeInsets.all(AppSpacing.s16),
                decoration: BoxDecoration(
                  color: c.error.withValues(alpha: 0.1),
                  shape: BoxShape.circle,
                ),
                child: Icon(Icons.warning_amber_rounded, size: 32, color: c.error),
              ),
            const SizedBox(height: AppSpacing.s24),
            Text(
              'An Error Occurred',
              style: context.text.headlineSmall?.copyWith(color: c.text),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: AppSpacing.s8),
            Text(
              message,
              style: context.text.bodyMedium?.copyWith(color: c.textMuted),
              textAlign: TextAlign.center,
            ),
            if (onRetry != null) ...[
              const SizedBox(height: AppSpacing.s32),
              SizedBox(
                width: 200,
                child: SecondaryButton(
                  icon: Icons.refresh_rounded,
                  label: 'Retry',
                  onPressed: onRetry,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class LoadingState extends StatelessWidget {
  final String? message;

  const LoadingState({super.key, this.message});

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          CircularProgressIndicator(color: c.primary, strokeWidth: 2),
          if (message != null) ...[
            const SizedBox(height: AppSpacing.s20),
            Text(message!, style: context.text.bodySmall?.copyWith(color: c.textMuted)),
          ],
        ],
      ),
    );
  }
}
