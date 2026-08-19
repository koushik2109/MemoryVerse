import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/widgets/buttons.dart';

class EmptyState extends StatelessWidget {
  final IconData? icon;
  final String? imageAsset;
  final String title;
  final String subtitle;
  final String? buttonText;
  final VoidCallback? onTap;
  final bool isSmall;

  const EmptyState({
    super.key,
    this.icon,
    this.imageAsset,
    required this.title,
    required this.subtitle,
    this.buttonText,
    this.onTap,
    this.isSmall = false,
  });

  @override
  Widget build(BuildContext context) {
    final c = context.colors;

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(AppRadii.xl),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
          child: Container(
            padding: EdgeInsets.all(isSmall ? AppSpacing.s16 : AppSpacing.s24),
            decoration: BoxDecoration(
              color: c.surface.withValues(alpha: context.isDark ? 0.05 : 0.4),
              borderRadius: BorderRadius.circular(AppRadii.xl),
              border: Border.all(
                color: c.border.withValues(alpha: context.isDark ? 0.1 : 0.5),
                style: BorderStyle.solid,
              ),
            ),
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (imageAsset != null)
                    Image.asset(
                      imageAsset!,
                      width: isSmall ? 80 : 120,
                      height: isSmall ? 80 : 120,
                      errorBuilder: (_, __, ___) => Icon(
                        icon ?? Icons.folder_open,
                        size: isSmall ? 24 : 32,
                        color: c.textMuted.withValues(alpha: 0.5),
                      ),
                    )
                  else if (icon != null)
                    Icon(
                      icon,
                      size: isSmall ? 24 : 32,
                      color: c.textMuted.withValues(alpha: 0.5),
                    ),
                  SizedBox(height: isSmall ? AppSpacing.s8 : AppSpacing.s16),
                  Text(
                    title,
                    style: TextStyle(
                      fontFamily: 'Inter',
                      fontSize: isSmall ? 14 : 16,
                      fontWeight: FontWeight.w600,
                      color: c.text,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    subtitle,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontFamily: 'Inter',
                      fontSize: isSmall ? 12 : 14,
                      color: c.textMuted,
                    ),
                  ),
                  if (buttonText != null && onTap != null) ...[
                    const SizedBox(height: AppSpacing.s16),
                    OutlinedButton(
                      onPressed: onTap,
                      style: OutlinedButton.styleFrom(
                        foregroundColor: c.text,
                        side: BorderSide(color: c.border),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(AppRadii.md),
                        ),
                      ),
                      child: Text(
                        buttonText!,
                        style: const TextStyle(
                          fontFamily: 'Inter',
                          fontSize: 13,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class ErrorState extends StatelessWidget {
  final String message;
  final VoidCallback? onRetry;
  final String? imageAsset;
  final Color? textColor;
  final Color? messageColor;

  const ErrorState({
    super.key,
    required this.message,
    this.onRetry,
    this.imageAsset, // Changed default to null so it uses the icon
    this.textColor,
    this.messageColor,
  });

  @override
  Widget build(BuildContext context) {
    final c = context.colors;

    return Center(
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(AppRadii.xl),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
            child: Container(
              padding: const EdgeInsets.all(AppSpacing.s32),
              decoration: BoxDecoration(
                color: c.surface.withValues(alpha: context.isDark ? 0.05 : 0.4),
                borderRadius: BorderRadius.circular(AppRadii.xl),
                border: Border.all(
                  color: c.error.withValues(alpha: 0.2), // Error tinted border
                  style: BorderStyle.solid,
                ),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (imageAsset != null)
                    Image.asset(
                      imageAsset!,
                      width: 120,
                      height: 120,
                      errorBuilder: (_, __, ___) => _buildIcon(c),
                    )
                  else
                    _buildIcon(c),
                  const SizedBox(height: AppSpacing.s24),
                  Text(
                    'An Error Occurred',
                    style: TextStyle(
                      fontFamily: 'Inter',
                      fontSize: 18,
                      fontWeight: FontWeight.w600,
                      color: textColor ?? c.text,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: AppSpacing.s8),
                  Text(
                    message,
                    style: TextStyle(
                      fontFamily: 'Inter',
                      fontSize: 14,
                      color: messageColor ?? c.textMuted,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  if (onRetry != null) ...[
                    const SizedBox(height: AppSpacing.s24),
                    OutlinedButton.icon(
                      onPressed: onRetry,
                      icon: const Icon(Icons.refresh_rounded, size: 18),
                      label: const Text('Retry'),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: c.text,
                        side: BorderSide(color: c.border.withValues(alpha: context.isDark ? 0.2 : 0.5)),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(AppRadii.md),
                        ),
                        padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24, vertical: AppSpacing.s12),
                        textStyle: const TextStyle(
                          fontFamily: 'Inter',
                          fontSize: 14,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildIcon(AppColors c) {
    return Container(
      padding: const EdgeInsets.all(AppSpacing.s16),
      decoration: BoxDecoration(
        color: c.error.withValues(alpha: 0.1),
        shape: BoxShape.circle,
      ),
      child: Icon(
        Icons.warning_amber_rounded,
        size: 32,
        color: c.error,
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
            Text(
              message!,
              style: context.text.bodySmall?.copyWith(color: c.textMuted),
            ),
          ],
        ],
      ),
    );
  }
}
