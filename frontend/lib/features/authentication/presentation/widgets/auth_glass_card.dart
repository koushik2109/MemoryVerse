import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:frontend/shared/constants/app_spacing.dart';

/// Reusable glassmorphism card used across all auth pages.
///
/// Wraps [child] in a frosted-glass container with a consistent
/// border, blur, and border radius.
class AuthGlassCard extends StatelessWidget {
  const AuthGlassCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(AppSpacing.lg),
  });

  final Widget child;
  final EdgeInsetsGeometry padding;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return ClipRRect(
      borderRadius: BorderRadius.circular(AppRadius.xxl),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
        child: Container(
          padding: padding,
          decoration: BoxDecoration(
            color: cs.surface.withValues(alpha: 0.10),
            borderRadius: BorderRadius.circular(AppRadius.xxl),
            border: Border.all(
              color: cs.onSurface.withValues(alpha: 0.14),
              width: 1.2,
            ),
          ),
          child: child,
        ),
      ),
    );
  }
}
