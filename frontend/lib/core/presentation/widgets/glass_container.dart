import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;

class GlassContainer extends StatelessWidget {
  final Widget child;
  final double borderRadius;
  final EdgeInsetsGeometry padding;
  final Color tint;
  final double blurSigma;

  const GlassContainer({
    super.key,
    required this.child,
    this.borderRadius = AppRadius.lg,
    this.padding = const EdgeInsets.all(AppSpacing.s16),
    this.tint = adt.AppColors.onDarkPrimary,
    this.blurSigma = 8,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(borderRadius),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: blurSigma, sigmaY: blurSigma),
        child: Container(
          padding: padding,
          decoration: BoxDecoration(
            color: tint.withValues(alpha: 0.55),
            borderRadius: BorderRadius.circular(borderRadius),
            border: Border.all(
              color: AppColors.plum800.withValues(alpha: 0.15),
              width: 0.5,
            ),
          ),
          child: child,
        ),
      ),
    );
  }
}
