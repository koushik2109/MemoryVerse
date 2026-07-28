import 'package:flutter/material.dart';

/// Semantic color token extensions on [ColorScheme].
///
/// Prefer these aliases over raw [ColorScheme] properties to ensure
/// tokens remain consistent and easy to remap.
extension AppColorTokens on ColorScheme {
  /// Subtle transparent surface suitable for glassmorphism overlays.
  Color get glassSurface => surface.withValues(alpha: 0.12);

  /// Glass card border color.
  Color get glassBorder => onSurface.withValues(alpha: 0.18);

  /// Subtle glass highlight (top-left shimmer).
  Color get glassHighlight => onSurface.withValues(alpha: 0.08);

  /// Input field background color.
  Color get inputFill => surfaceContainerHighest.withValues(alpha: 0.6);

  /// Muted label / hint text color.
  Color get subtleText => onSurface.withValues(alpha: 0.5);

  /// Low-emphasis divider color.
  Color get divider => outlineVariant;

  /// Danger / destructive action color.
  Color get danger => error;

  /// Password strength: weak.
  Color get strengthWeak => error;

  /// Password strength: medium.
  Color get strengthMedium => tertiary;

  /// Password strength: strong.
  Color get strengthStrong => Color.lerp(primary, Colors.green, 0.5) ?? primary;
}
