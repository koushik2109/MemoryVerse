import 'package:flutter/material.dart';

/// Convenience extensions on [BuildContext] to reduce boilerplate.
extension BuildContextX on BuildContext {
  // ── Theme ──────────────────────────────────────────────────────────────────
  ThemeData get theme => Theme.of(this);
  ColorScheme get colorScheme => Theme.of(this).colorScheme;
  TextTheme get textTheme => Theme.of(this).textTheme;

  // ── Brightness ─────────────────────────────────────────────────────────────
  bool get isDarkMode => Theme.of(this).brightness == Brightness.dark;

  // ── Screen dimensions ─────────────────────────────────────────────────────
  Size get screenSize => MediaQuery.sizeOf(this);
  double get screenWidth => MediaQuery.sizeOf(this).width;
  double get screenHeight => MediaQuery.sizeOf(this).height;

  // ── Safe area ─────────────────────────────────────────────────────────────
  EdgeInsets get viewPadding => MediaQuery.viewPaddingOf(this);
  EdgeInsets get viewInsets => MediaQuery.viewInsetsOf(this);
}
