import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:memory_verse/core/design/tokens.dart';

abstract final class AppTheme {
  static ThemeData get light => _build(Brightness.light, AppColors.light);
  static ThemeData get dark  => _build(Brightness.dark,  AppColors.dark);

  static ThemeData _build(Brightness brightness, AppColors c) {
    // We use Inter for a stark, sophisticated, and highly legible editorial feel.
    final baseText = GoogleFonts.interTextTheme(
      brightness == Brightness.light ? ThemeData.light().textTheme : ThemeData.dark().textTheme,
    );

    final textTheme = baseText.copyWith(
      displayLarge:  baseText.displayLarge?.copyWith(fontSize: 48, fontWeight: FontWeight.w700, letterSpacing: -1.5, color: c.text, height: 1.1),
      displayMedium: baseText.displayMedium?.copyWith(fontSize: 36, fontWeight: FontWeight.w700, letterSpacing: -1.0, color: c.text, height: 1.1),
      displaySmall:  baseText.displaySmall?.copyWith(fontSize: 28, fontWeight: FontWeight.w600, letterSpacing: -0.8, color: c.text, height: 1.2),
      headlineLarge: baseText.headlineLarge?.copyWith(fontSize: 24, fontWeight: FontWeight.w600, letterSpacing: -0.5, color: c.text, height: 1.2),
      headlineMedium:baseText.headlineMedium?.copyWith(fontSize: 20, fontWeight: FontWeight.w600, letterSpacing: -0.4, color: c.text, height: 1.3),
      headlineSmall: baseText.headlineSmall?.copyWith(fontSize: 18, fontWeight: FontWeight.w600, letterSpacing: -0.3, color: c.text, height: 1.3),
      titleLarge:    baseText.titleLarge?.copyWith(fontSize: 16, fontWeight: FontWeight.w600, color: c.text, height: 1.4),
      titleMedium:   baseText.titleMedium?.copyWith(fontSize: 14, fontWeight: FontWeight.w600, color: c.text, height: 1.4),
      titleSmall:    baseText.titleSmall?.copyWith(fontSize: 13, fontWeight: FontWeight.w600, color: c.text, height: 1.4),
      bodyLarge:     baseText.bodyLarge?.copyWith(fontSize: 16, fontWeight: FontWeight.w400, color: c.text, height: 1.5, letterSpacing: 0),
      bodyMedium:    baseText.bodyMedium?.copyWith(fontSize: 14, fontWeight: FontWeight.w400, color: c.text, height: 1.5, letterSpacing: 0),
      bodySmall:     baseText.bodySmall?.copyWith(fontSize: 13, fontWeight: FontWeight.w400, color: c.textMuted, height: 1.5, letterSpacing: 0),
      labelLarge:    baseText.labelLarge?.copyWith(fontSize: 14, fontWeight: FontWeight.w500, color: c.text, letterSpacing: 0.2),
      labelMedium:   baseText.labelMedium?.copyWith(fontSize: 12, fontWeight: FontWeight.w500, color: c.textMuted, letterSpacing: 0.2),
      labelSmall:    baseText.labelSmall?.copyWith(fontSize: 11, fontWeight: FontWeight.w500, color: c.textMuted, letterSpacing: 0.2),
    );

    final colorScheme = ColorScheme(
      brightness:   brightness,
      primary:      c.primary,
      onPrimary:    c.primaryInverse,
      secondary:    c.surfaceElevated,
      onSecondary:  c.text,
      error:        c.error,
      onError:      c.primaryInverse,
      surface:      c.bg,
      onSurface:    c.text,
      outline:      c.border,
      outlineVariant: c.borderSubtle,
    );

    return ThemeData(
      useMaterial3: true,
      brightness: brightness,
      colorScheme: colorScheme,
      scaffoldBackgroundColor: c.bg,
      textTheme: textTheme,
      extensions: <ThemeExtension<dynamic>>[c],

      appBarTheme: AppBarTheme(
        backgroundColor: c.bg,
        foregroundColor: c.text,
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: false,
        titleTextStyle: textTheme.headlineSmall,
        iconTheme: IconThemeData(color: c.text, size: 24),
      ),

      cardTheme: CardThemeData(
        color: c.surface,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(AppRadii.lg),
          side: BorderSide(color: c.borderSubtle, width: 1),
        ),
      ),

      dividerTheme: DividerThemeData(
        color: c.borderSubtle,
        thickness: 1,
        space: 1,
      ),
      
      bottomNavigationBarTheme: BottomNavigationBarThemeData(
        backgroundColor: c.bg,
        selectedItemColor: c.primary,
        unselectedItemColor: c.textMuted,
        type: BottomNavigationBarType.fixed,
        elevation: 0,
        selectedLabelStyle: textTheme.labelSmall?.copyWith(fontWeight: FontWeight.w600),
        unselectedLabelStyle: textTheme.labelSmall,
      ),
    );
  }
}
