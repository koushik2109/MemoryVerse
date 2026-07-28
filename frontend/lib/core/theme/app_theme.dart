import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// Cosmic Space Theme — Deep Void, Nebula Indigo & Starlight Cyan
const _kBrandCosmicPurple = Color(0xFF7C3AED); // Electric Violet Nebula
const _kStardustCyan = Color(0xFF06B6D4);     // Cyan Starlight Accent
const _kSupernovaAmber = Color(0xFFF59E0B);    // Supernova Gold Accent

class AppTheme {
  AppTheme._();

  static ThemeData get lightTheme => _build(Brightness.light);
  static ThemeData get darkTheme => _build(Brightness.dark);

  static ThemeData _build(Brightness brightness) {
    final isDark = brightness == Brightness.dark;
    final base = isDark ? ThemeData.dark() : ThemeData.light();

    final colorScheme = ColorScheme.fromSeed(
      seedColor: _kBrandCosmicPurple,
      brightness: brightness,
      primary: isDark ? const Color(0xFFA78BFA) : const Color(0xFF6D28D9),
      onPrimary: isDark ? const Color(0xFF1E1B4B) : Colors.white,
      primaryContainer:
          isDark ? const Color(0xFF2E1065) : const Color(0xFFEDE9FE),
      onPrimaryContainer:
          isDark ? const Color(0xFFDDD6FE) : const Color(0xFF371B58),
      secondary: isDark ? _kStardustCyan : const Color(0xFF0284C7),
      onSecondary: Colors.white,
      secondaryContainer:
          isDark ? const Color(0xFF083344) : const Color(0xFFE0F2FE),
      onSecondaryContainer:
          isDark ? const Color(0xFFA5F3FC) : const Color(0xFF075985),
      tertiary: _kSupernovaAmber,
      surface: isDark ? const Color(0xFF050814) : const Color(0xFFF1F5F9),
      surfaceContainerLow:
          isDark ? const Color(0xFF0D1527) : const Color(0xFFFFFFFF),
      surfaceContainerHighest:
          isDark ? const Color(0xFF19253E) : const Color(0xFFE2E8F0),
      onSurface: isDark ? const Color(0xFFF1F5F9) : const Color(0xFF0F172A),
      onSurfaceVariant:
          isDark ? const Color(0xFF94A3B8) : const Color(0xFF475569),
      outline:
          isDark ? const Color(0xFF2A3958) : const Color(0xFFCBD5E1),
      outlineVariant:
          isDark ? const Color(0xFF1E293B) : const Color(0xFFE2E8F0),
      error: const Color(0xFFF43F5E),
      onError: Colors.white,
    );

    return base.copyWith(
      colorScheme: colorScheme,
      scaffoldBackgroundColor: colorScheme.surface,
      textTheme: GoogleFonts.outfitTextTheme(base.textTheme),

      appBarTheme: AppBarTheme(
        centerTitle: false,
        elevation: 0,
        scrolledUnderElevation: 0,
        backgroundColor: Colors.transparent,
        foregroundColor: colorScheme.onSurface,
      ),

      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
        contentPadding: const EdgeInsets.symmetric(
          horizontal: 16,
          vertical: 14,
        ),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(
            color: colorScheme.outline.withValues(alpha: 0.4),
          ),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: colorScheme.primary, width: 2),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: colorScheme.error),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: colorScheme.error, width: 2),
        ),
        labelStyle: TextStyle(color: colorScheme.onSurfaceVariant),
        floatingLabelStyle: TextStyle(color: colorScheme.primary),
      ),

      cardTheme: CardThemeData(
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(22),
          side: BorderSide(
            color: colorScheme.outline.withValues(alpha: 0.4),
          ),
        ),
        color: colorScheme.surfaceContainerLow,
      ),

      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: colorScheme.primary,
          foregroundColor: colorScheme.onPrimary,
          minimumSize: const Size(double.infinity, 52),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          textStyle: GoogleFonts.outfit(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.3,
          ),
          elevation: 4,
          shadowColor: colorScheme.primary.withValues(alpha: 0.4),
        ),
      ),

      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: colorScheme.onSurface,
          minimumSize: const Size(double.infinity, 52),
          side: BorderSide(color: colorScheme.outline),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          textStyle: GoogleFonts.outfit(
            fontSize: 15,
            fontWeight: FontWeight.w500,
          ),
        ),
      ),

      floatingActionButtonTheme: FloatingActionButtonThemeData(
        backgroundColor: colorScheme.primary,
        foregroundColor: colorScheme.onPrimary,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        elevation: 6,
      ),

      chipTheme: ChipThemeData(
        backgroundColor: colorScheme.surfaceContainerHighest,
        selectedColor: colorScheme.primaryContainer,
        labelStyle: GoogleFonts.outfit(fontSize: 13),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        side: BorderSide.none,
      ),

      dividerTheme: DividerThemeData(
        color: colorScheme.outlineVariant,
        thickness: 1,
      ),
    );
  }
}
