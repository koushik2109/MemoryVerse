import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

// ─────────────────────────────────────────────
//  MemoryVerse Design System – App Theme
// ─────────────────────────────────────────────

class AppColors {
  AppColors._();

  // Core brand gradient (deep night-sky violet → warm memory amber)
  static const Color primary = Color(0xFF6C3FDB);       // vivid purple
  static const Color primaryLight = Color(0xFF9B6DFF);  // soft lavender
  static const Color secondary = Color(0xFFFF7E5F);     // warm coral
  static const Color secondaryLight = Color(0xFFFFB085); // peach

  // Backgrounds
  static const Color bgDark = Color(0xFF0D0B1A);        // near-black indigo
  static const Color bgCard = Color(0xFF1A1630);        // card surface
  static const Color bgSurface = Color(0xFF231F42);     // elevated surface

  // Text
  static const Color textPrimary = Color(0xFFF5F0FF);   // off-white
  static const Color textSecondary = Color(0xFFADA6C8); // muted lavender
  static const Color textHint = Color(0xFF6B6580);

  // Accents
  static const Color gold = Color(0xFFFFD97D);          // nostalgia gold
  static const Color teal = Color(0xFF4DDFC4);          // AI teal highlight
  static const Color white = Color(0xFFFFFFFF);

  // Gradients
  static const LinearGradient brandGradient = LinearGradient(
    colors: [primary, Color(0xFF9B6DFF), secondary],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static const LinearGradient heroGradient = LinearGradient(
    colors: [
      Color(0xFF0D0B1A),
      Color(0xFF1A1630),
      Color(0xFF0D0B1A),
    ],
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
  );

  static const LinearGradient overlayGradient = LinearGradient(
    colors: [Colors.transparent, Color(0xCC0D0B1A)],
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
  );

  static const LinearGradient buttonGradient = LinearGradient(
    colors: [primary, secondary],
    begin: Alignment.centerLeft,
    end: Alignment.centerRight,
  );
}

class AppTheme {
  AppTheme._();

  static ThemeData get dark {
    final base = ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: AppColors.bgDark,
      colorScheme: ColorScheme.dark(
        primary: AppColors.primary,
        secondary: AppColors.secondary,
        surface: AppColors.bgCard,
        onPrimary: AppColors.white,
        onSurface: AppColors.textPrimary,
      ),
    );

    return base.copyWith(
      textTheme: GoogleFonts.interTextTheme(base.textTheme).copyWith(
        displayLarge: GoogleFonts.inter(
          fontSize: 36,
          fontWeight: FontWeight.w800,
          color: AppColors.textPrimary,
          letterSpacing: -0.5,
          height: 1.1,
        ),
        displayMedium: GoogleFonts.inter(
          fontSize: 28,
          fontWeight: FontWeight.w700,
          color: AppColors.textPrimary,
          letterSpacing: -0.3,
        ),
        headlineMedium: GoogleFonts.inter(
          fontSize: 22,
          fontWeight: FontWeight.w600,
          color: AppColors.textPrimary,
        ),
        bodyLarge: GoogleFonts.inter(
          fontSize: 16,
          fontWeight: FontWeight.w400,
          color: AppColors.textSecondary,
          height: 1.6,
        ),
        bodyMedium: GoogleFonts.inter(
          fontSize: 14,
          fontWeight: FontWeight.w400,
          color: AppColors.textSecondary,
          height: 1.5,
        ),
        labelLarge: GoogleFonts.inter(
          fontSize: 16,
          fontWeight: FontWeight.w600,
          color: AppColors.white,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}
