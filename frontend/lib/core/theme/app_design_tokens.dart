import 'package:flutter/material.dart';

/// Brand ramp — one hue family, 50 (lightest) to 900 (darkest).
class AppColors {
  AppColors._();

  static const plum50 = Color(0xFFFBEFF6);
  static const plum100 = Color(0xFFF2D9EA);
  static const plum200 = Color(0xFFE2B4D6);
  static const plum300 = Color(0xFFC97FB8);
  static const plum400 = Color(0xFFA34F95);
  static const plum500 = Color(0xFF7A3A6E);
  static const plum600 = Color(0xFF5C2F5A);
  static const plum700 = Color(0xFF442345);
  static const plum800 = Color(0xFF2E1A30);
  static const plum900 = Color(0xFF1A0F1E);

  static const accent = Color(0xFFF2B84B);
  static const accentInk = Color(0xFF442303);

  static const onDarkPrimary = Color(0xF5FFFFFF); // 96%
  static const onDarkSecondary = Color(0xCCFFFFFF); // 80%
  static const onDarkMuted = Color(0x99FFFFFF); // 60%

  static const onLightPrimary = Color(0xFF241A33);
  static const onLightSecondary = Color(0xFF5B4A6D);
  static const onLightMuted = Color(0xFF8A7A9C);
}

class AppGradients {
  AppGradients._();

  static const dark = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [AppColors.plum700, AppColors.plum500, AppColors.plum900],
  );

  static const light = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [AppColors.plum50, AppColors.plum100, AppColors.plum200],
  );

  static const scrim = LinearGradient(
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
    colors: [
      Color(0x1A000000), // 10%
      Color(0x00000000), // transparent
      Color(0x00000000), // transparent
      Color(0x8C000000), // 55%
    ],
    stops: [0.0, 0.3, 0.55, 1.0],
  );
}

class AppSpacing {
  AppSpacing._();

  static const s4 = 4.0;
  static const s8 = 8.0;
  static const s12 = 12.0;
  static const s16 = 16.0;
  static const s20 = 20.0;
  static const s24 = 24.0;
  static const s28 = 28.0;
  static const s32 = 32.0;
  static const s40 = 40.0;
  static const s48 = 48.0;
}

class AppRadius {
  AppRadius._();

  static const sm = 8.0;
  static const md = 12.0;
  static const lg = 16.0;
  static const pill = 999.0;
}

class AppTextStyles {
  AppTextStyles._();

  static const _family = 'Inter';

  static const display = TextStyle(
    fontFamily: _family,
    fontSize: 32,
    height: 38 / 32,
    fontWeight: FontWeight.w500,
  );
  static const h1 = TextStyle(
    fontFamily: _family,
    fontSize: 24,
    height: 30 / 24,
    fontWeight: FontWeight.w500,
  );
  static const h2 = TextStyle(
    fontFamily: _family,
    fontSize: 18,
    height: 24 / 18,
    fontWeight: FontWeight.w500,
  );
  static const body = TextStyle(
    fontFamily: _family,
    fontSize: 14,
    height: 22 / 14,
    fontWeight: FontWeight.w400,
  );
  static const caption = TextStyle(
    fontFamily: _family,
    fontSize: 12,
    height: 16 / 12,
    fontWeight: FontWeight.w400,
  );
  static const micro = TextStyle(
    fontFamily: _family,
    fontSize: 11,
    height: 14 / 11,
    fontWeight: FontWeight.w400,
  );
}

class AppTheme {
  AppTheme._();

  static final primaryButtonOnDark = ElevatedButton.styleFrom(
    backgroundColor: Colors.white,
    foregroundColor: AppColors.plum800,
    elevation: 0,
    minimumSize: const Size(0, 44),
    padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(AppRadius.md),
    ),
    textStyle: AppTextStyles.body.copyWith(fontWeight: FontWeight.w500),
  );

  static final primaryButtonOnLight = ElevatedButton.styleFrom(
    backgroundColor: AppColors.plum800,
    foregroundColor: Colors.white,
    elevation: 0,
    minimumSize: const Size(0, 44),
    padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(AppRadius.md),
    ),
    textStyle: AppTextStyles.body.copyWith(fontWeight: FontWeight.w500),
  );

  static final secondaryButtonOnDark = OutlinedButton.styleFrom(
    backgroundColor: Colors.transparent,
    foregroundColor: Colors.white,
    side: const BorderSide(color: Color(0x66FFFFFF), width: 0.5),
    minimumSize: const Size(0, 44),
    padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(AppRadius.md),
    ),
    textStyle: AppTextStyles.body.copyWith(fontWeight: FontWeight.w500),
  );

  static final secondaryButtonOnLight = OutlinedButton.styleFrom(
    backgroundColor: Colors.transparent,
    foregroundColor: AppColors.plum800,
    side: BorderSide(color: AppColors.plum800.withOpacity(0.25), width: 0.5),
    minimumSize: const Size(0, 44),
    padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(AppRadius.md),
    ),
    textStyle: AppTextStyles.body.copyWith(fontWeight: FontWeight.w500),
  );

  static ButtonStyle iconButtonStyle({required bool onDark}) {
    return IconButton.styleFrom(
      backgroundColor: onDark
          ? const Color(0x26FFFFFF)
          : AppColors.plum800.withOpacity(0.08),
      foregroundColor: onDark ? Colors.white : AppColors.plum800,
      fixedSize: const Size(36, 36),
      shape: const CircleBorder(),
    );
  }

  static ThemeData get themeData {
    return ThemeData(
      useMaterial3: true,
      fontFamily: 'Inter',
      colorScheme: ColorScheme.fromSeed(
        seedColor: AppColors.plum500,
        primary: AppColors.plum800,
        secondary: AppColors.accent,
      ),
      textTheme: const TextTheme(
        displayLarge: AppTextStyles.display,
        headlineLarge: AppTextStyles.h1,
        headlineMedium: AppTextStyles.h2,
        bodyMedium: AppTextStyles.body,
        bodySmall: AppTextStyles.caption,
        labelSmall: AppTextStyles.micro,
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(style: primaryButtonOnLight),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: secondaryButtonOnLight,
      ),
    );
  }
}
