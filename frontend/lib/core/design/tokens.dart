import 'package:flutter/material.dart';

@immutable
class AppColors extends ThemeExtension<AppColors> {
  const AppColors({
    required this.bg,
    required this.surface,
    required this.surfaceElevated,
    required this.border,
    required this.borderSubtle,
    required this.text,
    required this.textMuted,
    required this.textInverse,
    required this.primary,
    required this.primaryInverse,
    required this.error,
    required this.success,
  });

  final Color bg;
  final Color surface;
  final Color surfaceElevated;
  final Color border;
  final Color borderSubtle;
  final Color text;
  final Color textMuted;
  final Color textInverse;
  final Color primary;
  final Color primaryInverse;
  final Color error;
  final Color success;

  static const light = AppColors(
    bg: Color(0xFFFFFFFF),
    surface: Color(0xFFF7F7F7),
    surfaceElevated: Color(0xFFFFFFFF),
    border: Color(0xFFE5E5E5),
    borderSubtle: Color(0xFFF0F0F0),
    text: Color(0xFF000000),
    textMuted: Color(0xFF737373),
    textInverse: Color(0xFFFFFFFF),
    primary: Color(0xFF000000),
    primaryInverse: Color(0xFFFFFFFF),
    error: Color(0xFFE5484D),
    success: Color(0xFF30A46C),
  );

  static const dark = AppColors(
    bg: Color(0xFF000000),
    surface: Color(0xFF111111),
    surfaceElevated: Color(0xFF1C1C1C),
    border: Color(0xFF282828),
    borderSubtle: Color(0xFF1C1C1C),
    text: Color(0xFFFFFFFF),
    textMuted: Color(0xFFA0A0A0),
    textInverse: Color(0xFF000000),
    primary: Color(0xFFFFFFFF),
    primaryInverse: Color(0xFF000000),
    error: Color(0xFFE5484D),
    success: Color(0xFF30A46C),
  );

  @override
  AppColors copyWith({
    Color? bg,
    Color? surface,
    Color? surfaceElevated,
    Color? border,
    Color? borderSubtle,
    Color? text,
    Color? textMuted,
    Color? textInverse,
    Color? primary,
    Color? primaryInverse,
    Color? error,
    Color? success,
  }) => AppColors(
    bg: bg ?? this.bg,
    surface: surface ?? this.surface,
    surfaceElevated: surfaceElevated ?? this.surfaceElevated,
    border: border ?? this.border,
    borderSubtle: borderSubtle ?? this.borderSubtle,
    text: text ?? this.text,
    textMuted: textMuted ?? this.textMuted,
    textInverse: textInverse ?? this.textInverse,
    primary: primary ?? this.primary,
    primaryInverse: primaryInverse ?? this.primaryInverse,
    error: error ?? this.error,
    success: success ?? this.success,
  );

  @override
  AppColors lerp(AppColors other, double t) => AppColors(
    bg: Color.lerp(bg, other.bg, t)!,
    surface: Color.lerp(surface, other.surface, t)!,
    surfaceElevated: Color.lerp(surfaceElevated, other.surfaceElevated, t)!,
    border: Color.lerp(border, other.border, t)!,
    borderSubtle: Color.lerp(borderSubtle, other.borderSubtle, t)!,
    text: Color.lerp(text, other.text, t)!,
    textMuted: Color.lerp(textMuted, other.textMuted, t)!,
    textInverse: Color.lerp(textInverse, other.textInverse, t)!,
    primary: Color.lerp(primary, other.primary, t)!,
    primaryInverse: Color.lerp(primaryInverse, other.primaryInverse, t)!,
    error: Color.lerp(error, other.error, t)!,
    success: Color.lerp(success, other.success, t)!,
  );
}

abstract final class AppSpacing {
  static const double s4 = 4;
  static const double s6 = 6;
  static const double s8 = 8;
  static const double s10 = 10;
  static const double s12 = 12;
  static const double s14 = 14;
  static const double s16 = 16;
  static const double s20 = 20;
  static const double s24 = 24;
  static const double s28 = 28;
  static const double s32 = 32;
  static const double s36 = 36;
  static const double s40 = 40;
  static const double s48 = 48;
  static const double s56 = 56;
  static const double s64 = 64;
  static const double s80 = 80;
}

abstract final class AppRadii {
  static const double sm = 4;
  static const double md = 8;
  static const double lg = 12;
  static const double xl = 16;
  static const double pill = 999;
}

abstract final class AppMotion {
  static const Duration fast = Duration(milliseconds: 150);
  static const Duration normal = Duration(milliseconds: 300);
  static const Duration slow = Duration(milliseconds: 500);
  static const Curve curve = Curves.easeOutQuart;
}

abstract final class AppShadows {
  static List<BoxShadow> elevated(Color baseColor) => [
    BoxShadow(
      color: baseColor.withValues(alpha: 0.05),
      blurRadius: 10,
      offset: const Offset(0, 4),
    ),
    BoxShadow(
      color: baseColor.withValues(alpha: 0.02),
      blurRadius: 4,
      offset: const Offset(0, 2),
    ),
  ];
}

extension AppContext on BuildContext {
  AppColors get colors => Theme.of(this).extension<AppColors>()!;
  bool get isDark => Theme.of(this).brightness == Brightness.dark;
  TextTheme get text => Theme.of(this).textTheme;
}
