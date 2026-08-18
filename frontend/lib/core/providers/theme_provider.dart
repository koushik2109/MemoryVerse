import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter/material.dart';

const _kThemeKey = 'mv_theme_mode';

class ThemeModeNotifier extends StateNotifier<ThemeMode> {
  ThemeModeNotifier(super.initial);

  void toggle() {
    state = state == ThemeMode.dark ? ThemeMode.light : ThemeMode.dark;
    SharedPreferences.getInstance().then(
      (p) => p.setString(_kThemeKey, state.name),
    );
  }

  void setMode(ThemeMode mode) {
    state = mode;
    SharedPreferences.getInstance().then(
      (p) => p.setString(_kThemeKey, mode.name),
    );
  }
}

final themeModeProvider = StateNotifierProvider<ThemeModeNotifier, ThemeMode>((
  ref,
) {
  // Initial value is overridden in main.dart after reading SharedPreferences
  return ThemeModeNotifier(ThemeMode.system);
});
