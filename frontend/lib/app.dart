import 'package:flutter/material.dart';
import 'package:memory_verse/config/theme/app_theme.dart';
import 'package:memory_verse/features/authentication/presentation/auth_screen.dart';
import 'package:memory_verse/features/onboarding/presentation/splash_screen.dart';
import 'package:memory_verse/features/onboarding/presentation/welcome_screen.dart';

// ─────────────────────────────────────────────
//  App Routes
// ─────────────────────────────────────────────

class AppRoutes {
  AppRoutes._();

  static const String splash = '/';
  static const String welcome = '/welcome';
  static const String auth = '/auth';
}

class AppRouter {
  AppRouter._();

  static Route<dynamic> generateRoute(RouteSettings settings) {
    switch (settings.name) {
      case AppRoutes.splash:
        return _fadeRoute(const SplashScreen(), settings);
      case AppRoutes.welcome:
        return _fadeRoute(const WelcomeScreen(), settings);
      case AppRoutes.auth:
        final isSignUp = (settings.arguments as bool?) ?? false;
        return _slideRoute(AuthScreen(initialSignUp: isSignUp), settings);
      default:
        return _fadeRoute(const SplashScreen(), settings);
    }
  }

  static PageRoute<T> _fadeRoute<T>(Widget page, RouteSettings settings) {
    return PageRouteBuilder<T>(
      settings: settings,
      pageBuilder: (_, __, ___) => page,
      transitionDuration: const Duration(milliseconds: 600),
      transitionsBuilder: (_, animation, __, child) {
        return FadeTransition(opacity: animation, child: child);
      },
    );
  }

  static PageRoute<T> _slideRoute<T>(Widget page, RouteSettings settings) {
    return PageRouteBuilder<T>(
      settings: settings,
      pageBuilder: (_, __, ___) => page,
      transitionDuration: const Duration(milliseconds: 500),
      transitionsBuilder: (_, animation, __, child) {
        final tween = Tween(begin: const Offset(0, 1), end: Offset.zero)
            .chain(CurveTween(curve: Curves.easeOutCubic));
        return SlideTransition(
          position: animation.drive(tween),
          child: child,
        );
      },
    );
  }
}

// ─────────────────────────────────────────────
//  Root App Widget
// ─────────────────────────────────────────────

class MemoryVerseApp extends StatelessWidget {
  const MemoryVerseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MemoryVerse',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.dark,
      initialRoute: AppRoutes.splash,
      onGenerateRoute: AppRouter.generateRoute,
    );
  }
}
