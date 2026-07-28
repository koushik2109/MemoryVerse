import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:go_router/go_router.dart';
import 'package:supabase_flutter/supabase_flutter.dart' as sb;

import 'core/theme/app_theme.dart';
import 'features/authentication/presentation/pages/splash_page.dart';
import 'features/authentication/presentation/pages/login_page.dart';
import 'features/authentication/presentation/pages/signup_page.dart';
import 'features/authentication/presentation/pages/forgot_password_page.dart';
import 'features/authentication/presentation/pages/otp_verification_page.dart';
import 'features/home/presentation/pages/home_page.dart';

// ── Auth-aware router ─────────────────────────────────────────────────────────

GoRouter _buildRouter() {
  return GoRouter(
    initialLocation: '/',
    redirect: (context, state) {
      final session = sb.Supabase.instance.client.auth.currentSession;
      final isAuthenticated = session != null;
      final isOnAuthPage = state.matchedLocation == '/login' ||
          state.matchedLocation == '/signup' ||
          state.matchedLocation == '/forgot-password' ||
          state.matchedLocation == '/otp-verification';

      // Logged-in user trying to access auth pages → send to home
      if (isAuthenticated && isOnAuthPage) return '/home';

      return null; // no redirect needed
    },
    routes: [
      GoRoute(
        path: '/',
        builder: (context, state) => const SplashPage(),
      ),
      GoRoute(
        path: '/login',
        builder: (context, state) => const LoginPage(),
      ),
      GoRoute(
        path: '/signup',
        builder: (context, state) => const SignupPage(),
      ),
      GoRoute(
        path: '/forgot-password',
        builder: (context, state) => const ForgotPasswordPage(),
      ),
      GoRoute(
        path: '/otp-verification',
        builder: (context, state) {
          final email = state.extra as String? ?? '';
          return OtpVerificationPage(email: email);
        },
      ),
      GoRoute(
        path: '/home',
        builder: (context, state) => const HomePage(),
      ),
    ],
  );
}

// ── App ───────────────────────────────────────────────────────────────────────

const _kDesignWidth = 390.0;
const _kDesignHeight = 844.0;

class MemoryVerseApp extends ConsumerWidget {
  const MemoryVerseApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return ScreenUtilInit(
      designSize: const Size(_kDesignWidth, _kDesignHeight),
      minTextAdapt: true,
      splitScreenMode: true,
      builder: (_, child) {
        return MaterialApp.router(
          title: 'MemoryVerse',
          theme: AppTheme.lightTheme,
          darkTheme: AppTheme.darkTheme,
          themeMode: ThemeMode.system,
          routerConfig: _buildRouter(),
          debugShowCheckedModeBanner: false,
        );
      },
    );
  }
}
