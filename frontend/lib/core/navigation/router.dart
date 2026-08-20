import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/core/navigation/shell.dart';
import 'package:memory_verse/features/auth/presentation/forgot_password_screen.dart';
import 'package:memory_verse/features/auth/presentation/landing_screen.dart';
import 'package:memory_verse/features/auth/presentation/sign_in_screen.dart';
import 'package:memory_verse/features/auth/presentation/sign_up_screen.dart';
import 'package:memory_verse/features/auth/presentation/otp_verification_screen.dart';
import 'package:memory_verse/features/auth/presentation/reset_password_screen.dart';
import 'package:memory_verse/features/home/presentation/home_screen.dart';
import 'package:memory_verse/features/timeline/presentation/timeline_screen.dart';
import 'package:memory_verse/features/ai/presentation/ai_screen.dart';
import 'package:memory_verse/features/memories/presentation/memories_screen.dart';
import 'package:memory_verse/features/profile/presentation/profile_screen.dart';
import 'package:memory_verse/features/splash/presentation/splash_screen.dart';
import 'package:memory_verse/features/search/presentation/search_screen.dart';
import 'package:memory_verse/features/notifications/presentation/notifications_screen.dart';
import 'package:memory_verse/features/settings/presentation/settings_screen.dart';
import 'package:memory_verse/features/vaults/presentation/join_vault_dialog.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

// ── Route paths ───────────────────────────────────────────
abstract final class Routes {
  static const splash = '/';
  static const landing = '/landing';
  static const signIn = '/sign-in';
  static const signUp = '/sign-up';
  static const otpVerify = '/otp-verify';
  static const forgotPassword = '/forgot-password';
  static const home = '/home';
  static const timeline = '/timeline';
  static const memories = '/memories';
  static const profile = '/profile';
  static const ai = '/ai';
  static const search = '/search';
  static const notifications = '/notifications';
  static const settings = '/settings';

  static const _authRoutes = {
    landing,
    signIn,
    signUp,
    forgotPassword,
    otpVerify,
  };
  static const _shellRoutes = {home, timeline, memories, profile};
}

// ── GoRouterRefreshStream ────────────────────────────────

class GoRouterRefreshStream extends ChangeNotifier {
  GoRouterRefreshStream(Stream<AuthState> stream) {
    notifyListeners();
    _sub = stream.listen((_) => notifyListeners());
  }
  late final StreamSubscription<AuthState> _sub;

  @override
  void dispose() {
    _sub.cancel();
    super.dispose();
  }
}

// ── Router provider ──────────────────────────────────────

final routerProvider = Provider<GoRouter>((ref) {
  final refresh = GoRouterRefreshStream(
    Supabase.instance.client.auth.onAuthStateChange,
  );

  return GoRouter(
    initialLocation: Routes.splash,
    refreshListenable: refresh,
    redirect: (context, state) {
      final path = state.uri.path;
      final hasSession = Supabase.instance.client.auth.currentSession != null;

      if (path == Routes.splash) return null;

      if (!hasSession && Routes._shellRoutes.contains(path)) {
        return Routes.landing;
      }
      if (hasSession && Routes._authRoutes.contains(path)) return Routes.home;

      return null;
    },
    routes: [
      GoRoute(
        path: Routes.splash,
        pageBuilder: (_, __) => const NoTransitionPage(child: SplashScreen()),
      ),
      GoRoute(
        path: Routes.landing,
        pageBuilder: (_, __) => _fade(const LandingScreen()),
      ),
      GoRoute(
        path: Routes.signIn,
        pageBuilder: (_, __) => _fade(const SignInScreen()),
      ),
      GoRoute(
        path: Routes.signUp,
        pageBuilder: (_, __) => _slide(const SignUpScreen()),
      ),
      GoRoute(
        path: Routes.forgotPassword,
        pageBuilder: (_, __) => _slide(const ForgotPasswordScreen()),
      ),
      GoRoute(
        path: '/reset-password',
        pageBuilder: (context, state) {
          final email = state.extra as String? ?? '';
          return _slide(ResetPasswordScreen(email: email));
        },
      ),
      GoRoute(
        path: Routes.otpVerify,
        pageBuilder: (context, state) {
          final email = state.extra as String? ?? '';
          return _slide(OtpVerificationScreen(email: email));
        },
      ),
      GoRoute(
        path: Routes.search,
        pageBuilder: (_, __) => _slide(const SearchScreen()),
      ),
      GoRoute(
        path: Routes.notifications,
        pageBuilder: (_, __) => _slide(const NotificationsScreen()),
      ),
      GoRoute(
        path: Routes.settings,
        pageBuilder: (_, __) => _slide(const SettingsScreen()),
      ),
      GoRoute(
        path: Routes.ai,
        pageBuilder: (_, __) => _slide(const AiScreen()),
      ),
      GoRoute(
        path: '/join/:code',
        pageBuilder: (context, state) {
          final code = state.pathParameters['code'];
          return _fade(JoinVaultDialog(initialCode: code));
        },
      ),
      // Shell with 4 tabs: Home, Timeline, Memories, Profile
      // (Create is a center action, not a tab)
      StatefulShellRoute.indexedStack(
        builder: (context, state, shell) => AppShell(shell: shell),
        branches: [
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: Routes.home,
                builder: (_, __) => const HomeScreen(),
              ),
            ],
          ),
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: Routes.timeline,
                builder: (_, __) => const TimelineScreen(),
              ),
            ],
          ),
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: Routes.memories,
                builder: (_, __) => const MemoriesScreen(),
              ),
            ],
          ),
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: Routes.profile,
                builder: (_, __) => const ProfileScreen(),
              ),
            ],
          ),
        ],
      ),
    ],
  );
});

// ── Transition helpers ───────────────────────────────────

CustomTransitionPage<void> _fade(Widget child) => CustomTransitionPage(
  child: child,
  transitionsBuilder: (_, anim, __, child) =>
      FadeTransition(opacity: anim, child: child),
  transitionDuration: const Duration(milliseconds: 280),
);

CustomTransitionPage<void> _slide(Widget child) => CustomTransitionPage(
  child: child,
  transitionsBuilder: (_, anim, __, child) => SlideTransition(
    position: Tween(
      begin: const Offset(1, 0),
      end: Offset.zero,
    ).chain(CurveTween(curve: Curves.easeOutCubic)).animate(anim),
    child: child,
  ),
  transitionDuration: const Duration(milliseconds: 300),
);
