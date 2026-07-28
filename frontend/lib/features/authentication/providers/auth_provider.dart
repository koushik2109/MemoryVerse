import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:supabase_flutter/supabase_flutter.dart'
    hide AuthUser, UserIdentity;
import '../models/auth_user.dart';
import 'package:frontend/core/services/token_storage_service.dart';

// ── Auth Status ────────────────────────────────────────────────────────────

enum AuthStatus { initial, loading, authenticated, unauthenticated, error }

// ── Auth State ─────────────────────────────────────────────────────────────

class AuthState {
  final AuthStatus status;
  final AuthUser? user;
  final String? errorMessage;
  final String? pendingEmail;

  const AuthState({
    required this.status,
    this.user,
    this.errorMessage,
    this.pendingEmail,
  });

  const AuthState.initial() : this(status: AuthStatus.initial);

  bool get isLoading => status == AuthStatus.loading;
  bool get isAuthenticated => status == AuthStatus.authenticated;

  AuthState copyWith({
    AuthStatus? status,
    AuthUser? user,
    String? errorMessage,
    String? pendingEmail,
  }) {
    return AuthState(
      status: status ?? this.status,
      user: user ?? this.user,
      errorMessage: errorMessage ?? this.errorMessage,
      pendingEmail: pendingEmail ?? this.pendingEmail,
    );
  }
}

// ── Auth Notifier ──────────────────────────────────────────────────────────

class AuthNotifier extends Notifier<AuthState> {
  SupabaseClient get _client => Supabase.instance.client;
  TokenStorageService get _tokenStorage => TokenStorageService.instance;

  @override
  AuthState build() {
    _restoreSession();
    return const AuthState.initial();
  }

  // ── Session restore ──────────────────────────────────────────────────────

  Future<void> _restoreSession() async {
    final session = _client.auth.currentSession;
    if (session != null) {
      await _persistSession(session);
    }
  }

  Future<void> _persistSession(Session session) async {
    final user = session.user;
    await _tokenStorage.saveSession(
      accessToken: session.accessToken,
      refreshToken: session.refreshToken ?? '',
      userId: user.id,
      email: user.email ?? '',
    );
    state = state.copyWith(
      status: AuthStatus.authenticated,
      user: AuthUser(
        id: user.id,
        email: user.email ?? '',
        displayName: user.userMetadata?['display_name'] as String?,
        avatarUrl: user.userMetadata?['avatar_url'] as String?,
      ),
    );
  }

  // ── Login ────────────────────────────────────────────────────────────────

  Future<void> login({
    required String email,
    required String password,
  }) async {
    state = state.copyWith(status: AuthStatus.loading, errorMessage: null);
    try {
      final response = await _client.auth.signInWithPassword(
        email: email,
        password: password,
      );
      if (response.session != null) {
        await _persistSession(response.session!);
      }
    } on AuthException catch (e) {
      state = state.copyWith(
        status: AuthStatus.error,
        errorMessage: _friendlyAuthError(e.message),
      );
    } catch (_) {
      state = state.copyWith(
        status: AuthStatus.error,
        errorMessage: 'Login failed. Please try again.',
      );
    }
  }

  // ── Sign Up ──────────────────────────────────────────────────────────────

  Future<void> signup({
    required String name,
    required String email,
    required String password,
  }) async {
    state = state.copyWith(status: AuthStatus.loading, errorMessage: null);
    try {
      final response = await _client.auth.signUp(
        email: email,
        password: password,
        data: {'display_name': name},
      );
      if (response.session != null) {
        await _persistSession(response.session!);
      } else {
        // Email confirmation required — Supabase sent a confirmation email.
        state = state.copyWith(
          status: AuthStatus.unauthenticated,
          errorMessage: null,
          pendingEmail: email,
        );
      }
    } on AuthException catch (e) {
      state = state.copyWith(
        status: AuthStatus.error,
        errorMessage: _friendlyAuthError(e.message),
      );
    } catch (_) {
      state = state.copyWith(
        status: AuthStatus.error,
        errorMessage: 'Sign up failed. Please try again.',
      );
    }
  }

  // ── Google OAuth (scaffold) ──────────────────────────────────────────────

  Future<void> signInWithGoogle() async {
    state = state.copyWith(status: AuthStatus.loading, errorMessage: null);
    try {
      await _client.auth.signInWithOAuth(OAuthProvider.google);
      // Session is set by Supabase deep-link callback.
      state = state.copyWith(status: AuthStatus.unauthenticated);
    } catch (_) {
      state = state.copyWith(
        status: AuthStatus.error,
        errorMessage: 'Google sign-in is not configured yet.',
      );
    }
  }

  // ── Forgot password — send OTP ────────────────────────────────────────────

  Future<void> sendOtp({required String email}) async {
    state = state.copyWith(status: AuthStatus.loading, errorMessage: null);
    try {
      await _client.auth.resetPasswordForEmail(email);
      state = state.copyWith(
        status: AuthStatus.unauthenticated,
        pendingEmail: email,
      );
    } on AuthException catch (e) {
      state = state.copyWith(
        status: AuthStatus.error,
        errorMessage: _friendlyAuthError(e.message),
      );
    } catch (_) {
      state = state.copyWith(
        status: AuthStatus.error,
        errorMessage: 'Could not send reset code. Check the email address.',
      );
    }
  }

  // ── Verify OTP ────────────────────────────────────────────────────────────

  Future<bool> verifyOtp({
    required String email,
    required String otp,
  }) async {
    state = state.copyWith(status: AuthStatus.loading, errorMessage: null);
    try {
      final response = await _client.auth.verifyOTP(
        email: email,
        token: otp,
        type: OtpType.recovery,
      );
      if (response.session != null) {
        await _persistSession(response.session!);
        return true;
      }
      state = state.copyWith(
        status: AuthStatus.unauthenticated,
        pendingEmail: null,
      );
      return true;
    } on AuthException catch (e) {
      state = state.copyWith(
        status: AuthStatus.error,
        errorMessage: _friendlyAuthError(e.message),
      );
      return false;
    } catch (_) {
      state = state.copyWith(
        status: AuthStatus.error,
        errorMessage: 'OTP verification failed. Please try again.',
      );
      return false;
    }
  }

  // ── Logout ────────────────────────────────────────────────────────────────

  Future<void> logout() async {
    await _client.auth.signOut();
    await _tokenStorage.clearSession();
    state = const AuthState.initial();
  }

  // ── Helper ────────────────────────────────────────────────────────────────

  String _friendlyAuthError(String raw) {
    final lower = raw.toLowerCase();
    if (lower.contains('invalid login')) return 'Invalid email or password.';
    if (lower.contains('email not confirmed')) {
      return 'Please confirm your email first.';
    }
    if (lower.contains('already registered')) return 'This email is already in use.';
    if (lower.contains('otp')) return 'Invalid or expired code. Try resending.';
    return raw;
  }
}

// ── Provider ───────────────────────────────────────────────────────────────

final authProvider = NotifierProvider<AuthNotifier, AuthState>(
  AuthNotifier.new,
);

/// Convenience provider — true when a JWT session exists in secure storage.
final hasSessionProvider = FutureProvider<bool>((ref) async {
  return TokenStorageService.instance.hasValidSession;
});
