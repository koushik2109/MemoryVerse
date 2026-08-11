import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

// ── Raw auth state stream ─────────────────────────────
final authStreamProvider = StreamProvider<AuthState>((ref) {
  return Supabase.instance.client.auth.onAuthStateChange;
});

// ── Current session shortcut ──────────────────────────
final sessionProvider = Provider<Session?>((ref) {
  return ref.watch(authStreamProvider).valueOrNull?.session;
});

// ── Current user shortcut ─────────────────────────────
final currentUserProvider = Provider<User?>((ref) {
  return Supabase.instance.client.auth.currentUser;
});

// ── Auth actions notifier ─────────────────────────────
class AuthNotifier extends AsyncNotifier<void> {
  SupabaseClient get _client => Supabase.instance.client;

  @override
  Future<void> build() async {}

  Future<void> signIn({required String email, required String password}) async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      await _client.auth.signInWithPassword(email: email, password: password);
    });
  }

  Future<void> signUp({
    required String email,
    required String password,
    required String displayName,
  }) async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      await _client.auth.signUp(
        email: email,
        password: password,
        data: {'full_name': displayName},
      );
    });
  }

  Future<void> verifyOtp({required String email, required String token, OtpType type = OtpType.signup}) async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      await _client.auth.verifyOTP(
        email: email,
        token: token,
        type: type,
      );
    });
  }

  Future<void> sendPasswordReset(String email) async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      await _client.auth.resetPasswordForEmail(
        email,
        redirectTo: 'io.supabase.memoryverse://login-callback',
      );
    });
  }

  Future<void> signOut() async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      await _client.auth.signOut();
    });
  }
}

final authNotifierProvider =
    AsyncNotifierProvider<AuthNotifier, void>(AuthNotifier.new);
