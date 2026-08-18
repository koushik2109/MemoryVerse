import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../api/api_client.dart';

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
  ApiClient get _api => ref.read(apiClientProvider);

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
    String? displayName,
  }) async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      await _api.post('/auth/signup', data: {
        'email': email,
        'password': password,
        'full_name': displayName,
      });
    });
  }

  Future<void> verifyOtp({required String email, required String token, OtpType type = OtpType.signup}) async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      if (type == OtpType.recovery) {
        // Recovery logic requires new password which isn't passed here. 
        // We will assume the UI sends the new password via another method or we update UI.
        // Actually, the UI for password reset will need to be updated. For now throw unimplemented if it's recovery.
        if (type == OtpType.recovery) {
          throw UnimplementedError("Recovery OTP requires new password. Use resetPassword API instead.");
        }
      }
      
      await _api.post('/auth/verify-otp', data: {
        'email': email,
        'otp': token,
      });
      // OTP verified successfully on backend, user must now sign in
    });
  }

  Future<void> sendPasswordReset(String email) async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      await _api.post('/auth/forgot-password', data: {
        'email': email,
      });
    });
  }

  Future<void> resendOtp({required String email}) async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      await _api.post('/auth/resend-otp', data: {
        'email': email,
      });
    });
  }

  Future<void> resetPassword({required String email, required String token, required String newPassword}) async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      await _api.post('/auth/reset-password', data: {
        'email': email,
        'otp': token,
        'new_password': newPassword,
      });
    });
  }

  Future<void> signInWithGoogle() async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      final googleSignIn = GoogleSignIn(
        clientId: '413763126044-jgj0jq3s55pemg1jj4f53ratglvmv662.apps.googleusercontent.com',
      );
      final googleUser = await googleSignIn.signIn();
      if (googleUser == null) throw Exception('Google sign-in cancelled');

      final googleAuth = await googleUser.authentication;
      final idToken = googleAuth.idToken;
      if (idToken == null) throw Exception('Failed to get Google ID token');

      await _client.auth.signInWithIdToken(
        provider: OAuthProvider.google,
        idToken: idToken,
        accessToken: googleAuth.accessToken,
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
