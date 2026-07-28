import 'package:flutter_secure_storage/flutter_secure_storage.dart';

/// Manages secure storage of the Supabase JWT access token and refresh token.
///
/// Uses [FlutterSecureStorage] (Keychain on iOS, Keystore on Android,
/// libsecret on Linux, DPAPI on Windows).
class TokenStorageService {
  const TokenStorageService._();
  static const TokenStorageService instance = TokenStorageService._();

  static final _storage = const FlutterSecureStorage(
    aOptions: AndroidOptions(encryptedSharedPreferences: true),
  );

  static const _kAccessToken = 'mv_access_token';
  static const _kRefreshToken = 'mv_refresh_token';
  static const _kUserId = 'mv_user_id';
  static const _kUserEmail = 'mv_user_email';

  // ── Write ────────────────────────────────────────────────────────────────

  Future<void> saveSession({
    required String accessToken,
    required String refreshToken,
    required String userId,
    required String email,
  }) async {
    await Future.wait([
      _storage.write(key: _kAccessToken, value: accessToken),
      _storage.write(key: _kRefreshToken, value: refreshToken),
      _storage.write(key: _kUserId, value: userId),
      _storage.write(key: _kUserEmail, value: email),
    ]);
  }

  // ── Read ─────────────────────────────────────────────────────────────────

  Future<String?> get accessToken => _storage.read(key: _kAccessToken);
  Future<String?> get refreshToken => _storage.read(key: _kRefreshToken);
  Future<String?> get userId => _storage.read(key: _kUserId);
  Future<String?> get userEmail => _storage.read(key: _kUserEmail);

  Future<bool> get hasValidSession async {
    final token = await accessToken;
    return token != null && token.isNotEmpty;
  }

  // ── Clear ────────────────────────────────────────────────────────────────

  Future<void> clearSession() async {
    await Future.wait([
      _storage.delete(key: _kAccessToken),
      _storage.delete(key: _kRefreshToken),
      _storage.delete(key: _kUserId),
      _storage.delete(key: _kUserEmail),
    ]);
  }
}
