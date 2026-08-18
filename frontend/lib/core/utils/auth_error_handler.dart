/// Converts raw Supabase / Dart exceptions into clean,
/// human-readable messages that never leak internal identifiers.
abstract final class AuthErrorHandler {
  static String parse(Object error) {
    final raw = error.toString().toLowerCase();

    // ── Credentials / account ─────────────────────────────
    if (raw.contains('invalid_credentials') ||
        raw.contains('invalid login credentials') ||
        raw.contains('email not confirmed') && raw.contains('password')) {
      return 'Incorrect email or password. Please try again.';
    }

    if (raw.contains('email not confirmed') ||
        raw.contains('email not verified') ||
        raw.contains('email_not_confirmed')) {
      return "Your email hasn't been verified. Please check your inbox.";
    }

    if (raw.contains('user already registered') ||
        raw.contains('already been registered') ||
        raw.contains('email_exists') ||
        raw.contains('email address is already')) {
      return 'This email is already registered. Try signing in instead.';
    }

    if (raw.contains('weak_password') || raw.contains('password should be')) {
      return 'Password is too weak. Use at least 8 characters.';
    }

    // ── Rate limiting ──────────────────────────────────────
    if (raw.contains('rate_limit') ||
        raw.contains('too many requests') ||
        raw.contains('for security') ||
        raw.contains('please wait')) {
      if (raw.contains('please wait')) return error.toString();
      return 'Too many attempts. Please wait a moment and try again.';
    }

    // ── Network / connectivity ─────────────────────────────
    if (raw.contains('socketexception') ||
        raw.contains('network') ||
        raw.contains('connection refused') ||
        raw.contains('failed host lookup')) {
      return 'No internet connection. Please check your network.';
    }

    if (raw.contains('timeout') || raw.contains('timed out')) {
      return 'Request timed out. Please try again.';
    }

    // ── OTP / token ────────────────────────────────────────
    if (raw.contains('otp_expired') ||
        raw.contains('token has expired') ||
        raw.contains('expired_token')) {
      return 'This code has expired. Please request a new one.';
    }

    if (raw.contains('otp_invalid') ||
        raw.contains('invalid otp') ||
        raw.contains('token is invalid')) {
      return 'The code you entered is incorrect.';
    }

    // ── Session / account state ────────────────────────────
    if (raw.contains('user not found') || raw.contains('no user found')) {
      return 'No account found with that email.';
    }

    if (raw.contains('session_not_found') || raw.contains('invalid session')) {
      return 'Your session has expired. Please sign in again.';
    }

    // ── Fallback ───────────────────────────────────────────
    if (!raw.contains('exception') &&
        !raw.contains('error:') &&
        error.toString().length < 150) {
      return error.toString();
    }
    return 'Error: $error';
  }
}
