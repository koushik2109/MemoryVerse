import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:frontend/features/authentication/providers/auth_provider.dart';
import 'package:frontend/features/authentication/presentation/widgets/auth_gradient_background.dart';
import 'package:frontend/features/authentication/presentation/widgets/auth_glass_card.dart';
import 'package:frontend/features/authentication/presentation/widgets/auth_primary_button.dart';
import 'package:frontend/features/authentication/presentation/widgets/otp_input_field.dart';
import 'package:frontend/shared/constants/app_spacing.dart';

/// OTP verification screen.
///
/// Receives the user's [email] via GoRouter [extra].
/// Shows a 6-digit OTP input, a 60-second resend countdown,
/// and a verify button.
class OtpVerificationPage extends ConsumerStatefulWidget {
  const OtpVerificationPage({super.key, required this.email});

  final String email;

  @override
  ConsumerState<OtpVerificationPage> createState() =>
      _OtpVerificationPageState();
}

class _OtpVerificationPageState extends ConsumerState<OtpVerificationPage> {
  static const int _kResendSeconds = 60;

  String _otp = '';
  bool _otpComplete = false;
  int _secondsLeft = _kResendSeconds;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _startTimer();
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  void _startTimer() {
    _timer?.cancel();
    setState(() => _secondsLeft = _kResendSeconds);
    _timer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (_secondsLeft == 0) {
        t.cancel();
      } else {
        if (mounted) setState(() => _secondsLeft--);
      }
    });
  }

  Future<void> _onResend() async {
    await ref.read(authProvider.notifier).sendOtp(email: widget.email);
    _startTimer();
  }

  Future<void> _onVerify() async {
    if (!_otpComplete) return;
    final success = await ref.read(authProvider.notifier).verifyOtp(
          email: widget.email,
          otp: _otp,
        );
    if (!mounted) return;
    if (success) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text('OTP verified! Please sign in.'),
          backgroundColor: Theme.of(context).colorScheme.primary,
        ),
      );
      context.go('/login');
    } else {
      final err = ref.read(authProvider).errorMessage;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(err ?? 'Invalid OTP.'),
          backgroundColor: Theme.of(context).colorScheme.error,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;
    final isLoading = ref.watch(authProvider).isLoading;
    final width = MediaQuery.sizeOf(context).width;
    final cardWidth = width > 600 ? 440.0 : width * 0.9;

    final maskedEmail = _maskEmail(widget.email);

    return Scaffold(
      resizeToAvoidBottomInset: true,
      body: AuthGradientBackground(
        child: SafeArea(
          child: Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(
                vertical: AppSpacing.xl,
                horizontal: AppSpacing.md,
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // ── Icon ────────────────────────────────────────────────────
                  Icon(
                    Icons.mark_email_read_rounded,
                    size: AppIconSize.xxl,
                    color: cs.onPrimary,
                  )
                      .animate()
                      .fadeIn(duration: 600.ms)
                      .scale(
                        begin: const Offset(0.6, 0.6),
                        duration: 600.ms,
                        curve: Curves.easeOutBack,
                      ),

                  const SizedBox(height: AppSpacing.md),

                  Text(
                    'Check your email',
                    style: tt.headlineLarge?.copyWith(
                      color: cs.onPrimary,
                      fontWeight: FontWeight.w800,
                      letterSpacing: -0.5,
                    ),
                  )
                      .animate()
                      .fadeIn(duration: 500.ms, delay: 100.ms)
                      .slideY(begin: 0.2, end: 0, duration: 400.ms, delay: 100.ms),

                  const SizedBox(height: AppSpacing.sm),

                  RichText(
                    textAlign: TextAlign.center,
                    text: TextSpan(
                      style: tt.bodyMedium?.copyWith(
                        color: cs.onPrimary.withValues(alpha: 0.75),
                      ),
                      children: [
                        const TextSpan(text: 'We sent a 6-digit code to\n'),
                        TextSpan(
                          text: maskedEmail,
                          style: tt.bodyMedium?.copyWith(
                            color: cs.onPrimary,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),
                  ).animate().fadeIn(duration: 400.ms, delay: 180.ms),

                  const SizedBox(height: AppSpacing.xl),

                  SizedBox(
                    width: cardWidth,
                    child: AuthGlassCard(
                      child: Column(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              Text(
                                'Enter verification code',
                                style: tt.titleMedium?.copyWith(
                                  color: cs.onPrimary,
                                  fontWeight: FontWeight.w600,
                                ),
                                textAlign: TextAlign.center,
                              ),

                              const SizedBox(height: AppSpacing.lg),

                              // ── OTP boxes ──────────────────────────────────
                              OtpInputField(
                                enabled: !isLoading,
                                onChanged: (v) => setState(() {
                                  _otp = v;
                                  _otpComplete = v.length == 6;
                                }),
                                onCompleted: (v) => setState(() {
                                  _otp = v;
                                  _otpComplete = true;
                                }),
                              ).animate().fadeIn(duration: 400.ms, delay: 300.ms),

                              const SizedBox(height: AppSpacing.xl),

                              // ── Verify button ──────────────────────────────
                              AuthPrimaryButton(
                                label: 'Verify Code',
                                onPressed: _otpComplete ? _onVerify : null,
                                isLoading: isLoading,
                              ).animate().fadeIn(duration: 400.ms, delay: 400.ms),

                              const SizedBox(height: AppSpacing.lg),

                              // ── Resend countdown ───────────────────────────
                              Center(
                                child: _secondsLeft > 0
                                    ? Text.rich(
                                        TextSpan(
                                          style: tt.bodySmall?.copyWith(
                                            color: cs.onPrimary
                                                .withValues(alpha: 0.65),
                                          ),
                                          children: [
                                            const TextSpan(
                                              text: 'Resend code in ',
                                            ),
                                            TextSpan(
                                              text: '${_secondsLeft}s',
                                              style: tt.bodySmall?.copyWith(
                                                color: cs.onPrimary,
                                                fontWeight: FontWeight.w700,
                                              ),
                                            ),
                                          ],
                                        ),
                                      )
                                    : GestureDetector(
                                        onTap: _onResend,
                                        child: Text(
                                          'Resend Code',
                                          style: tt.labelMedium?.copyWith(
                                            color: cs.onPrimary,
                                            fontWeight: FontWeight.w700,
                                            decoration: TextDecoration.underline,
                                            decorationColor: cs.onPrimary
                                                .withValues(alpha: 0.5),
                                          ),
                                        ),
                                      ),
                              ).animate().fadeIn(duration: 300.ms, delay: 500.ms),

                              const SizedBox(height: AppSpacing.md),

                              // ── Back ────────────────────────────────────────
                              Center(
                                child: GestureDetector(
                                  onTap: () => context.go('/forgot-password'),
                                  child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Icon(
                                        Icons.arrow_back_rounded,
                                        size: AppIconSize.sm,
                                        color: cs.onPrimary
                                            .withValues(alpha: 0.75),
                                      ),
                                      const SizedBox(width: AppSpacing.xs),
                                      Text(
                                        'Change email',
                                        style: tt.labelMedium?.copyWith(
                                          color: cs.onPrimary
                                              .withValues(alpha: 0.75),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ).animate().fadeIn(duration: 300.ms, delay: 560.ms),
                            ],
                          ),
                    ),
                  )
                      .animate()
                      .fadeIn(duration: 700.ms, delay: 200.ms)
                      .slideY(
                        begin: 0.25,
                        end: 0,
                        duration: 700.ms,
                        delay: 200.ms,
                        curve: Curves.easeOutQuint,
                      ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  /// Masks `user@example.com` → `u***@example.com`
  String _maskEmail(String email) {
    final parts = email.split('@');
    if (parts.length != 2) return email;
    final local = parts[0];
    final domain = parts[1];
    if (local.length <= 1) return email;
    return '${local[0]}${'*' * (local.length - 1)}@$domain';
  }
}
