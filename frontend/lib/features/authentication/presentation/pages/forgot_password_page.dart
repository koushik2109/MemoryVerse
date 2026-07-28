import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:frontend/features/authentication/providers/auth_provider.dart';
import 'package:frontend/features/authentication/presentation/widgets/auth_gradient_background.dart';
import 'package:frontend/features/authentication/presentation/widgets/auth_glass_card.dart';
import 'package:frontend/features/authentication/presentation/widgets/auth_primary_button.dart';
import 'package:frontend/features/authentication/presentation/widgets/auth_text_field.dart';
import 'package:frontend/shared/constants/app_spacing.dart';

/// Screen where users enter their email to receive a password-reset OTP.
class ForgotPasswordPage extends ConsumerStatefulWidget {
  const ForgotPasswordPage({super.key});

  @override
  ConsumerState<ForgotPasswordPage> createState() => _ForgotPasswordPageState();
}

class _ForgotPasswordPageState extends ConsumerState<ForgotPasswordPage> {
  final _formKey = GlobalKey<FormState>();
  final _emailCtrl = TextEditingController();

  @override
  void dispose() {
    _emailCtrl.dispose();
    super.dispose();
  }

  Future<void> _onSend() async {
    if (!_formKey.currentState!.validate()) return;
    FocusScope.of(context).unfocus();
    await ref.read(authProvider.notifier).sendOtp(email: _emailCtrl.text.trim());
    final authState = ref.read(authProvider);
    if (!mounted) return;
    if (authState.errorMessage != null) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text(authState.errorMessage!),
        backgroundColor: Theme.of(context).colorScheme.error,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(AppRadius.md)),
      ));
      return;
    }
    context.push('/otp-verification', extra: _emailCtrl.text.trim());
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;
    final isLoading = ref.watch(authProvider).isLoading;
    final width = MediaQuery.sizeOf(context).width;
    final cardWidth = width > 600 ? 420.0 : width - AppSpacing.xl;

    return Scaffold(
      resizeToAvoidBottomInset: true,
      body: AuthGradientBackground(
        child: SafeArea(
          child: Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(
                vertical: AppSpacing.xxl,
                horizontal: AppSpacing.md,
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // ── Icon ──────────────────────────────────────────────────
                  Icon(Icons.lock_reset_rounded, size: AppIconSize.xxl, color: cs.onPrimary)
                      .animate()
                      .fadeIn(duration: 600.ms)
                      .scale(begin: const Offset(0.6, 0.6), duration: 600.ms, curve: Curves.easeOutBack),

                  const SizedBox(height: AppSpacing.md),

                  Text(
                    'Forgot Password?',
                    style: tt.headlineMedium?.copyWith(
                      color: cs.onPrimary,
                      fontWeight: FontWeight.w800,
                      letterSpacing: -0.8,
                    ),
                  ).animate().fadeIn(duration: 500.ms, delay: 100.ms),

                  const SizedBox(height: AppSpacing.xs),

                  Text(
                    "No worries — we'll send a reset code to your email.",
                    style: tt.bodySmall?.copyWith(color: cs.onPrimary.withValues(alpha: 0.7)),
                    textAlign: TextAlign.center,
                  ).animate().fadeIn(duration: 400.ms, delay: 160.ms),

                  const SizedBox(height: AppSpacing.xl),

                  // ── Card ──────────────────────────────────────────────────
                  SizedBox(
                    width: cardWidth,
                    child: AuthGlassCard(
                      child: Form(
                        key: _formKey,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            // Email
                            AuthTextField(
                              controller: _emailCtrl,
                              label: 'Email address',
                              prefixIcon: Icons.email_outlined,
                              keyboardType: TextInputType.emailAddress,
                              textInputAction: TextInputAction.done,
                              autofillHints: const [AutofillHints.email],
                              enabled: !isLoading,
                              onFieldSubmitted: _onSend,
                              validator: (v) {
                                if (v == null || v.isEmpty) return 'Email is required';
                                if (!v.contains('@')) return 'Enter a valid email';
                                return null;
                              },
                            ).animate().fadeIn(duration: 350.ms, delay: 280.ms),

                            const SizedBox(height: AppSpacing.lg),

                            // Send button
                            AuthPrimaryButton(
                              label: 'Send Reset Code',
                              onPressed: _onSend,
                              isLoading: isLoading,
                            ).animate().fadeIn(duration: 350.ms, delay: 360.ms),

                            const SizedBox(height: AppSpacing.lg),

                            // Back to login
                            Center(
                              child: GestureDetector(
                                onTap: () => context.go('/login'),
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Icon(
                                      Icons.arrow_back_rounded,
                                      size: AppIconSize.sm,
                                      color: cs.onPrimary.withValues(alpha: 0.8),
                                    ),
                                    const SizedBox(width: AppSpacing.xs),
                                    Text(
                                      'Back to Sign In',
                                      style: tt.labelMedium?.copyWith(
                                        color: cs.onPrimary.withValues(alpha: 0.8),
                                        fontWeight: FontWeight.w600,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ).animate().fadeIn(duration: 300.ms, delay: 440.ms),
                          ],
                        ),
                      ),
                    ),
                  )
                      .animate()
                      .fadeIn(duration: 600.ms, delay: 200.ms)
                      .slideY(begin: 0.2, end: 0, duration: 600.ms, delay: 200.ms, curve: Curves.easeOutQuint),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
