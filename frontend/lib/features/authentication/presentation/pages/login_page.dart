import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:frontend/features/authentication/providers/auth_provider.dart';
import 'package:frontend/features/authentication/presentation/widgets/auth_gradient_background.dart';
import 'package:frontend/features/authentication/presentation/widgets/auth_glass_card.dart';
import 'package:frontend/features/authentication/presentation/widgets/auth_primary_button.dart';
import 'package:frontend/features/authentication/presentation/widgets/auth_text_field.dart';
import 'package:frontend/features/authentication/presentation/widgets/social_login_button.dart';
import 'package:frontend/shared/constants/app_spacing.dart';

class LoginPage extends ConsumerStatefulWidget {
  const LoginPage({super.key});

  @override
  ConsumerState<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends ConsumerState<LoginPage> {
  final _formKey = GlobalKey<FormState>();
  final _emailCtrl = TextEditingController();
  final _passwordCtrl = TextEditingController();
  bool _googleLoading = false;

  @override
  void dispose() {
    _emailCtrl.dispose();
    _passwordCtrl.dispose();
    super.dispose();
  }

  Future<void> _onLogin() async {
    if (!_formKey.currentState!.validate()) return;
    FocusScope.of(context).unfocus();
    await ref.read(authProvider.notifier).login(
          email: _emailCtrl.text.trim(),
          password: _passwordCtrl.text,
        );
    final authState = ref.read(authProvider);
    if (!mounted) return;
    if (authState.isAuthenticated) {
      context.go('/home');
    } else if (authState.errorMessage != null) {
      _showError(authState.errorMessage!);
    }
  }

  Future<void> _onGoogle() async {
    setState(() => _googleLoading = true);
    await ref.read(authProvider.notifier).signInWithGoogle();
    if (mounted) setState(() => _googleLoading = false);
  }

  void _showError(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg),
        backgroundColor: Theme.of(context).colorScheme.error,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(AppRadius.md)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;
    final isLoading = ref.watch(authProvider).isLoading;
    final isDark = Theme.of(context).brightness == Brightness.dark;
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
                  // ── Logo & brand ─────────────────────────────────────────
                  Image.asset(
                    isDark
                        ? 'assets/icons/memoryverse_darktheme.png'
                        : 'assets/icons/memoryverse_lighttheme.png',
                    width: 72,
                    height: 72,
                    errorBuilder: (_, e, s) => Icon(
                      Icons.auto_stories_rounded,
                      size: 64,
                      color: cs.onPrimary,
                    ),
                  )
                      .animate()
                      .fadeIn(duration: 600.ms)
                      .scale(begin: const Offset(0.7, 0.7), duration: 600.ms, curve: Curves.easeOutBack),

                  const SizedBox(height: AppSpacing.sm),

                  Text(
                    'MemoryVerse',
                    style: tt.headlineMedium?.copyWith(
                      color: cs.onPrimary,
                      fontWeight: FontWeight.w800,
                      letterSpacing: -0.8,
                    ),
                  ).animate().fadeIn(duration: 500.ms, delay: 120.ms),

                  const SizedBox(height: AppSpacing.xs),

                  Text(
                    'Your personal memory journal',
                    style: tt.bodySmall?.copyWith(
                      color: cs.onPrimary.withValues(alpha: 0.7),
                      letterSpacing: 0.3,
                    ),
                  ).animate().fadeIn(duration: 400.ms, delay: 200.ms),

                  const SizedBox(height: AppSpacing.xl),

                  // ── Card ─────────────────────────────────────────────────
                  SizedBox(
                    width: cardWidth,
                    child: AuthGlassCard(
                      child: Form(
                        key: _formKey,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            // Header
                            Text(
                              'Welcome back',
                              style: tt.titleLarge?.copyWith(
                                color: cs.onPrimary,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            const SizedBox(height: AppSpacing.xs),
                            Text(
                              'Sign in to continue your story',
                              style: tt.bodySmall?.copyWith(
                                color: cs.onPrimary.withValues(alpha: 0.65),
                              ),
                            ),

                            const SizedBox(height: AppSpacing.lg),

                            // Email
                            AuthTextField(
                              controller: _emailCtrl,
                              label: 'Email address',
                              prefixIcon: Icons.email_outlined,
                              keyboardType: TextInputType.emailAddress,
                              textInputAction: TextInputAction.next,
                              autofillHints: const [AutofillHints.email],
                              enabled: !isLoading,
                              validator: (v) {
                                if (v == null || v.isEmpty) return 'Email is required';
                                if (!v.contains('@')) return 'Enter a valid email';
                                return null;
                              },
                            ).animate().fadeIn(duration: 350.ms, delay: 280.ms),

                            const SizedBox(height: AppSpacing.md),

                            // Password
                            AuthTextField(
                              controller: _passwordCtrl,
                              label: 'Password',
                              prefixIcon: Icons.lock_outline_rounded,
                              isPassword: true,
                              textInputAction: TextInputAction.done,
                              autofillHints: const [AutofillHints.password],
                              enabled: !isLoading,
                              onFieldSubmitted: _onLogin,
                              validator: (v) {
                                if (v == null || v.isEmpty) return 'Password is required';
                                if (v.length < 6) return 'Minimum 6 characters';
                                return null;
                              },
                            ).animate().fadeIn(duration: 350.ms, delay: 340.ms),

                            // Forgot password
                            Align(
                              alignment: Alignment.centerRight,
                              child: TextButton(
                                onPressed: () => context.push('/forgot-password'),
                                style: TextButton.styleFrom(
                                  padding: const EdgeInsets.symmetric(
                                    vertical: AppSpacing.xs,
                                    horizontal: AppSpacing.xs,
                                  ),
                                  foregroundColor: cs.onPrimary.withValues(alpha: 0.85),
                                  minimumSize: Size.zero,
                                  tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                                ),
                                child: Text(
                                  'Forgot password?',
                                  style: tt.labelSmall?.copyWith(
                                    color: cs.onPrimary.withValues(alpha: 0.85),
                                    decoration: TextDecoration.underline,
                                    decorationColor: cs.onPrimary.withValues(alpha: 0.45),
                                  ),
                                ),
                              ),
                            ).animate().fadeIn(duration: 300.ms, delay: 380.ms),

                            const SizedBox(height: AppSpacing.sm),

                            // Sign In button
                            AuthPrimaryButton(
                              label: 'Sign In',
                              onPressed: _onLogin,
                              isLoading: isLoading,
                            ).animate().fadeIn(duration: 350.ms, delay: 420.ms),

                            const SizedBox(height: AppSpacing.md),

                            // Divider
                            _Divider(color: cs.onPrimary.withValues(alpha: 0.3))
                                .animate().fadeIn(duration: 300.ms, delay: 480.ms),

                            const SizedBox(height: AppSpacing.md),

                            // Google button
                            SocialLoginButton(
                              provider: SocialAuthProvider.google,
                              onTap: _onGoogle,
                              isLoading: _googleLoading,
                            ).animate().fadeIn(duration: 350.ms, delay: 520.ms),

                            const SizedBox(height: AppSpacing.lg),

                            // Sign up link
                            Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Text(
                                  "Don't have an account?  ",
                                  style: tt.bodySmall?.copyWith(
                                    color: cs.onPrimary.withValues(alpha: 0.65),
                                  ),
                                ),
                                GestureDetector(
                                  onTap: () => context.go('/signup'),
                                  child: Text(
                                    'Sign up',
                                    style: tt.bodySmall?.copyWith(
                                      color: cs.onPrimary,
                                      fontWeight: FontWeight.w700,
                                      decoration: TextDecoration.underline,
                                      decorationColor: cs.onPrimary.withValues(alpha: 0.5),
                                    ),
                                  ),
                                ),
                              ],
                            ).animate().fadeIn(duration: 300.ms, delay: 580.ms),
                          ],
                        ),
                      ),
                    ),
                  )
                      .animate()
                      .fadeIn(duration: 700.ms, delay: 180.ms)
                      .slideY(begin: 0.2, end: 0, duration: 600.ms, delay: 180.ms, curve: Curves.easeOutQuint),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ── Divider ───────────────────────────────────────────────────────────────────

class _Divider extends StatelessWidget {
  const _Divider({required this.color});
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(child: Divider(color: color, thickness: 0.8)),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: AppSpacing.sm),
          child: Text(
            'or',
            style: Theme.of(context).textTheme.labelSmall?.copyWith(color: color),
          ),
        ),
        Expanded(child: Divider(color: color, thickness: 0.8)),
      ],
    );
  }
}
