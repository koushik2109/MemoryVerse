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
import 'package:frontend/shared/theme/app_color_tokens.dart';

class SignupPage extends ConsumerStatefulWidget {
  const SignupPage({super.key});

  @override
  ConsumerState<SignupPage> createState() => _SignupPageState();
}

class _SignupPageState extends ConsumerState<SignupPage> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _emailCtrl = TextEditingController();
  final _passwordCtrl = TextEditingController();
  final _confirmCtrl = TextEditingController();
  bool _termsAccepted = false;
  double _passwordStrength = 0;

  @override
  void initState() {
    super.initState();
    _passwordCtrl.addListener(_updateStrength);
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _emailCtrl.dispose();
    _passwordCtrl.dispose();
    _confirmCtrl.dispose();
    super.dispose();
  }

  void _updateStrength() {
    final pw = _passwordCtrl.text;
    double s = 0;
    if (pw.length >= 8) s += 0.25;
    if (pw.contains(RegExp(r'[A-Z]'))) s += 0.25;
    if (pw.contains(RegExp(r'[0-9]'))) s += 0.25;
    if (pw.contains(RegExp(r'[!@#\$%^&*(),.?":{}|<>]'))) s += 0.25;
    if (mounted) setState(() => _passwordStrength = s);
  }

  Future<void> _onSignup() async {
    if (!_formKey.currentState!.validate()) return;
    if (!_termsAccepted) {
      _showInfo('Please accept the Terms of Service to continue.');
      return;
    }
    FocusScope.of(context).unfocus();
    await ref.read(authProvider.notifier).signup(
          name: _nameCtrl.text.trim(),
          email: _emailCtrl.text.trim(),
          password: _passwordCtrl.text,
        );
    final authState = ref.read(authProvider);
    if (!mounted) return;
    if (authState.isAuthenticated) {
      context.go('/home');
    } else if (authState.pendingEmail != null) {
      // Supabase sent confirmation email — inform user
      _showInfo('Check your inbox to confirm your email, then sign in.');
      context.go('/login');
    } else if (authState.errorMessage != null) {
      _showError(authState.errorMessage!);
    }
  }

  void _showError(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(msg),
      backgroundColor: Theme.of(context).colorScheme.error,
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(AppRadius.md)),
    ));
  }

  void _showInfo(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(msg),
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(AppRadius.md)),
    ));
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
                vertical: AppSpacing.xl,
                horizontal: AppSpacing.md,
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // ── Brand ───────────────────────────────────────────────
                  Image.asset(
                    isDark
                        ? 'assets/icons/memoryverse_darktheme.png'
                        : 'assets/icons/memoryverse_lighttheme.png',
                    width: 64,
                    height: 64,
                    errorBuilder: (_, e, s) => Icon(
                      Icons.auto_stories_rounded,
                      size: 56,
                      color: cs.onPrimary,
                    ),
                  )
                      .animate()
                      .fadeIn(duration: 500.ms)
                      .scale(begin: const Offset(0.7, 0.7), duration: 500.ms, curve: Curves.easeOutBack),

                  const SizedBox(height: AppSpacing.sm),

                  Text(
                    'Create Account',
                    style: tt.headlineMedium?.copyWith(
                      color: cs.onPrimary,
                      fontWeight: FontWeight.w800,
                      letterSpacing: -0.8,
                    ),
                  ).animate().fadeIn(duration: 400.ms, delay: 100.ms),

                  const SizedBox(height: AppSpacing.xs),
                  Text(
                    'Start capturing your memories today',
                    style: tt.bodySmall?.copyWith(
                      color: cs.onPrimary.withValues(alpha: 0.7),
                    ),
                  ).animate().fadeIn(duration: 400.ms, delay: 160.ms),

                  const SizedBox(height: AppSpacing.lg),

                  // ── Card ────────────────────────────────────────────────
                  SizedBox(
                    width: cardWidth,
                    child: AuthGlassCard(
                      child: Form(
                        key: _formKey,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            // Name
                            AuthTextField(
                              controller: _nameCtrl,
                              label: 'Full name',
                              prefixIcon: Icons.person_outline_rounded,
                              keyboardType: TextInputType.name,
                              enabled: !isLoading,
                              autofillHints: const [AutofillHints.name],
                              validator: (v) {
                                if (v == null || v.trim().isEmpty) return 'Name is required';
                                return null;
                              },
                            ).animate().fadeIn(duration: 350.ms, delay: 260.ms),

                            const SizedBox(height: AppSpacing.md),

                            // Email
                            AuthTextField(
                              controller: _emailCtrl,
                              label: 'Email address',
                              prefixIcon: Icons.email_outlined,
                              keyboardType: TextInputType.emailAddress,
                              enabled: !isLoading,
                              autofillHints: const [AutofillHints.email],
                              validator: (v) {
                                if (v == null || v.isEmpty) return 'Email is required';
                                if (!v.contains('@')) return 'Enter a valid email';
                                return null;
                              },
                            ).animate().fadeIn(duration: 350.ms, delay: 310.ms),

                            const SizedBox(height: AppSpacing.md),

                            // Password
                            AuthTextField(
                              controller: _passwordCtrl,
                              label: 'Password',
                              prefixIcon: Icons.lock_outline_rounded,
                              isPassword: true,
                              enabled: !isLoading,
                              autofillHints: const [AutofillHints.newPassword],
                              validator: (v) {
                                if (v == null || v.isEmpty) return 'Password is required';
                                if (v.length < 8) return 'Minimum 8 characters';
                                return null;
                              },
                            ).animate().fadeIn(duration: 350.ms, delay: 360.ms),

                            const SizedBox(height: AppSpacing.sm),

                            // Strength bar
                            _StrengthBar(strength: _passwordStrength)
                                .animate().fadeIn(duration: 300.ms, delay: 400.ms),

                            const SizedBox(height: AppSpacing.md),

                            // Confirm password
                            AuthTextField(
                              controller: _confirmCtrl,
                              label: 'Confirm password',
                              prefixIcon: Icons.lock_reset_rounded,
                              isPassword: true,
                              textInputAction: TextInputAction.done,
                              enabled: !isLoading,
                              onFieldSubmitted: _onSignup,
                              validator: (v) {
                                if (v != _passwordCtrl.text) return 'Passwords do not match';
                                return null;
                              },
                            ).animate().fadeIn(duration: 350.ms, delay: 420.ms),

                            const SizedBox(height: AppSpacing.md),

                            // Terms
                            _TermsRow(
                              value: _termsAccepted,
                              onChanged: (v) => setState(() => _termsAccepted = v ?? false),
                            ).animate().fadeIn(duration: 300.ms, delay: 460.ms),

                            const SizedBox(height: AppSpacing.md),

                            // Create account button
                            AuthPrimaryButton(
                              label: 'Create Account',
                              onPressed: _onSignup,
                              isLoading: isLoading,
                            ).animate().fadeIn(duration: 350.ms, delay: 520.ms),

                            const SizedBox(height: AppSpacing.md),

                            _Divider(color: cs.onPrimary.withValues(alpha: 0.3))
                                .animate().fadeIn(duration: 300.ms, delay: 560.ms),

                            const SizedBox(height: AppSpacing.md),

                            SocialLoginButton(
                              provider: SocialAuthProvider.google,
                              onTap: () => ref.read(authProvider.notifier).signInWithGoogle(),
                            ).animate().fadeIn(duration: 350.ms, delay: 600.ms),

                            const SizedBox(height: AppSpacing.lg),

                            Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Text(
                                  'Already have an account?  ',
                                  style: tt.bodySmall?.copyWith(
                                    color: cs.onPrimary.withValues(alpha: 0.65),
                                  ),
                                ),
                                GestureDetector(
                                  onTap: () => context.go('/login'),
                                  child: Text(
                                    'Sign in',
                                    style: tt.bodySmall?.copyWith(
                                      color: cs.onPrimary,
                                      fontWeight: FontWeight.w700,
                                      decoration: TextDecoration.underline,
                                      decorationColor: cs.onPrimary.withValues(alpha: 0.5),
                                    ),
                                  ),
                                ),
                              ],
                            ).animate().fadeIn(duration: 300.ms, delay: 640.ms),
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

// ── Internal widgets ──────────────────────────────────────────────────────────

class _StrengthBar extends StatelessWidget {
  const _StrengthBar({required this.strength});
  final double strength;

  String get _label => switch (strength) {
        0 => '',
        <= 0.25 => 'Weak',
        <= 0.5 => 'Fair',
        <= 0.75 => 'Good',
        _ => 'Strong',
      };

  Color _color(ColorScheme cs) => switch (strength) {
        0 => cs.outlineVariant,
        <= 0.25 => cs.error,
        <= 0.5 => cs.tertiary,
        <= 0.75 => cs.primary,
        _ => cs.strengthStrong,
      };

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;
    final color = _color(cs);
    return Row(
      children: [
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(AppRadius.full),
            child: LinearProgressIndicator(
              value: strength,
              minHeight: 4,
              backgroundColor: cs.outlineVariant.withValues(alpha: 0.3),
              valueColor: AlwaysStoppedAnimation<Color>(color),
            ),
          ),
        ),
        if (_label.isNotEmpty) ...[
          const SizedBox(width: AppSpacing.sm),
          Text(
            _label,
            style: tt.labelSmall?.copyWith(color: color, fontWeight: FontWeight.w600),
          ),
        ],
      ],
    );
  }
}

class _TermsRow extends StatelessWidget {
  const _TermsRow({required this.value, required this.onChanged});
  final bool value;
  final ValueChanged<bool?> onChanged;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        SizedBox(
          width: 20,
          height: 20,
          child: Checkbox(
            value: value,
            onChanged: onChanged,
            materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
            side: BorderSide(color: cs.onPrimary.withValues(alpha: 0.5)),
          ),
        ),
        const SizedBox(width: AppSpacing.sm),
        Expanded(
          child: Text.rich(
            TextSpan(
              style: tt.labelSmall?.copyWith(color: cs.onPrimary.withValues(alpha: 0.7)),
              children: [
                const TextSpan(text: 'I agree to the '),
                TextSpan(
                  text: 'Terms of Service',
                  style: tt.labelSmall?.copyWith(
                    color: cs.onPrimary,
                    fontWeight: FontWeight.w700,
                    decoration: TextDecoration.underline,
                  ),
                ),
                const TextSpan(text: ' & '),
                TextSpan(
                  text: 'Privacy Policy',
                  style: tt.labelSmall?.copyWith(
                    color: cs.onPrimary,
                    fontWeight: FontWeight.w700,
                    decoration: TextDecoration.underline,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

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
