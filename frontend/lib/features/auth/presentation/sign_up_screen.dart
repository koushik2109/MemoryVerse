import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart';
import 'package:memory_verse/core/navigation/router.dart';
import 'package:memory_verse/core/providers/auth_provider.dart';
import 'package:memory_verse/core/utils/auth_error_handler.dart';
import 'package:memory_verse/features/auth/presentation/auth_widgets.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:memory_verse/core/presentation/widgets/aurora_background.dart';
import 'package:memory_verse/core/presentation/widgets/flow_button.dart';

class SignUpScreen extends ConsumerStatefulWidget {
  const SignUpScreen({super.key});
  @override
  ConsumerState<SignUpScreen> createState() => _SignUpScreenState();
}

class _SignUpScreenState extends ConsumerState<SignUpScreen>
    with SingleTickerProviderStateMixin {
  final _email = TextEditingController();
  final _password = TextEditingController();
  final _confirmPass = TextEditingController();
  final _emailFocus = FocusNode();
  final _passFocus = FocusNode();
  final _confirmFocus = FocusNode();

  bool _showPw = false,
      _showCPw = false,
      _agreedTerms = false,
      _isSubmitting = false;
  String? _error, _emailErr, _passErr, _confErr;

  late AnimationController _ec;
  late Animation<double> _fa;

  @override
  void initState() {
    super.initState();
    _ec = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 450),
    )..forward();
    _fa = CurvedAnimation(parent: _ec, curve: Curves.easeOut);
  }

  @override
  void dispose() {
    for (final c in [_email, _password, _confirmPass]) {
      c.dispose();
    }
    for (final f in [_emailFocus, _passFocus, _confirmFocus]) {
      f.dispose();
    }
    _ec.dispose();
    super.dispose();
  }

  bool _validate() {
    final em = _email.text.trim(), pw = _password.text, cp = _confirmPass.text;

    String? ee, pe, ce;

    if (em.isEmpty) {
      ee = 'Email address is required.';
    } else if (!RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$').hasMatch(em)) {
      ee = 'Please enter a valid email.';
    }

    if (pw.isEmpty) {
      pe = 'Password is required.';
    } else if (pw.length < 8) {
      pe = 'Password must be at least 8 characters.';
    } else if (!RegExp(r'[A-Za-z]').hasMatch(pw) ||
        !RegExp(r'[0-9]').hasMatch(pw)) {
      pe = 'Use a mix of letters and numbers.';
    }

    if (cp.isEmpty) {
      ce = 'Please confirm your password.';
    } else if (pw != cp) {
      ce = 'Passwords do not match.';
    }

    setState(() {
      _emailErr = ee;
      _passErr = pe;
      _confErr = ce;
      _error = null;
    });
    return [ee, pe, ce].every((e) => e == null);
  }

  Future<void> _signUp() async {
    if (_isSubmitting) return;
    if (!_validate()) return;
    if (!_agreedTerms) {
      setState(
        () => _error = 'Please agree to the Terms of Service to continue.',
      );
      return;
    }
    setState(() {
      _error = null;
      _isSubmitting = true;
    });

    await ref
        .read(authNotifierProvider.notifier)
        .signUp(email: _email.text.trim(), password: _password.text);

    if (!mounted) return;
    final s = ref.read(authNotifierProvider);
    if (s.hasError) {
      setState(() {
        _error = AuthErrorHandler.parse(s.error!);
        _isSubmitting = false;
      });
    } else {
      setState(() => _isSubmitting = false);
      if (Supabase.instance.client.auth.currentSession == null && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Verification email sent! Check your inbox.'),
            behavior: SnackBarBehavior.floating,
          ),
        );
        context.push(Routes.otpVerify, extra: _email.text.trim());
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(
        children: [
          const AuroraBackground(isDark: true),
          SafeArea(
            child: FadeTransition(
              opacity: _fa,
              child: SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
                keyboardDismissBehavior:
                    ScrollViewKeyboardDismissBehavior.onDrag,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: AppSpacing.s16),
                    Row(
                      children: [
                        AuthBackButton(onTap: () => context.pop()),
                        const Spacer(),
                        const AuthLogo(),
                      ],
                    ),
                    const SizedBox(height: AppSpacing.s24),
                    const Text(
                      'Create your\naccount.',
                      style: TextStyle(
                        fontFamily: 'Inter',
                        fontSize: 32,
                        fontWeight: FontWeight.w700,
                        letterSpacing: -1.2,
                        height: 1.05,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: AppSpacing.s8),
                    const Text(
                      'Begin preserving your most important memories.',
                      style: TextStyle(
                        fontFamily: 'Inter',
                        fontSize: 13,
                        color: Colors.white70,
                        height: 1.5,
                      ),
                    ),
                    const SizedBox(height: AppSpacing.s24),

                    AuthField(
                      label: 'Email Address',
                      hint: 'you@example.com',
                      controller: _email,
                      focusNode: _emailFocus,
                      keyboardType: TextInputType.emailAddress,
                      textInputAction: TextInputAction.next,
                      onSubmitted: (_) => _passFocus.requestFocus(),
                      prefixIcon: Icons.mail_outline_rounded,
                    ),
                    _FErr(error: _emailErr),
                    const SizedBox(height: AppSpacing.s12),

                    AuthField(
                      label: 'Password',
                      hint: '••••••••',
                      controller: _password,
                      focusNode: _passFocus,
                      obscureText: !_showPw,
                      textInputAction: TextInputAction.next,
                      onSubmitted: (_) => _confirmFocus.requestFocus(),
                      prefixIcon: Icons.lock_outline_rounded,
                      suffixIcon: IconButton(
                        icon: Icon(
                          _showPw
                              ? Icons.visibility_off_outlined
                              : Icons.visibility_outlined,
                          size: 20,
                          color: AppColors.onDarkMuted,
                        ),
                        onPressed: () => setState(() => _showPw = !_showPw),
                      ),
                    ),
                    _FErr(error: _passErr),
                    if (_passErr == null && _password.text.isNotEmpty)
                      _PwStrength(password: _password.text),
                    const SizedBox(height: AppSpacing.s12),

                    AuthField(
                      label: 'Confirm Password',
                      hint: '••••••••',
                      controller: _confirmPass,
                      focusNode: _confirmFocus,
                      obscureText: !_showCPw,
                      textInputAction: TextInputAction.done,
                      onSubmitted: (_) => _signUp(),
                      prefixIcon: Icons.lock_outline_rounded,
                      suffixIcon: IconButton(
                        icon: Icon(
                          _showCPw
                              ? Icons.visibility_off_outlined
                              : Icons.visibility_outlined,
                          size: 20,
                          color: AppColors.onDarkMuted,
                        ),
                        onPressed: () => setState(() => _showCPw = !_showCPw),
                      ),
                    ),
                    _FErr(error: _confErr),
                    const SizedBox(height: AppSpacing.s16),

                    GestureDetector(
                      onTap: () => setState(() => _agreedTerms = !_agreedTerms),
                      behavior: HitTestBehavior.opaque,
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          AnimatedContainer(
                            duration: const Duration(milliseconds: 200),
                            width: 20,
                            height: 20,
                            decoration: BoxDecoration(
                              color: _agreedTerms
                                  ? Colors.white
                                  : Colors.transparent,
                              borderRadius: BorderRadius.circular(4),
                              border: Border.all(
                                color: _agreedTerms
                                    ? Colors.white
                                    : Colors.white.withOpacity(0.3),
                                width: 1.5,
                              ),
                            ),
                            child: _agreedTerms
                                ? const Icon(
                                    Icons.check_rounded,
                                    size: 14,
                                    color: AppColors.plum800,
                                  )
                                : null,
                          ),
                          const SizedBox(width: AppSpacing.s12),
                          Expanded(
                            child: Text.rich(
                              TextSpan(
                                style: AppTextStyles.caption.copyWith(
                                  color: AppColors.onDarkMuted,
                                  height: 1.4,
                                ),
                                children: [
                                  const TextSpan(text: 'I agree to the '),
                                  TextSpan(
                                    text: 'Terms of Service',
                                    style: AppTextStyles.caption.copyWith(
                                      color: AppColors.onDarkPrimary,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                  const TextSpan(text: ' and '),
                                  TextSpan(
                                    text: 'Privacy Policy',
                                    style: AppTextStyles.caption.copyWith(
                                      color: AppColors.onDarkPrimary,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),

                    AuthErrorBanner(error: _error),
                    const SizedBox(height: AppSpacing.s24),
                    Center(
                      child: FlowButton(
                        text: _isSubmitting ? 'Creating...' : 'Create Account',
                        isDark: true,
                        onPressed: _signUp,
                      ),
                    ),
                    const SizedBox(height: AppSpacing.s20),
                    const AuthOrDivider(),
                    const SizedBox(height: AppSpacing.s16),

                    AuthSocialButton(
                      label: 'Continue with Google',
                      icon: const GoogleIcon(),
                      onPressed: _signInWithGoogle,
                    ),
                    const SizedBox(height: AppSpacing.s24),
                    Center(
                      child: AuthNavLink(
                        question: 'Already have an account?',
                        action: 'Sign in',
                        onTap: () => context.pop(),
                      ),
                    ),
                    const SizedBox(height: AppSpacing.s16),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _signInWithGoogle() async {
    setState(() => _error = null);
    await ref.read(authNotifierProvider.notifier).signInWithGoogle();
    if (!mounted) return;
    final authState = ref.read(authNotifierProvider);
    if (authState.hasError) {
      final err = authState.error.toString();
      if (!err.contains('cancelled')) {
        setState(() {
          _error = AuthErrorHandler.parse(authState.error!);
        });
      }
    }
  }
}

class _FErr extends StatelessWidget {
  final String? error;
  const _FErr({required this.error});
  @override
  Widget build(BuildContext context) {
    return AnimatedSize(
      duration: const Duration(milliseconds: 200),
      child: error != null
          ? Padding(
              padding: const EdgeInsets.only(
                top: AppSpacing.s8,
                left: AppSpacing.s4,
              ),
              child: Row(
                children: [
                  const Icon(
                    Icons.info_outline_rounded,
                    size: 12,
                    color: Color(0xFFE5484D),
                  ),
                  const SizedBox(width: 4),
                  Text(
                    error!,
                    style: const TextStyle(
                      fontFamily: 'Inter',
                      fontSize: 12,
                      color: Color(0xFFE5484D),
                    ),
                  ),
                ],
              ),
            )
          : const SizedBox.shrink(),
    );
  }
}

class _PwStrength extends StatelessWidget {
  final String password;
  const _PwStrength({required this.password});

  int _s() {
    int s = 0;
    if (password.length >= 8) s++;
    if (password.length >= 12) s++;
    if (RegExp(r'[A-Z]').hasMatch(password)) s++;
    if (RegExp(r'[0-9]').hasMatch(password)) s++;
    if (RegExp(r'[!@#\$%^&*(),.?":{}|<>]').hasMatch(password)) s++;
    return s;
  }

  Color _c(int s) {
    if (s <= 1) return const Color(0xFFE5484D);
    if (s <= 2) return const Color(0xFFE5A44D);
    if (s <= 3) return const Color(0xFF30A46C);
    return const Color(0xFF1D9A5E);
  }

  String _l(int s) => s <= 1
      ? 'Weak'
      : s <= 2
      ? 'Fair'
      : s <= 3
      ? 'Good'
      : 'Strong';

  @override
  Widget build(BuildContext context) {
    final s = _s();
    final col = _c(s);
    return Padding(
      padding: const EdgeInsets.only(top: AppSpacing.s8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: List.generate(
              5,
              (i) => Expanded(
                child: Container(
                  height: 3,
                  margin: const EdgeInsets.only(right: 3),
                  decoration: BoxDecoration(
                    color: i < s ? col : Colors.white.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Password strength: ${_l(s)}',
            style: TextStyle(
              fontFamily: 'Inter',
              fontSize: 11,
              color: col,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}
