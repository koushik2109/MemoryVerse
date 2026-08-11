import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/navigation/router.dart';
import 'package:memory_verse/core/providers/auth_provider.dart';
import 'package:memory_verse/core/utils/auth_error_handler.dart';
import 'package:memory_verse/features/auth/presentation/auth_widgets.dart';

class SignInScreen extends ConsumerStatefulWidget {
  const SignInScreen({super.key});

  @override
  ConsumerState<SignInScreen> createState() => _SignInScreenState();
}

class _SignInScreenState extends ConsumerState<SignInScreen>
    with SingleTickerProviderStateMixin {
  final _email = TextEditingController();
  final _password = TextEditingController();
  final _emailFocus = FocusNode();
  final _passwordFocus = FocusNode();

  bool _showPassword = false;
  bool _isSubmitting = false;
  String? _error;

  late AnimationController _entryCtrl;
  late Animation<double> _fadeAnim;
  late Animation<Offset> _slideAnim;

  @override
  void initState() {
    super.initState();
    _entryCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 500))
      ..forward();
    _fadeAnim = CurvedAnimation(parent: _entryCtrl, curve: Curves.easeOut);
    _slideAnim = Tween<Offset>(begin: const Offset(0, 0.06), end: Offset.zero)
        .animate(CurvedAnimation(parent: _entryCtrl, curve: Curves.easeOutCubic));
  }

  @override
  void dispose() {
    _email.dispose();
    _password.dispose();
    _emailFocus.dispose();
    _passwordFocus.dispose();
    _entryCtrl.dispose();
    super.dispose();
  }

  bool _isValidEmail(String e) =>
      RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$').hasMatch(e);

  Future<void> _signIn() async {
    if (_isSubmitting) return;
    setState(() { _error = null; _isSubmitting = true; });

    final email = _email.text.trim();
    final pass = _password.text;

    if (email.isEmpty || pass.isEmpty) {
      setState(() { _error = 'Please enter your email and password.'; _isSubmitting = false; });
      return;
    }
    if (!_isValidEmail(email)) {
      setState(() { _error = 'Please enter a valid email address.'; _isSubmitting = false; });
      return;
    }

    await ref.read(authNotifierProvider.notifier).signIn(email: email, password: pass);

    if (!mounted) return;
    final authState = ref.read(authNotifierProvider);

    if (authState.hasError) {
      final msg = AuthErrorHandler.parse(authState.error!);
      setState(() { _error = msg; _isSubmitting = false; });
      if (authState.error.toString().toLowerCase().contains('email not confirmed')) {
        await Future<void>.delayed(const Duration(milliseconds: 600));
        if (mounted) context.push(Routes.otpVerify, extra: email);
      }
    } else {
      setState(() => _isSubmitting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;

    return Scaffold(
      backgroundColor: c.bg,
      body: SafeArea(
        child: FadeTransition(
          opacity: _fadeAnim,
          child: SlideTransition(
            position: _slideAnim,
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
              keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: AppSpacing.s64),

                  AuthLogo(colors: c),
                  const SizedBox(height: AppSpacing.s32),

                  Text(
                    'Welcome\nback.',
                    style: TextStyle(
                      fontFamily: 'Inter', fontSize: 38,
                      fontWeight: FontWeight.w700, letterSpacing: -1.5,
                      height: 1.05, color: c.text,
                    ),
                  ),
                  const SizedBox(height: AppSpacing.s8),
                  Text(
                    'Sign in to continue your story.',
                    style: TextStyle(fontFamily: 'Inter', fontSize: 15,
                        fontWeight: FontWeight.w400, color: c.textMuted, height: 1.5),
                  ),

                  const SizedBox(height: AppSpacing.s40),

                  AuthField(
                    label: 'Email',
                    hint: 'you@example.com',
                    controller: _email,
                    focusNode: _emailFocus,
                    keyboardType: TextInputType.emailAddress,
                    textInputAction: TextInputAction.next,
                    onSubmitted: (_) => _passwordFocus.requestFocus(),
                    colors: c,
                    prefixIcon: Icons.mail_outline_rounded,
                  ),
                  const SizedBox(height: AppSpacing.s14),

                  AuthField(
                    label: 'Password',
                    hint: '••••••••',
                    controller: _password,
                    focusNode: _passwordFocus,
                    obscureText: !_showPassword,
                    textInputAction: TextInputAction.done,
                    onSubmitted: (_) => _signIn(),
                    colors: c,
                    prefixIcon: Icons.lock_outline_rounded,
                    suffixIcon: IconButton(
                      icon: Icon(
                        _showPassword ? Icons.visibility_off_outlined : Icons.visibility_outlined,
                        size: 20, color: c.textMuted,
                      ),
                      onPressed: () => setState(() => _showPassword = !_showPassword),
                    ),
                  ),

                  Align(
                    alignment: Alignment.centerRight,
                    child: TextButton(
                      onPressed: () => context.push(Routes.forgotPassword),
                      style: TextButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: AppSpacing.s8, horizontal: AppSpacing.s4),
                      ),
                      child: Text(
                        'Forgot password?',
                        style: TextStyle(fontFamily: 'Inter', fontSize: 13,
                            fontWeight: FontWeight.w500, color: c.textMuted),
                      ),
                    ),
                  ),

                  AuthErrorBanner(error: _error, colors: c),
                  const SizedBox(height: AppSpacing.s20),

                  AuthPrimaryButton(label: 'Sign In', isLoading: _isSubmitting,
                      onPressed: _signIn, colors: c),

                  const SizedBox(height: AppSpacing.s28),
                  AuthOrDivider(colors: c),
                  const SizedBox(height: AppSpacing.s20),

                  Row(
                    children: [
                      Expanded(child: AuthSocialButton(
                        label: 'Google', icon: const GoogleIcon(),
                        onPressed: () => _showComingSoon(context, 'Google'), colors: c,
                      )),
                      const SizedBox(width: AppSpacing.s12),
                      Expanded(child: AuthSocialButton(
                        label: 'Apple', icon: Icon(Icons.apple_rounded, size: 20, color: c.text),
                        onPressed: () => _showComingSoon(context, 'Apple'), colors: c,
                      )),
                    ],
                  ),

                  const SizedBox(height: AppSpacing.s40),

                  Center(child: AuthNavLink(
                    question: "Don't have an account?", action: 'Create one',
                    onTap: () => context.push(Routes.signUp), colors: c,
                  )),

                  const SizedBox(height: AppSpacing.s32),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  void _showComingSoon(BuildContext ctx, String provider) {
    ScaffoldMessenger.of(ctx).showSnackBar(SnackBar(
      content: Text('$provider sign-in coming soon'),
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(AppRadii.md)),
    ));
  }
}
