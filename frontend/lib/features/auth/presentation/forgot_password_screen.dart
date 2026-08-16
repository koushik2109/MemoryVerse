import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/auth_provider.dart';
import 'package:memory_verse/core/utils/auth_error_handler.dart';
import 'package:memory_verse/features/auth/presentation/auth_widgets.dart';

class ForgotPasswordScreen extends ConsumerStatefulWidget {
  const ForgotPasswordScreen({super.key});
  @override
  ConsumerState<ForgotPasswordScreen> createState() => _ForgotPasswordScreenState();
}

class _ForgotPasswordScreenState extends ConsumerState<ForgotPasswordScreen>
    with SingleTickerProviderStateMixin {
  final _email = TextEditingController();
  String? _error;
  final bool _sent = false;
  bool _isSubmitting = false;

  late AnimationController _ec;
  late Animation<double> _fa;

  @override
  void initState() {
    super.initState();
    _ec = AnimationController(vsync: this, duration: const Duration(milliseconds: 400))..forward();
    _fa = CurvedAnimation(parent: _ec, curve: Curves.easeOut);
  }

  @override
  void dispose() {
    _email.dispose();
    _ec.dispose();
    super.dispose();
  }

  Future<void> _sendReset() async {
    if (_isSubmitting) return;
    final email = _email.text.trim();
    if (email.isEmpty) { setState(() => _error = 'Please enter your email address.'); return; }
    if (!RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$').hasMatch(email)) {
      setState(() => _error = 'Please enter a valid email address.'); return;
    }
    setState(() { _error = null; _isSubmitting = true; });

    await ref.read(authNotifierProvider.notifier).sendPasswordReset(email);
    if (!mounted) return;
    final s = ref.read(authNotifierProvider);
    if (s.hasError) {
      setState(() { _error = AuthErrorHandler.parse(s.error!); _isSubmitting = false; });
    } else {
      setState(() { _isSubmitting = false; });
      context.push('/reset-password', extra: _email.text.trim());
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Scaffold(
      backgroundColor: c.bg,
      body: SafeArea(
        child: FadeTransition(
          opacity: _fa,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const SizedBox(height: AppSpacing.s56),
              Row(children: [
                AuthBackButton(onTap: () => context.pop(), colors: c),
                const Spacer(),
                AuthLogo(colors: c),
              ]),
              const SizedBox(height: AppSpacing.s40),
              AnimatedSwitcher(
                duration: AppMotion.normal,
                transitionBuilder: (child, anim) => FadeTransition(
                  opacity: anim,
                  child: SlideTransition(
                    position: Tween<Offset>(begin: const Offset(0, 0.05), end: Offset.zero).animate(anim),
                    child: child),
                ),
                child: _sent
                    ? _SuccessView(key: const ValueKey('s'), email: _email.text.trim(),
                        colors: c, onBack: () => context.pop())
                    : _FormView(key: const ValueKey('f'), emailCtrl: _email, error: _error,
                        isSubmitting: _isSubmitting, colors: c, onSubmit: _sendReset),
              ),
            ]),
          ),
        ),
      ),
    );
  }
}

class _FormView extends StatelessWidget {
  final TextEditingController emailCtrl;
  final String? error;
  final bool isSubmitting;
  final AppColors colors;
  final VoidCallback onSubmit;
  const _FormView({super.key, required this.emailCtrl, required this.error,
    required this.isSubmitting, required this.colors, required this.onSubmit});

  @override
  Widget build(BuildContext context) {
    final c = colors;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Container(width: 52, height: 52,
        decoration: BoxDecoration(color: c.surface, borderRadius: BorderRadius.circular(AppRadii.md),
          border: Border.all(color: c.border)),
        child: Icon(Icons.lock_reset_rounded, color: c.text, size: 24)),
      const SizedBox(height: AppSpacing.s24),
      Text('Forgot your\npassword?', style: TextStyle(fontFamily: 'Inter', fontSize: 34,
        fontWeight: FontWeight.w700, letterSpacing: -1.2, height: 1.1, color: c.text)),
      const SizedBox(height: AppSpacing.s10),
      Text("No worries. Enter your email and we'll send you a reset link.",
        style: TextStyle(fontFamily: 'Inter', fontSize: 14, color: c.textMuted, height: 1.5)),
      const SizedBox(height: AppSpacing.s36),
      AuthField(label: 'Email Address', hint: 'you@example.com', controller: emailCtrl,
        keyboardType: TextInputType.emailAddress, textInputAction: TextInputAction.done,
        onSubmitted: (_) => onSubmit(), colors: c, prefixIcon: Icons.mail_outline_rounded),
      AuthErrorBanner(error: error, colors: c),
      const SizedBox(height: AppSpacing.s24),
      AuthPrimaryButton(label: 'Send Reset Link', isLoading: isSubmitting, onPressed: onSubmit, colors: c),
    ]);
  }
}

class _SuccessView extends StatelessWidget {
  final String email;
  final AppColors colors;
  final VoidCallback onBack;
  const _SuccessView({super.key, required this.email, required this.colors, required this.onBack});

  @override
  Widget build(BuildContext context) {
    final c = colors;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Container(width: 52, height: 52,
        decoration: BoxDecoration(color: const Color(0xFF30A46C).withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(AppRadii.md)),
        child: const Icon(Icons.mark_email_read_rounded, color: Color(0xFF30A46C), size: 26)),
      const SizedBox(height: AppSpacing.s24),
      Text('Check your\ninbox.', style: TextStyle(fontFamily: 'Inter', fontSize: 34,
        fontWeight: FontWeight.w700, letterSpacing: -1.2, height: 1.1, color: c.text)),
      const SizedBox(height: AppSpacing.s10),
      Text.rich(TextSpan(
        style: TextStyle(fontFamily: 'Inter', fontSize: 14, color: c.textMuted, height: 1.6),
        children: [
          const TextSpan(text: "We've sent a password reset link to "),
          TextSpan(text: email, style: TextStyle(color: c.text, fontWeight: FontWeight.w500)),
          const TextSpan(text: '.'),
        ])),
      const SizedBox(height: AppSpacing.s12),
      Text("Didn't receive it? Check your spam folder.",
        style: TextStyle(fontFamily: 'Inter', fontSize: 13, color: c.textMuted.withValues(alpha: 0.7))),
      const SizedBox(height: AppSpacing.s40),
      AuthPrimaryButton(label: 'Back to Sign In', isLoading: false, onPressed: onBack, colors: c),
    ]);
  }
}
