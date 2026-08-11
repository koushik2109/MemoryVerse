import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/navigation/router.dart';
import 'package:memory_verse/core/providers/auth_provider.dart';
import 'package:memory_verse/core/utils/auth_error_handler.dart';
import 'package:memory_verse/features/auth/presentation/auth_widgets.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class SignUpScreen extends ConsumerStatefulWidget {
  const SignUpScreen({super.key});
  @override
  ConsumerState<SignUpScreen> createState() => _SignUpScreenState();
}

class _SignUpScreenState extends ConsumerState<SignUpScreen>
    with SingleTickerProviderStateMixin {
  final _name = TextEditingController();
  final _email = TextEditingController();
  final _password = TextEditingController();
  final _confirmPass = TextEditingController();
  final _nameFocus = FocusNode();
  final _emailFocus = FocusNode();
  final _passFocus = FocusNode();
  final _confirmFocus = FocusNode();

  bool _showPw = false, _showCPw = false, _agreedTerms = false, _isSubmitting = false;
  String? _error, _nameErr, _emailErr, _passErr, _confErr;

  late AnimationController _ec;
  late Animation<double> _fa;

  @override
  void initState() {
    super.initState();
    _ec = AnimationController(vsync: this, duration: const Duration(milliseconds: 450))..forward();
    _fa = CurvedAnimation(parent: _ec, curve: Curves.easeOut);
  }

  @override
  void dispose() {
    for (final c in [_name, _email, _password, _confirmPass]) {
      c.dispose();
    }
    for (final f in [_nameFocus, _emailFocus, _passFocus, _confirmFocus]) {
      f.dispose();
    }
    _ec.dispose();
    super.dispose();
  }

  bool _validate() {
    final nm = _name.text.trim(), em = _email.text.trim(),
          pw = _password.text, cp = _confirmPass.text;

    String? ne, ee, pe, ce;
    if (nm.isEmpty) {
      ne = 'Full name is required.';
    } else if (nm.length < 2) {
      ne = 'Name is too short.';
    }

    if (em.isEmpty) {
      ee = 'Email address is required.';
    } else if (!RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$').hasMatch(em)) {
      ee = 'Please enter a valid email.';
    }

    if (pw.isEmpty) {
      pe = 'Password is required.';
    } else if (pw.length < 8) {
      pe = 'Password must be at least 8 characters.';
    } else if (!RegExp(r'[A-Za-z]').hasMatch(pw) || !RegExp(r'[0-9]').hasMatch(pw)) {
      pe = 'Use a mix of letters and numbers.';
    }

    if (cp.isEmpty) {
      ce = 'Please confirm your password.';
    } else if (pw != cp) {
      ce = 'Passwords do not match.';
    }

    setState(() { _nameErr = ne; _emailErr = ee; _passErr = pe; _confErr = ce; _error = null; });
    return [ne, ee, pe, ce].every((e) => e == null);
  }

  Future<void> _signUp() async {
    if (_isSubmitting) return;
    if (!_validate()) return;
    if (!_agreedTerms) {
      setState(() => _error = 'Please agree to the Terms of Service to continue.');
      return;
    }
    setState(() { _error = null; _isSubmitting = true; });

    await ref.read(authNotifierProvider.notifier).signUp(
      email: _email.text.trim(), password: _password.text, displayName: _name.text.trim());

    if (!mounted) return;
    final s = ref.read(authNotifierProvider);
    if (s.hasError) {
      setState(() { _error = AuthErrorHandler.parse(s.error!); _isSubmitting = false; });
    } else {
      setState(() => _isSubmitting = false);
      if (Supabase.instance.client.auth.currentSession == null && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Verification email sent! Check your inbox.'), behavior: SnackBarBehavior.floating));
        context.push(Routes.otpVerify, extra: _email.text.trim());
      }
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
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
            keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const SizedBox(height: AppSpacing.s56),
              Row(children: [
                AuthBackButton(onTap: () => context.pop(), colors: c),
                const Spacer(),
                AuthLogo(colors: c),
              ]),
              const SizedBox(height: AppSpacing.s32),
              Text('Create your\naccount.', style: TextStyle(fontFamily: 'Inter',
                fontSize: 36, fontWeight: FontWeight.w700, letterSpacing: -1.4, height: 1.05, color: c.text)),
              const SizedBox(height: AppSpacing.s8),
              Text('Begin preserving your most important memories.',
                style: TextStyle(fontFamily: 'Inter', fontSize: 14, color: c.textMuted, height: 1.5)),
              const SizedBox(height: AppSpacing.s36),

              AuthField(label: 'Full Name', hint: 'Alex Johnson', controller: _name, focusNode: _nameFocus,
                textInputAction: TextInputAction.next, onSubmitted: (_) => _emailFocus.requestFocus(),
                colors: c, prefixIcon: Icons.person_outline_rounded),
              _FErr(error: _nameErr),
              const SizedBox(height: AppSpacing.s14),

              AuthField(label: 'Email Address', hint: 'you@example.com', controller: _email, focusNode: _emailFocus,
                keyboardType: TextInputType.emailAddress, textInputAction: TextInputAction.next,
                onSubmitted: (_) => _passFocus.requestFocus(), colors: c, prefixIcon: Icons.mail_outline_rounded),
              _FErr(error: _emailErr),
              const SizedBox(height: AppSpacing.s14),

              AuthField(label: 'Password', hint: '••••••••', controller: _password, focusNode: _passFocus,
                obscureText: !_showPw, textInputAction: TextInputAction.next,
                onSubmitted: (_) => _confirmFocus.requestFocus(), colors: c,
                prefixIcon: Icons.lock_outline_rounded,
                suffixIcon: IconButton(
                  icon: Icon(_showPw ? Icons.visibility_off_outlined : Icons.visibility_outlined, size: 20, color: c.textMuted),
                  onPressed: () => setState(() => _showPw = !_showPw))),
              _FErr(error: _passErr),
              if (_passErr == null && _password.text.isNotEmpty)
                _PwStrength(password: _password.text, colors: c),
              const SizedBox(height: AppSpacing.s14),

              AuthField(label: 'Confirm Password', hint: '••••••••', controller: _confirmPass, focusNode: _confirmFocus,
                obscureText: !_showCPw, textInputAction: TextInputAction.done, onSubmitted: (_) => _signUp(),
                colors: c, prefixIcon: Icons.lock_outline_rounded,
                suffixIcon: IconButton(
                  icon: Icon(_showCPw ? Icons.visibility_off_outlined : Icons.visibility_outlined, size: 20, color: c.textMuted),
                  onPressed: () => setState(() => _showCPw = !_showCPw))),
              _FErr(error: _confErr),
              const SizedBox(height: AppSpacing.s20),

              GestureDetector(
                onTap: () => setState(() => _agreedTerms = !_agreedTerms),
                behavior: HitTestBehavior.opaque,
                child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  AnimatedContainer(duration: AppMotion.fast, width: 20, height: 20,
                    decoration: BoxDecoration(
                      color: _agreedTerms ? c.primary : Colors.transparent,
                      borderRadius: BorderRadius.circular(4),
                      border: Border.all(color: _agreedTerms ? c.primary : c.border, width: 1.5)),
                    child: _agreedTerms ? Icon(Icons.check_rounded, size: 13, color: c.primaryInverse) : null),
                  const SizedBox(width: AppSpacing.s12),
                  Expanded(child: Text.rich(TextSpan(
                    style: TextStyle(fontFamily: 'Inter', fontSize: 13, color: c.textMuted, height: 1.4),
                    children: [
                      const TextSpan(text: 'I agree to the '),
                      TextSpan(text: 'Terms of Service', style: TextStyle(color: c.text, fontWeight: FontWeight.w600)),
                      const TextSpan(text: ' and '),
                      TextSpan(text: 'Privacy Policy', style: TextStyle(color: c.text, fontWeight: FontWeight.w600)),
                    ]))),
                ]),
              ),

              AuthErrorBanner(error: _error, colors: c),
              const SizedBox(height: AppSpacing.s24),
              AuthPrimaryButton(label: 'Create Account', isLoading: _isSubmitting, onPressed: _signUp, colors: c),
              const SizedBox(height: AppSpacing.s28),
              AuthOrDivider(colors: c),
              const SizedBox(height: AppSpacing.s20),

              Row(children: [
                Expanded(child: AuthSocialButton(label: 'Google', icon: const GoogleIcon(),
                  onPressed: () => ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Google sign-up coming soon'))), colors: c)),
                const SizedBox(width: AppSpacing.s12),
                Expanded(child: AuthSocialButton(label: 'Apple',
                  icon: Icon(Icons.apple_rounded, size: 20, color: c.text),
                  onPressed: () => ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Apple sign-up coming soon'))), colors: c)),
              ]),
              const SizedBox(height: AppSpacing.s36),
              Center(child: AuthNavLink(question: 'Already have an account?', action: 'Sign in',
                onTap: () => context.pop(), colors: c)),
              const SizedBox(height: AppSpacing.s32),
            ]),
          ),
        ),
      ),
    );
  }
}

class _FErr extends StatelessWidget {
  final String? error;
  const _FErr({required this.error});
  @override
  Widget build(BuildContext context) {
    return AnimatedSize(
      duration: AppMotion.fast,
      child: error != null
          ? Padding(
              padding: const EdgeInsets.only(top: AppSpacing.s6, left: AppSpacing.s4),
              child: Row(children: [
                const Icon(Icons.info_outline_rounded, size: 12, color: Color(0xFFE5484D)),
                const SizedBox(width: 4),
                Text(error!, style: const TextStyle(fontFamily: 'Inter', fontSize: 12, color: Color(0xFFE5484D))),
              ]))
          : const SizedBox.shrink(),
    );
  }
}

class _PwStrength extends StatelessWidget {
  final String password;
  final AppColors colors;
  const _PwStrength({required this.password, required this.colors});

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

  String _l(int s) => s <= 1 ? 'Weak' : s <= 2 ? 'Fair' : s <= 3 ? 'Good' : 'Strong';

  @override
  Widget build(BuildContext context) {
    final s = _s(); final col = _c(s);
    return Padding(
      padding: const EdgeInsets.only(top: AppSpacing.s8),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: List.generate(5, (i) => Expanded(child: Container(
          height: 3, margin: const EdgeInsets.only(right: 3),
          decoration: BoxDecoration(color: i < s ? col : colors.border,
            borderRadius: BorderRadius.circular(2)))))),
        const SizedBox(height: 4),
        Text('Password strength: ${_l(s)}',
          style: TextStyle(fontFamily: 'Inter', fontSize: 11, color: col, fontWeight: FontWeight.w500)),
      ]),
    );
  }
}
