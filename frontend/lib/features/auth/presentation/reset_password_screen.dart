import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart';
import 'package:memory_verse/core/navigation/router.dart';
import 'package:memory_verse/core/providers/auth_provider.dart';
import 'package:memory_verse/core/utils/auth_error_handler.dart';
import 'package:memory_verse/features/auth/presentation/auth_widgets.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;

class ResetPasswordScreen extends ConsumerStatefulWidget {
  final String email;
  const ResetPasswordScreen({super.key, required this.email});
  @override
  ConsumerState<ResetPasswordScreen> createState() =>
      _ResetPasswordScreenState();
}

class _ResetPasswordScreenState extends ConsumerState<ResetPasswordScreen>
    with SingleTickerProviderStateMixin {
  final _otpCtrl = TextEditingController();
  final _passCtrl = TextEditingController();
  final _otpFocus = FocusNode();
  final _passFocus = FocusNode();

  String? _error;
  bool _isSubmitting = false, _showPw = false;

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
    WidgetsBinding.instance.addPostFrameCallback(
      (_) => _otpFocus.requestFocus(),
    );
  }

  @override
  void dispose() {
    _otpCtrl.dispose();
    _passCtrl.dispose();
    _otpFocus.dispose();
    _passFocus.dispose();
    _ec.dispose();
    super.dispose();
  }

  Future<void> _reset() async {
    if (_isSubmitting) return;

    final otp = _otpCtrl.text.trim();
    final pw = _passCtrl.text;

    if (otp.length < 6) {
      setState(() => _error = 'Please enter the complete 6-digit code.');
      return;
    }
    if (pw.length < 8) {
      setState(() => _error = 'Password must be at least 8 characters.');
      return;
    }

    setState(() {
      _error = null;
      _isSubmitting = true;
    });

    await ref
        .read(authNotifierProvider.notifier)
        .resetPassword(email: widget.email, token: otp, newPassword: pw);

    if (!mounted) return;
    final s = ref.read(authNotifierProvider);
    if (s.hasError) {
      setState(() {
        _error = AuthErrorHandler.parse(s.error!);
        _isSubmitting = false;
      });
      _otpFocus.requestFocus();
    } else {
      setState(() => _isSubmitting = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text('Password reset successfully. Please log in.'),
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(AppRadius.md),
          ),
        ),
      );
      context.go(Routes.signIn);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(
        children: [
          const Positioned.fill(
            child: DecoratedBox(
              decoration: BoxDecoration(gradient: AppGradients.dark),
            ),
          ),
          SafeArea(
            child: FadeTransition(
          opacity: _fa,
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
            keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: AppSpacing.s48),
                Row(
                  children: [
                    AuthBackButton(onTap: () => context.pop()),
                    const Spacer(),
                    const AuthLogo(),
                  ],
                ),
                const SizedBox(height: AppSpacing.s40),
                const Text(
                  'Reset your\npassword.',
                  style: TextStyle(
                    fontSize: 34,
                    fontWeight: FontWeight.w700,
                    letterSpacing: -1.2,
                    height: 1.1,
                    color: adt.AppColors.onDarkPrimary,
                  ),
                ),
                const SizedBox(height: AppSpacing.s12),
                Text.rich(
                  TextSpan(
                    style: const TextStyle(
                      fontSize: 14,
                      color: adt.AppColors.onDarkSecondary,
                      height: 1.5,
                    ),
                    children: [
                      const TextSpan(text: 'We sent a 6-digit code to '),
                      TextSpan(
                        text: widget.email,
                        style: const TextStyle(
                          color: adt.AppColors.onDarkPrimary,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      const TextSpan(
                        text: '. Enter it below along with your new password.',
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: AppSpacing.s32),

                AuthField(
                  label: '6-Digit Reset Code',
                  hint: '123456',
                  controller: _otpCtrl,
                  focusNode: _otpFocus,
                  keyboardType: TextInputType.number,
                  textInputAction: TextInputAction.next,
                  onSubmitted: (_) => _passFocus.requestFocus(),
                  prefixIcon: Icons.pin_outlined,
                ),
                const SizedBox(height: AppSpacing.s20),

                AuthField(
                  label: 'New Password',
                  hint: '••••••••',
                  controller: _passCtrl,
                  focusNode: _passFocus,
                  obscureText: !_showPw,
                  textInputAction: TextInputAction.done,
                  onSubmitted: (_) => _reset(),
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

                AuthErrorBanner(error: _error),
                const SizedBox(height: AppSpacing.s32),
                AuthPrimaryButton(
                  label: 'Reset Password',
                  isLoading: _isSubmitting,
                  onPressed: _reset,
                ),
                const SizedBox(height: AppSpacing.s32),
              ],
            ),
          ),
        ),
      ),
        ],
      ),
    );
  }
}
