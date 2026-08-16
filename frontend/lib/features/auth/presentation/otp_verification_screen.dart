import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/auth_provider.dart';
import 'package:memory_verse/core/utils/auth_error_handler.dart';
import 'package:memory_verse/features/auth/presentation/auth_widgets.dart';

class OtpVerificationScreen extends ConsumerStatefulWidget {
  final String email;
  const OtpVerificationScreen({super.key, required this.email});
  @override
  ConsumerState<OtpVerificationScreen> createState() => _OtpVerificationScreenState();
}

class _OtpVerificationScreenState extends ConsumerState<OtpVerificationScreen>
    with SingleTickerProviderStateMixin {
  final List<TextEditingController> _ctrls = List.generate(6, (_) => TextEditingController());
  final List<FocusNode> _foci = List.generate(6, (_) => FocusNode());

  String? _error;
  bool _isSubmitting = false, _resendCooldown = false;
  int _resendSeconds = 0;

  late AnimationController _ec;
  late Animation<double> _fa;

  @override
  void initState() {
    super.initState();
    _ec = AnimationController(vsync: this, duration: const Duration(milliseconds: 450))..forward();
    _fa = CurvedAnimation(parent: _ec, curve: Curves.easeOut);
    WidgetsBinding.instance.addPostFrameCallback((_) => _foci[0].requestFocus());
  }

  @override
  void dispose() {
    for (final c in _ctrls) {
      c.dispose();
    }
    for (final f in _foci) {
      f.dispose();
    }
    _ec.dispose();
    super.dispose();
  }

  String get _otp => _ctrls.map((c) => c.text).join();

  Future<void> _verify() async {
    if (_isSubmitting) return;
    if (_otp.length < 6) {
      setState(() => _error = 'Please enter the complete 6-digit code.');
      return;
    }
    setState(() { _error = null; _isSubmitting = true; });

    await ref.read(authNotifierProvider.notifier).verifyOtp(email: widget.email, token: _otp);

    if (!mounted) return;
    final s = ref.read(authNotifierProvider);
    if (s.hasError) {
      setState(() { _error = AuthErrorHandler.parse(s.error!); _isSubmitting = false; });
      for (final c in _ctrls) {
        c.clear();
      }
      _foci[0].requestFocus();
    } else {
      setState(() => _isSubmitting = false);
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: const Text('Email verified successfully. Please log in.'),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(AppRadii.md))));
      context.go('/sign-in');
    }
  }

  Future<void> _resend() async {
    if (_resendCooldown) return;
    setState(() { _resendCooldown = true; _resendSeconds = 30; _error = null; });
    try {
      await ref.read(authNotifierProvider.notifier).resendOtp(email: widget.email);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: const Text('New code sent!'), behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(AppRadii.md))));
      }
    } catch (e) {
      if (mounted) setState(() => _error = AuthErrorHandler.parse(e));
    }
    _tick();
  }

  void _tick() {
    if (!mounted || _resendSeconds <= 0) { if (mounted) setState(() => _resendCooldown = false); return; }
    Future.delayed(const Duration(seconds: 1), () {
      if (mounted) { setState(() => _resendSeconds--); _tick(); }
    });
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
              const SizedBox(height: AppSpacing.s40),
              Container(width: 52, height: 52,
                decoration: BoxDecoration(color: c.surface, borderRadius: BorderRadius.circular(AppRadii.md),
                  border: Border.all(color: c.border)),
                child: Icon(Icons.email_outlined, color: c.text, size: 24)),
              const SizedBox(height: AppSpacing.s24),
              Text('Verify your\nemail.', style: TextStyle(fontFamily: 'Inter', fontSize: 34,
                fontWeight: FontWeight.w700, letterSpacing: -1.2, height: 1.1, color: c.text)),
              const SizedBox(height: AppSpacing.s10),
              Text.rich(TextSpan(
                style: TextStyle(fontFamily: 'Inter', fontSize: 14, color: c.textMuted, height: 1.5),
                children: [
                  const TextSpan(text: 'We sent a 6-digit code to '),
                  TextSpan(text: widget.email, style: TextStyle(color: c.text, fontWeight: FontWeight.w500)),
                  const TextSpan(text: '.'),
                ])),
              const SizedBox(height: AppSpacing.s40),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: List.generate(6, (i) => _OtpBox(
                  controller: _ctrls[i], focusNode: _foci[i], colors: c,
                  onInput: (v) {
                    if (v.isNotEmpty && i < 5) _foci[i + 1].requestFocus();
                    if (v.isNotEmpty && i == 5) _verify();
                  },
                  onBackspace: () { if (i > 0) { _ctrls[i].clear(); _foci[i - 1].requestFocus(); } },
                )),
              ),
              AuthErrorBanner(error: _error, colors: c),
              const SizedBox(height: AppSpacing.s32),
              AuthPrimaryButton(label: 'Verify & Sign In', isLoading: _isSubmitting, onPressed: _verify, colors: c),
              const SizedBox(height: AppSpacing.s24),
              Center(child: _resendCooldown
                ? Text('Resend code in ${_resendSeconds}s',
                    style: TextStyle(fontFamily: 'Inter', fontSize: 14, color: c.textMuted))
                : GestureDetector(
                    onTap: _resend,
                    child: Text.rich(TextSpan(
                      style: TextStyle(fontFamily: 'Inter', fontSize: 14, color: c.textMuted),
                      children: [
                        const TextSpan(text: "Didn't receive it? "),
                        TextSpan(text: 'Resend code', style: TextStyle(
                          color: c.text, fontWeight: FontWeight.w600,
                          decoration: TextDecoration.underline, decorationColor: c.text)),
                      ])))),
              const SizedBox(height: AppSpacing.s32),
            ]),
          ),
        ),
      ),
    );
  }
}

class _OtpBox extends StatefulWidget {
  final TextEditingController controller;
  final FocusNode focusNode;
  final AppColors colors;
  final ValueChanged<String> onInput;
  final VoidCallback onBackspace;
  const _OtpBox({required this.controller, required this.focusNode, required this.colors,
    required this.onInput, required this.onBackspace});
  @override
  State<_OtpBox> createState() => _OtpBoxState();
}

class _OtpBoxState extends State<_OtpBox> {
  bool _focused = false;

  @override
  void initState() {
    super.initState();
    widget.focusNode.addListener(() { if (mounted) setState(() => _focused = widget.focusNode.hasFocus); });
  }

  @override
  Widget build(BuildContext context) {
    final c = widget.colors;
    final has = widget.controller.text.isNotEmpty;
    return AnimatedContainer(duration: AppMotion.fast, width: 44, height: 56,
      decoration: BoxDecoration(
        color: has ? c.primary.withValues(alpha: 0.05) : c.surface,
        borderRadius: BorderRadius.circular(AppRadii.md),
        border: Border.all(
          color: _focused ? c.primary : has ? c.primary.withValues(alpha: 0.4) : c.border,
          width: _focused ? 1.5 : 1.0)),
      child: KeyboardListener(
        focusNode: FocusNode(),
        onKeyEvent: (e) {
          if (e is KeyDownEvent && e.logicalKey == LogicalKeyboardKey.backspace
              && widget.controller.text.isEmpty) {
            widget.onBackspace();
          }
        },
        child: TextField(
          controller: widget.controller,
          focusNode: widget.focusNode,
          maxLength: 1,
          keyboardType: TextInputType.number,
          textAlign: TextAlign.center,
          inputFormatters: [FilteringTextInputFormatter.digitsOnly],
          style: TextStyle(fontFamily: 'Inter', fontSize: 22, fontWeight: FontWeight.w700, color: c.text),
          decoration: const InputDecoration(counterText: '', border: InputBorder.none, contentPadding: EdgeInsets.zero),
          onChanged: (val) {
            if (val.length > 1) widget.controller.text = val[0];
            widget.onInput(widget.controller.text);
          })));
  }
}
