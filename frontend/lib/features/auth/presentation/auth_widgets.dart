// Shared auth UI primitives — imported by Sign Up, Forgot Password, OTP screens.
// All classes here are public so they can be shared across auth screens.

import 'package:flutter/material.dart';
import 'package:memory_verse/core/design/tokens.dart';

// ─────────────────────────────────────────────────────────────────
// Small MemoryVerse logo mark for auth screens
// ─────────────────────────────────────────────────────────────────

class AuthLogo extends StatelessWidget {
  final AppColors colors;
  const AuthLogo({super.key, required this.colors});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 44,
      height: 44,
      decoration: BoxDecoration(
        color: colors.primary,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: colors.primary.withValues(alpha: 0.15),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Icon(
        Icons.auto_awesome_rounded,
        color: colors.primaryInverse,
        size: 22,
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Premium auth text field
// ─────────────────────────────────────────────────────────────────

class AuthField extends StatelessWidget {
  final String label;
  final String hint;
  final TextEditingController controller;
  final FocusNode? focusNode;
  final bool obscureText;
  final TextInputType keyboardType;
  final TextInputAction textInputAction;
  final ValueChanged<String>? onSubmitted;
  final AppColors colors;
  final IconData? prefixIcon;
  final Widget? suffixIcon;

  const AuthField({
    super.key,
    required this.label,
    required this.hint,
    required this.controller,
    this.focusNode,
    this.obscureText = false,
    this.keyboardType = TextInputType.text,
    this.textInputAction = TextInputAction.next,
    this.onSubmitted,
    required this.colors,
    this.prefixIcon,
    this.suffixIcon,
  });

  @override
  Widget build(BuildContext context) {
    final c = colors;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontFamily: 'Inter',
            fontSize: 12,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.4,
            color: Color(0xFF737373),
          ),
        ),
        const SizedBox(height: AppSpacing.s8),
        TextField(
          controller: controller,
          focusNode: focusNode,
          obscureText: obscureText,
          keyboardType: keyboardType,
          textInputAction: textInputAction,
          onSubmitted: onSubmitted,
          style: TextStyle(
            fontFamily: 'Inter',
            fontSize: 15,
            fontWeight: FontWeight.w400,
            color: c.text,
          ),
          decoration: InputDecoration(
            hintText: hint,
            hintStyle: TextStyle(
              fontFamily: 'Inter',
              fontSize: 15,
              color: c.textMuted.withValues(alpha: 0.6),
            ),
            filled: true,
            fillColor: c.surface,
            prefixIcon: prefixIcon != null
                ? Icon(prefixIcon, size: 18, color: c.textMuted)
                : null,
            suffixIcon: suffixIcon,
            contentPadding: const EdgeInsets.symmetric(
              horizontal: AppSpacing.s16,
              vertical: AppSpacing.s16,
            ),
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(AppRadii.md),
              borderSide: BorderSide(color: c.border),
            ),
            enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(AppRadii.md),
              borderSide: BorderSide(color: c.border),
            ),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(AppRadii.md),
              borderSide: BorderSide(color: c.primary, width: 1.5),
            ),
          ),
        ),
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Animated error banner
// ─────────────────────────────────────────────────────────────────

class AuthErrorBanner extends StatelessWidget {
  final String? error;
  final AppColors colors;
  const AuthErrorBanner({super.key, required this.error, required this.colors});

  @override
  Widget build(BuildContext context) {
    return AnimatedSize(
      duration: AppMotion.normal,
      curve: Curves.easeOutCubic,
      child: error != null
          ? Padding(
              padding: const EdgeInsets.only(top: AppSpacing.s16),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(AppSpacing.s14),
                decoration: BoxDecoration(
                  color: colors.error.withValues(alpha: 0.07),
                  borderRadius: BorderRadius.circular(AppRadii.md),
                  border: Border.all(color: colors.error.withValues(alpha: 0.2)),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.error_outline_rounded, size: 16, color: colors.error),
                    const SizedBox(width: AppSpacing.s10),
                    Expanded(
                      child: Text(
                        error!,
                        style: TextStyle(
                          fontFamily: 'Inter',
                          fontSize: 13,
                          color: colors.error,
                          height: 1.4,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            )
          : const SizedBox.shrink(),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Full-width primary auth button
// ─────────────────────────────────────────────────────────────────

class AuthPrimaryButton extends StatelessWidget {
  final String label;
  final bool isLoading;
  final VoidCallback onPressed;
  final AppColors colors;

  const AuthPrimaryButton({
    super.key,
    required this.label,
    required this.isLoading,
    required this.onPressed,
    required this.colors,
  });

  @override
  Widget build(BuildContext context) {
    final c = colors;
    return SizedBox(
      width: double.infinity,
      height: 54,
      child: AnimatedOpacity(
        opacity: isLoading ? 0.7 : 1.0,
        duration: AppMotion.fast,
        child: ElevatedButton(
          onPressed: isLoading ? null : onPressed,
          style: ElevatedButton.styleFrom(
            backgroundColor: c.primary,
            foregroundColor: c.primaryInverse,
            elevation: 0,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(AppRadii.md),
            ),
            disabledBackgroundColor: c.primary.withValues(alpha: 0.7),
          ),
          child: isLoading
              ? SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(
                    color: c.primaryInverse,
                    strokeWidth: 2.0,
                  ),
                )
              : Text(
                  label,
                  style: const TextStyle(
                    fontFamily: 'Inter',
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.1,
                  ),
                ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// "or continue with" divider
// ─────────────────────────────────────────────────────────────────

class AuthOrDivider extends StatelessWidget {
  final AppColors colors;
  const AuthOrDivider({super.key, required this.colors});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(child: Divider(color: colors.border, height: 1)),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s16),
          child: Text(
            'or continue with',
            style: TextStyle(fontFamily: 'Inter', fontSize: 12, color: colors.textMuted),
          ),
        ),
        Expanded(child: Divider(color: colors.border, height: 1)),
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Social auth button (Google / Apple)
// ─────────────────────────────────────────────────────────────────

class AuthSocialButton extends StatelessWidget {
  final String label;
  final Widget icon;
  final VoidCallback onPressed;
  final AppColors colors;

  const AuthSocialButton({
    super.key,
    required this.label,
    required this.icon,
    required this.onPressed,
    required this.colors,
  });

  @override
  Widget build(BuildContext context) {
    final c = colors;
    return SizedBox(
      height: 50,
      child: OutlinedButton(
        onPressed: onPressed,
        style: OutlinedButton.styleFrom(
          foregroundColor: c.text,
          side: BorderSide(color: c.border),
          backgroundColor: c.surface,
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(AppRadii.md)),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            icon,
            const SizedBox(width: AppSpacing.s8),
            Text(
              label,
              style: TextStyle(
                fontFamily: 'Inter',
                fontSize: 14,
                fontWeight: FontWeight.w500,
                color: c.text,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Navigation link (e.g. "Don't have an account? Create one")
// ─────────────────────────────────────────────────────────────────

class AuthNavLink extends StatelessWidget {
  final String question;
  final String action;
  final VoidCallback onTap;
  final AppColors colors;

  const AuthNavLink({
    super.key,
    required this.question,
    required this.action,
    required this.onTap,
    required this.colors,
  });

  @override
  Widget build(BuildContext context) {
    return Text.rich(
      TextSpan(
        style: TextStyle(fontFamily: 'Inter', fontSize: 14, color: colors.textMuted),
        children: [
          TextSpan(text: '$question '),
          WidgetSpan(
            alignment: PlaceholderAlignment.baseline,
            baseline: TextBaseline.alphabetic,
            child: GestureDetector(
              onTap: onTap,
              child: Text(
                action,
                style: TextStyle(
                  fontFamily: 'Inter',
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: colors.text,
                  decoration: TextDecoration.underline,
                  decorationColor: colors.text,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Stylized Google "G" icon
// ─────────────────────────────────────────────────────────────────

class GoogleIcon extends StatelessWidget {
  const GoogleIcon({super.key});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 20,
      height: 20,
      child: Center(
        child: Image.asset(
          'assets/images/google_logo.png',
          width: 18,
          height: 18,
          fit: BoxFit.contain,
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Stylized back button for auth nav rows
// ─────────────────────────────────────────────────────────────────

class AuthBackButton extends StatelessWidget {
  final VoidCallback onTap;
  final AppColors colors;

  const AuthBackButton({super.key, required this.onTap, required this.colors});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(AppSpacing.s10),
        decoration: BoxDecoration(
          color: colors.surface,
          borderRadius: BorderRadius.circular(AppRadii.md),
          border: Border.all(color: colors.border),
        ),
        child: Icon(Icons.arrow_back_ios_new_rounded, size: 16, color: colors.text),
      ),
    );
  }
}
