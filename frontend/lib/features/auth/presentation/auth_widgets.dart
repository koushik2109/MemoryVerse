// Shared auth UI primitives — imported by Sign Up, Forgot Password, OTP screens.

import 'package:flutter/material.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart';

class AuthLogo extends StatelessWidget {
  const AuthLogo({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 44,
      height: 44,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(AppRadius.md),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.15),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: const Icon(
        Icons.auto_awesome_rounded,
        color: AppColors.plum800,
        size: 22,
      ),
    );
  }
}

class AuthField extends StatelessWidget {
  final String label;
  final String hint;
  final TextEditingController controller;
  final FocusNode? focusNode;
  final bool obscureText;
  final TextInputType keyboardType;
  final TextInputAction textInputAction;
  final ValueChanged<String>? onSubmitted;
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
    this.prefixIcon,
    this.suffixIcon,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: AppTextStyles.caption.copyWith(
            fontWeight: FontWeight.w600,
            color: AppColors.onDarkPrimary,
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
          style: AppTextStyles.body.copyWith(
            color: AppColors.onDarkPrimary,
          ),
          decoration: InputDecoration(
            hintText: hint,
            hintStyle: AppTextStyles.body.copyWith(
              color: AppColors.onDarkMuted,
            ),
            filled: true,
            fillColor: Colors.white.withOpacity(0.1),
            prefixIcon: prefixIcon != null
                ? Icon(prefixIcon, size: 18, color: AppColors.onDarkMuted)
                : null,
            suffixIcon: suffixIcon,
            contentPadding: const EdgeInsets.symmetric(
              horizontal: AppSpacing.s16,
              vertical: AppSpacing.s16,
            ),
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(AppRadius.md),
              borderSide: BorderSide(color: Colors.white.withOpacity(0.2)),
            ),
            enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(AppRadius.md),
              borderSide: BorderSide(color: Colors.white.withOpacity(0.2)),
            ),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(AppRadius.md),
              borderSide: const BorderSide(color: Colors.white, width: 1.5),
            ),
          ),
        ),
      ],
    );
  }
}

class AuthErrorBanner extends StatelessWidget {
  final String? error;
  const AuthErrorBanner({super.key, required this.error});

  @override
  Widget build(BuildContext context) {
    return AnimatedSize(
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeOutCubic,
      child: error != null
          ? Padding(
              padding: const EdgeInsets.only(top: AppSpacing.s16),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(AppSpacing.s12),
                decoration: BoxDecoration(
                  color: const Color(0xFFE5484D).withOpacity(0.1),
                  borderRadius: BorderRadius.circular(AppRadius.md),
                  border: Border.all(color: const Color(0xFFE5484D).withOpacity(0.3)),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(Icons.error_outline_rounded, size: 16, color: Color(0xFFE5484D)),
                    const SizedBox(width: AppSpacing.s8),
                    Expanded(
                      child: Text(
                        error!,
                        style: AppTextStyles.caption.copyWith(
                          color: const Color(0xFFE5484D),
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

class AuthPrimaryButton extends StatelessWidget {
  final String label;
  final bool isLoading;
  final VoidCallback onPressed;

  const AuthPrimaryButton({
    super.key,
    required this.label,
    required this.isLoading,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: AnimatedOpacity(
        opacity: isLoading ? 0.7 : 1.0,
        duration: const Duration(milliseconds: 200),
        child: ElevatedButton(
          onPressed: isLoading ? null : onPressed,
          style: AppTheme.primaryButtonOnDark,
          child: isLoading
              ? const SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(
                    color: AppColors.plum800,
                    strokeWidth: 2.0,
                  ),
                )
              : Text(label),
        ),
      ),
    );
  }
}

class AuthOrDivider extends StatelessWidget {
  const AuthOrDivider({super.key});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(child: Divider(color: Colors.white.withOpacity(0.2), height: 1)),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s16),
          child: Text(
            'or continue with',
            style: AppTextStyles.caption.copyWith(color: AppColors.onDarkMuted),
          ),
        ),
        Expanded(child: Divider(color: Colors.white.withOpacity(0.2), height: 1)),
      ],
    );
  }
}

class AuthSocialButton extends StatelessWidget {
  final String label;
  final Widget icon;
  final VoidCallback onPressed;

  const AuthSocialButton({
    super.key,
    required this.label,
    required this.icon,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 44,
      width: double.infinity,
      child: OutlinedButton(
        onPressed: onPressed,
        style: AppTheme.secondaryButtonOnDark,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            icon,
            const SizedBox(width: AppSpacing.s8),
            Text(label),
          ],
        ),
      ),
    );
  }
}

class AuthNavLink extends StatelessWidget {
  final String question;
  final String action;
  final VoidCallback onTap;

  const AuthNavLink({
    super.key,
    required this.question,
    required this.action,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Text.rich(
      TextSpan(
        style: AppTextStyles.body.copyWith(color: AppColors.onDarkMuted),
        children: [
          TextSpan(text: '$question '),
          WidgetSpan(
            alignment: PlaceholderAlignment.baseline,
            baseline: TextBaseline.alphabetic,
            child: GestureDetector(
              onTap: onTap,
              child: Text(
                action,
                style: AppTextStyles.body.copyWith(
                  fontWeight: FontWeight.w600,
                  color: AppColors.onDarkPrimary,
                  decoration: TextDecoration.underline,
                  decorationColor: AppColors.onDarkPrimary,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

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

class AuthBackButton extends StatelessWidget {
  final VoidCallback onTap;

  const AuthBackButton({super.key, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return IconButton(
      onPressed: onTap,
      style: AppTheme.iconButtonStyle(onDark: true),
      icon: const Icon(Icons.arrow_back_ios_new_rounded, size: 16),
    );
  }
}
