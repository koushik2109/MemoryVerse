import 'package:flutter/material.dart';
import 'package:frontend/shared/constants/app_spacing.dart';

/// The supported third-party authentication providers.
enum SocialAuthProvider { google, apple }

/// An outlined social-login button (Google / Apple).
///
/// The Google icon uses [Icons.g_mobiledata] as a Material-native placeholder.
/// Swap with a branded icon package if required by your brand guidelines.
class SocialLoginButton extends StatelessWidget {
  const SocialLoginButton({
    super.key,
    required this.provider,
    required this.onTap,
    this.isLoading = false,
  });

  final SocialAuthProvider provider;
  final VoidCallback? onTap;
  final bool isLoading;

  String get _label => switch (provider) {
        SocialAuthProvider.google => 'Continue with Google',
        SocialAuthProvider.apple => 'Continue with Apple',
      };

  IconData get _icon => switch (provider) {
        SocialAuthProvider.google => Icons.g_mobiledata,
        SocialAuthProvider.apple => Icons.apple,
      };

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;

    return OutlinedButton(
      onPressed: isLoading ? null : onTap,
      style: OutlinedButton.styleFrom(
        side: BorderSide(color: colorScheme.outlineVariant),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(AppRadius.md),
        ),
        minimumSize: const Size(double.infinity, 52),
        padding: const EdgeInsets.symmetric(horizontal: AppSpacing.md),
      ),
      child: AnimatedSwitcher(
        duration: const Duration(milliseconds: 200),
        child: isLoading
            ? SizedBox(
                key: const ValueKey('social-loader'),
                width: AppIconSize.md,
                height: AppIconSize.md,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  valueColor: AlwaysStoppedAnimation<Color>(
                    colorScheme.primary,
                  ),
                ),
              )
            : Row(
                key: const ValueKey('social-label'),
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    _icon,
                    size: AppIconSize.lg,
                    color: colorScheme.onSurface,
                  ),
                  const SizedBox(width: AppSpacing.sm),
                  Text(
                    _label,
                    style: textTheme.labelLarge?.copyWith(
                      color: colorScheme.onSurface,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}
