import 'package:flutter/material.dart';
import 'package:frontend/shared/constants/app_spacing.dart';

/// A reusable, animated text field for auth forms.
///
/// Handles password visibility toggling and exposes standard
/// [TextFormField] configuration through its constructor.
class AuthTextField extends StatefulWidget {
  const AuthTextField({
    super.key,
    required this.controller,
    required this.label,
    required this.prefixIcon,
    this.isPassword = false,
    this.keyboardType = TextInputType.text,
    this.textInputAction = TextInputAction.next,
    this.validator,
    this.onFieldSubmitted,
    this.autofillHints,
    this.enabled = true,
  });

  final TextEditingController controller;
  final String label;
  final IconData prefixIcon;
  final bool isPassword;
  final TextInputType keyboardType;
  final TextInputAction textInputAction;
  final String? Function(String?)? validator;
  final VoidCallback? onFieldSubmitted;
  final Iterable<String>? autofillHints;
  final bool enabled;

  @override
  State<AuthTextField> createState() => _AuthTextFieldState();
}

class _AuthTextFieldState extends State<AuthTextField> {
  bool _obscured = true;

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    
    // On the auth pages, the background is a rich primary gradient.
    // We need to force light-colored text and borders so they are visible.
    final textColor = Colors.white;
    final hintColor = Colors.white.withValues(alpha: 0.7);
    final borderColor = Colors.white.withValues(alpha: 0.3);
    final focusBorderColor = Colors.white;
    final fillColor = Colors.white.withValues(alpha: 0.1);

    return TextFormField(
      controller: widget.controller,
      obscureText: widget.isPassword && _obscured,
      keyboardType: widget.keyboardType,
      textInputAction: widget.textInputAction,
      validator: widget.validator,
      enabled: widget.enabled,
      autofillHints: widget.autofillHints,
      style: TextStyle(color: textColor, fontWeight: FontWeight.w500),
      cursorColor: focusBorderColor,
      onFieldSubmitted:
          widget.onFieldSubmitted != null ? (_) => widget.onFieldSubmitted!() : null,
      decoration: InputDecoration(
        labelText: widget.label,
        labelStyle: TextStyle(color: hintColor),
        floatingLabelStyle: TextStyle(color: focusBorderColor, fontWeight: FontWeight.w600),
        filled: true,
        fillColor: fillColor,
        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(AppRadius.lg),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(AppRadius.lg),
          borderSide: BorderSide(color: borderColor),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(AppRadius.lg),
          borderSide: BorderSide(color: focusBorderColor, width: 2),
        ),
        errorStyle: TextStyle(
          color: colorScheme.errorContainer, // Light error color for dark background
          fontWeight: FontWeight.w600,
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(AppRadius.lg),
          borderSide: BorderSide(color: colorScheme.errorContainer),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(AppRadius.lg),
          borderSide: BorderSide(color: colorScheme.errorContainer, width: 2),
        ),
        prefixIcon: Icon(widget.prefixIcon, size: AppIconSize.md, color: hintColor),
        suffixIcon: widget.isPassword
            ? IconButton(
                icon: Icon(
                  _obscured ? Icons.visibility_off_outlined : Icons.visibility_outlined,
                  color: hintColor,
                  size: AppIconSize.md,
                ),
                onPressed: () => setState(() => _obscured = !_obscured),
                tooltip: _obscured ? 'Show password' : 'Hide password',
              )
            : null,
      ),
    );
  }
}
