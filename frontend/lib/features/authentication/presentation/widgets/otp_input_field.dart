import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:frontend/shared/constants/app_spacing.dart';

/// A row of 6 individual digit boxes for OTP entry.
///
/// Each box auto-advances focus to the next on input and
/// auto-retreats on backspace. The [onCompleted] callback fires
/// with the full 6-digit string when all boxes are filled.
class OtpInputField extends StatefulWidget {
  const OtpInputField({
    super.key,
    required this.onCompleted,
    this.onChanged,
    this.enabled = true,
  });

  final ValueChanged<String> onCompleted;
  final ValueChanged<String>? onChanged;
  final bool enabled;

  @override
  State<OtpInputField> createState() => _OtpInputFieldState();
}

class _OtpInputFieldState extends State<OtpInputField> {
  static const int _length = 6;

  late final List<TextEditingController> _controllers;
  late final List<FocusNode> _focusNodes;

  @override
  void initState() {
    super.initState();
    _controllers = List.generate(_length, (_) => TextEditingController());
    _focusNodes = List.generate(_length, (_) => FocusNode());
  }

  @override
  void dispose() {
    for (final c in _controllers) {
      c.dispose();
    }
    for (final f in _focusNodes) {
      f.dispose();
    }
    super.dispose();
  }

  String get _currentOtp =>
      _controllers.map((c) => c.text).join();

  void _onChanged(int index, String value) {
    if (value.length > 1) {
      // Handle paste: distribute across boxes
      final digits = value.replaceAll(RegExp(r'\D'), '');
      for (int i = 0; i < _length && i < digits.length; i++) {
        _controllers[i].text = digits[i];
      }
      final nextEmpty = _controllers.indexWhere((c) => c.text.isEmpty);
      final focus = nextEmpty == -1 ? _length - 1 : nextEmpty;
      _focusNodes[focus].requestFocus();
    } else if (value.isNotEmpty) {
      // Single character typed — move to next
      if (index < _length - 1) {
        _focusNodes[index + 1].requestFocus();
      } else {
        _focusNodes[index].unfocus();
      }
    }
    widget.onChanged?.call(_currentOtp);
    if (_currentOtp.length == _length) {
      widget.onCompleted(_currentOtp);
    }
    setState(() {});
  }

  void _onKeyEvent(int index, KeyEvent event) {
    if (event is KeyDownEvent &&
        event.logicalKey == LogicalKeyboardKey.backspace &&
        _controllers[index].text.isEmpty &&
        index > 0) {
      _focusNodes[index - 1].requestFocus();
      _controllers[index - 1].clear();
      widget.onChanged?.call(_currentOtp);
      setState(() {});
    }
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: List.generate(
        _length,
        (i) => _OtpBox(
          controller: _controllers[i],
          focusNode: _focusNodes[i],
          enabled: widget.enabled,
          isFilled: _controllers[i].text.isNotEmpty,
          onChanged: (v) => _onChanged(i, v),
          onKeyEvent: (e) => _onKeyEvent(i, e),
        ),
      ),
    );
  }
}

// ── Single OTP box ─────────────────────────────────────────────────────────

class _OtpBox extends StatelessWidget {
  const _OtpBox({
    required this.controller,
    required this.focusNode,
    required this.onChanged,
    required this.onKeyEvent,
    required this.isFilled,
    required this.enabled,
  });

  final TextEditingController controller;
  final FocusNode focusNode;
  final ValueChanged<String> onChanged;
  final ValueChanged<KeyEvent> onKeyEvent;
  final bool isFilled;
  final bool enabled;

  @override
  Widget build(BuildContext context) {
    final isFocused = focusNode.hasFocus;
    
    // Use white/translucent colors to ensure visibility on the auth gradient background.
    final textColor = Colors.white;
    final borderColor = Colors.white.withValues(alpha: 0.3);
    final focusBorderColor = Colors.white;
    final filledBorderColor = Colors.white.withValues(alpha: 0.7);
    final fillColor = Colors.white.withValues(alpha: 0.1);
    final filledBgColor = Colors.white.withValues(alpha: 0.15);

    return KeyboardListener(
      focusNode: FocusNode(),
      onKeyEvent: onKeyEvent,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        width: 44,
        height: 54,
        decoration: BoxDecoration(
          color: isFilled ? filledBgColor : fillColor,
          borderRadius: BorderRadius.circular(AppRadius.md),
          border: Border.all(
            color: isFilled
                ? filledBorderColor
                : isFocused
                    ? focusBorderColor
                    : borderColor,
            width: isFilled || isFocused ? 2 : 1,
          ),
        ),
        child: Center(
          child: TextField(
            controller: controller,
            focusNode: focusNode,
            enabled: enabled,
            textAlign: TextAlign.center,
            keyboardType: TextInputType.number,
            maxLength: 1,
            inputFormatters: [FilteringTextInputFormatter.digitsOnly],
            onChanged: onChanged,
            cursorColor: focusBorderColor,
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  color: textColor,
                  fontWeight: FontWeight.w700,
                ),
            decoration: const InputDecoration(
              counterText: '',
              border: InputBorder.none,
              enabledBorder: InputBorder.none,
              focusedBorder: InputBorder.none,
              filled: false,
              contentPadding: EdgeInsets.zero,
            ),
          ),
        ),
      ),
    );
  }
}
