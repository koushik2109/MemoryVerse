import 'package:flutter/material.dart';

/// Animated cosmic nebula gradient background used on Login, Signup and Auth pages.
class AuthGradientBackground extends StatefulWidget {
  const AuthGradientBackground({super.key, required this.child});

  final Widget child;

  @override
  State<AuthGradientBackground> createState() =>
      _AuthGradientBackgroundState();
}

class _AuthGradientBackgroundState extends State<AuthGradientBackground>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<Alignment> _beginAlign;
  late final Animation<Alignment> _endAlign;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 12),
    )..repeat(reverse: true);

    _beginAlign = TweenSequence<Alignment>([
      TweenSequenceItem(
        tween: AlignmentTween(
          begin: Alignment.topLeft,
          end: Alignment.topRight,
        ),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(
          begin: Alignment.topRight,
          end: Alignment.bottomLeft,
        ),
        weight: 1,
      ),
    ]).animate(_controller);

    _endAlign = TweenSequence<Alignment>([
      TweenSequenceItem(
        tween: AlignmentTween(
          begin: Alignment.bottomRight,
          end: Alignment.bottomLeft,
        ),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(
          begin: Alignment.bottomLeft,
          end: Alignment.topRight,
        ),
        weight: 1,
      ),
    ]).animate(_controller);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: _beginAlign.value,
              end: _endAlign.value,
              colors: const [
                Color(0xFF030712), // Deep Cosmic Void
                Color(0xFF1E1035), // Nebula Violet
                Color(0xFF0F172A), // Deep Space Blue
                Color(0xFF2E1065), // Cosmic Indigo
              ],
              stops: [0.0, 0.4, 0.7, 1.0],
            ),
          ),
          child: child,
        );
      },
      child: widget.child,
    );
  }
}
