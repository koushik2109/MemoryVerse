import 'dart:math';
import 'dart:ui';

import 'package:flutter/material.dart';

class PixelatedMeshBackground extends StatefulWidget {
  const PixelatedMeshBackground({super.key});

  @override
  State<PixelatedMeshBackground> createState() =>
      _PixelatedMeshBackgroundState();
}

class _PixelatedMeshBackgroundState extends State<PixelatedMeshBackground>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 20),
    )..repeat();
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
        final t = _controller.value * 2 * pi;

        // Animate positions in a circular/lissajous pattern
        final pos1 = Alignment(0.8 * cos(t), 0.8 * sin(t));
        final pos2 = Alignment(-0.6 * cos(t * 1.5), 0.7 * sin(t * 1.2));
        final pos3 = Alignment(0.7 * sin(t * 0.8), -0.6 * cos(t * 1.1));
        final pos4 = Alignment(-0.8 * sin(t * 1.3), -0.8 * cos(t * 0.9));

        return Stack(
          children: [
            // Base dark background
            Container(color: const Color(0xFF120D31)),

            // Gradient Orbs
            _GradientOrb(
              alignment: pos1,
              color: const Color(0xFF6C4CE1),
              radius: 1.5,
            ),
            _GradientOrb(
              alignment: pos2,
              color: const Color(0xFFEF5D89),
              radius: 1.3,
            ),
            _GradientOrb(
              alignment: pos3,
              color: const Color(0xFFFFD166),
              radius: 1.4,
            ),
            _GradientOrb(
              alignment: pos4,
              color: const Color(0xFF6C4CE1),
              radius: 1.6,
            ),

            // Heavy blur filter to blend them and create the soft "abstract" mesh feel
            Positioned.fill(
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 80.0, sigmaY: 80.0),
                child: Container(color: Colors.black.withValues(alpha: 0.2)),
              ),
            ),
          ],
        );
      },
    );
  }
}

class _GradientOrb extends StatelessWidget {
  final Alignment alignment;
  final Color color;
  final double radius;

  const _GradientOrb({
    required this.alignment,
    required this.color,
    required this.radius,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned.fill(
      child: Container(
        decoration: BoxDecoration(
          gradient: RadialGradient(
            center: alignment,
            radius: radius,
            colors: [
              color.withValues(alpha: 0.8),
              color.withValues(alpha: 0.0),
            ],
            stops: const [0.0, 1.0],
          ),
        ),
      ),
    );
  }
}
