import 'package:flutter/material.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart';

class AuroraBackground extends StatelessWidget {
  final bool isDark;

  const AuroraBackground({super.key, this.isDark = false});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _AuroraPainter(isDark: isDark),
      child: const SizedBox.expand(),
    );
  }
}

class _AuroraPainter extends CustomPainter {
  final bool isDark;

  _AuroraPainter({required this.isDark});

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;

    canvas.drawRect(
      rect,
      Paint()
        ..shader = (isDark ? AppGradients.dark : AppGradients.light)
            .createShader(rect),
    );

    // If it's dark, we might want slightly different blob colors or opacities,
    // but the user requested "aurora theme here too" for the dark screens.
    // The blobs on a dark gradient will look like nebulas.
    // Let's use the same blobs but adjust opacity slightly for dark mode if needed,
    // or just keep them as they are since they are translucent.

    _paintBlob(
      canvas,
      size,
      const Alignment(-0.84, -0.84),
      0.85,
      const Color(0x6BAF6DFF),
    );
    _paintBlob(
      canvas,
      size,
      const Alignment(0.5, -0.3),
      0.75,
      const Color(0x8CFFEBAA),
    );
    _paintBlob(
      canvas,
      size,
      const Alignment(-0.7, 0.6),
      0.7,
      const Color(0x66FF64B4),
    );
    _paintBlob(
      canvas,
      size,
      const Alignment(0.84, 0.84),
      0.7,
      const Color(0x7378BEFF),
    );

    if (isDark) {
      // Apply the scrim on top of the blobs if it's dark mode, to ensure text legibility
      canvas.drawRect(
        rect,
        Paint()..shader = AppGradients.scrim.createShader(rect),
      );
    }
  }

  void _paintBlob(
    Canvas canvas,
    Size size,
    Alignment center,
    double radiusFactor,
    Color color,
  ) {
    final paint = Paint()
      ..shader = RadialGradient(colors: [color, color.withOpacity(0)])
          .createShader(
            Rect.fromCircle(
              center: center.alongSize(size),
              radius: size.shortestSide * radiusFactor,
            ),
          );
    canvas.drawRect(Offset.zero & size, paint);
  }

  @override
  bool shouldRepaint(covariant _AuroraPainter oldDelegate) =>
      oldDelegate.isDark != isDark;
}
