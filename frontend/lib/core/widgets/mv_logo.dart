import 'package:flutter/material.dart';
import 'package:memory_verse/core/design/tokens.dart';

/// MemoryVerse logo mark — used in splash and branding.
class MvLogo extends StatelessWidget {
  const MvLogo({super.key, this.size = 48});
  final double size;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: c.primary,
        borderRadius: BorderRadius.circular(size * 0.28),
      ),
      child: Icon(
        Icons.auto_awesome_rounded,
        color: c.primaryInverse,
        size: size * 0.48,
      ),
    );
  }
}
