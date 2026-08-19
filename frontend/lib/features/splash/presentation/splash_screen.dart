import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/navigation/router.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  // ── Controllers ───────────────────────────────────────────
  late AnimationController _bgCtrl;
  late AnimationController _logoCtrl;
  late AnimationController _wordmarkCtrl;
  late AnimationController _taglineCtrl;

  // ── Animations ────────────────────────────────────────────
  late Animation<double> _bgOpacity;
  late Animation<double> _logoScale;
  late Animation<double> _logoOpacity;
  late Animation<double> _logoRotation;
  late Animation<double> _wordmarkOpacity;
  late Animation<Offset> _wordmarkSlide;
  late Animation<double> _taglineOpacity;

  @override
  void initState() {
    super.initState();

    // Background fade: 0→1 over 400ms
    _bgCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 400),
    );
    _bgOpacity = Tween<double>(
      begin: 0,
      end: 1,
    ).animate(CurvedAnimation(parent: _bgCtrl, curve: Curves.easeOut));

    // Logo: scale 0.6→1.0 + fade, over 550ms with spring feel
    _logoCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 550));
    _logoScale = Tween<double>(begin: 0.60, end: 1.0)
        .animate(CurvedAnimation(parent: _logoCtrl, curve: Curves.easeOutBack));
    _logoOpacity = Tween<double>(begin: 0, end: 1)
        .animate(CurvedAnimation(parent: _logoCtrl, curve: const Interval(0.0, 0.6, curve: Curves.easeOut)));
    _logoRotation = Tween<double>(begin: -0.08, end: 0.0)
        .animate(CurvedAnimation(parent: _logoCtrl, curve: Curves.easeOutBack));

    // Wordmark: subtle upward slide + fade over 450ms
    _wordmarkCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 450),
    );
    _wordmarkOpacity = Tween<double>(
      begin: 0,
      end: 1,
    ).animate(CurvedAnimation(parent: _wordmarkCtrl, curve: Curves.easeOut));
    _wordmarkSlide =
        Tween<Offset>(begin: const Offset(0, 0.2), end: Offset.zero).animate(
          CurvedAnimation(parent: _wordmarkCtrl, curve: Curves.easeOutCubic),
        );

    // Tagline: fade in after wordmark
    _taglineCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 400),
    );
    _taglineOpacity = Tween<double>(
      begin: 0,
      end: 1,
    ).animate(CurvedAnimation(parent: _taglineCtrl, curve: Curves.easeOut));

    _run();
  }

  Future<void> _run() async {
    _bgCtrl.forward();

    await Future<void>.delayed(const Duration(milliseconds: 180));
    if (!mounted) return;

    await _logoCtrl.forward();
    if (!mounted) return;

    await Future<void>.delayed(const Duration(milliseconds: 80));
    if (!mounted) return;
    _wordmarkCtrl.forward();

    await Future<void>.delayed(const Duration(milliseconds: 200));
    if (!mounted) return;
    _taglineCtrl.forward();

    await Future<void>.delayed(const Duration(milliseconds: 900));
    if (!mounted) return;

    _navigate();
  }

  void _navigate() {
    final session = Supabase.instance.client.auth.currentSession;
    if (!mounted) return;
    context.go(session != null ? Routes.home : Routes.landing);
  }

  @override
  void dispose() {
    _bgCtrl.dispose();
    _logoCtrl.dispose();
    _wordmarkCtrl.dispose();
    _taglineCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;

    return Scaffold(
      backgroundColor: c.bg,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // ── Warm atmospheric background ──────────────────
          FadeTransition(
            opacity: _bgOpacity,
            child: _WarmBackground(colors: c),
          ),

          // ── Center content ────────────────────────────────
          Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Logo mark — warm amber with camera
                AnimatedBuilder(
                  animation: _logoCtrl,
                  builder: (_, child) => Transform.rotate(
                    angle: _logoRotation.value,
                    child: Transform.scale(
                      scale: _logoScale.value,
                      child: Opacity(
                        opacity: _logoOpacity.value,
                        child: child,
                      ),
                    ),
                  ),
                  child: _WarmLogoMark(size: 84, colors: c),
                ),

                const SizedBox(height: AppSpacing.s28),

                // Wordmark — serif font
                SlideTransition(
                  position: _wordmarkSlide,
                  child: FadeTransition(
                    opacity: _wordmarkOpacity,
                    child: Text(
                      'MemoryVerse',
                      style: GoogleFonts.playfairDisplay(
                        fontSize: 32,
                        fontWeight: FontWeight.w700,
                        letterSpacing: -0.5,
                        color: c.text,
                        height: 1,
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: AppSpacing.s10),

                // Tagline
                FadeTransition(
                  opacity: _taglineOpacity,
                  child: Text(
                    'Where memories live forever.',
                    style: TextStyle(
                      fontFamily: 'Inter',
                      fontSize: 14,
                      fontWeight: FontWeight.w400,
                      letterSpacing: 0.2,
                      color: c.textMuted,
                    ),
                  ),
                ),
              ],
            ),
          ),

          // ── Bottom loading indicator ───────────────────────
          Positioned(
            bottom: 60,
            left: 0,
            right: 0,
            child: FadeTransition(
              opacity: _taglineOpacity,
              child: _WarmPulsingDots(color: c.primary),
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Warm Logo Mark — amber gradient with camera lens
// ─────────────────────────────────────────────────────────────────

class _WarmLogoMark extends StatelessWidget {
  final double size;
  final AppColors colors;
  const _WarmLogoMark({required this.size, required this.colors});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Color(0xFFE8A838),
            Color(0xFFD97B5C),
          ],
        ),
        borderRadius: BorderRadius.circular(size * 0.28),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFFE8A838).withValues(alpha: 0.3),
            blurRadius: 32,
            spreadRadius: 0,
            offset: const Offset(0, 10),
          ),
          BoxShadow(
            color: const Color(0xFFD97B5C).withValues(alpha: 0.15),
            blurRadius: 16,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Stack(
        alignment: Alignment.center,
        children: [
          // Subtle lens flare ring
          Positioned(
            right: size * 0.08,
            top: size * 0.08,
            child: Container(
              width: size * 0.3,
              height: size * 0.3,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.white.withValues(alpha: 0.12),
              ),
            ),
          ),
          Icon(
            Icons.camera_alt_rounded,
            color: Colors.white,
            size: size * 0.42,
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Warm atmospheric background — golden gradient orbs + film motifs
// ─────────────────────────────────────────────────────────────────

class _WarmBackground extends StatelessWidget {
  final AppColors colors;
  const _WarmBackground({required this.colors});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _WarmMotifPainter(
        colors: colors,
        isDark: context.isDark,
      ),
    );
  }
}

class _WarmMotifPainter extends CustomPainter {
  final AppColors colors;
  final bool isDark;
  const _WarmMotifPainter({required this.colors, required this.isDark});

  @override
  void paint(Canvas canvas, Size size) {
    final baseAlpha = isDark ? 0.06 : 0.04;

    // Subtle memory frame motifs — top-right corner
    final paint = Paint()
      ..color = (isDark ? Colors.white : Colors.black).withValues(alpha: baseAlpha)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;

    // Top-right frames
    _drawFrame(canvas, Offset(size.width - 60, 80), 40, -0.1, paint);
    _drawFrame(canvas, Offset(size.width - 30, 140), 30, 0.08, paint);

    // Bottom-left dots (like film sprocket holes)
    final dotPaint = Paint()
      ..color = (isDark ? Colors.white : Colors.black).withValues(alpha: baseAlpha * 0.8)
      ..style = PaintingStyle.fill;

    for (var row = 0; row < 3; row++) {
      for (var col = 0; col < 3; col++) {
        canvas.drawCircle(
          Offset(28.0 + (col * 16), size.height - 70 + (row * 16)),
          2.0,
          dotPaint,
        );
      }
    }
  }

  void _drawFrame(Canvas canvas, Offset center, double size, double angle, Paint paint) {
    canvas.save();
    canvas.translate(center.dx, center.dy);
    canvas.rotate(angle);
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(center: Offset.zero, width: size, height: size * 1.2),
        const Radius.circular(4),
      ),
      paint,
    );
    canvas.restore();
  }

  @override
  bool shouldRepaint(_WarmMotifPainter old) => old.isDark != isDark;
}

// ─────────────────────────────────────────────────────────────────
// Warm pulsing loading dots — amber colored
// ─────────────────────────────────────────────────────────────────

class _WarmPulsingDots extends StatefulWidget {
  final Color color;
  const _WarmPulsingDots({required this.color});

  @override
  State<_WarmPulsingDots> createState() => _WarmPulsingDotsState();
}

class _WarmPulsingDotsState extends State<_WarmPulsingDots>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double> _anim;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat(reverse: true);
    _anim = Tween<double>(
      begin: 0.3,
      end: 1.0,
    ).animate(CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _anim,
      builder: (_, __) => Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: List.generate(3, (i) {
          final phase = (i / 3.0);
          final opacity = ((_anim.value + phase) % 1.0).clamp(0.25, 1.0);
          return Container(
            margin: const EdgeInsets.symmetric(horizontal: 3),
            width: 5,
            height: 5,
            decoration: BoxDecoration(
              color: widget.color.withValues(alpha: opacity),
              shape: BoxShape.circle,
            ),
          );
        }),
      ),
    );
  }
}
