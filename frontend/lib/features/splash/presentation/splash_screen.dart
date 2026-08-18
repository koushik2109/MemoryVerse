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
  late Animation<double> _wordmarkOpacity;
  late Animation<Offset> _wordmarkSlide;
  late Animation<double> _taglineOpacity;

  @override
  void initState() {
    super.initState();

    // Background fade: 0→1 over 400ms
    _bgCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 400));
    _bgOpacity = Tween<double>(begin: 0, end: 1)
        .animate(CurvedAnimation(parent: _bgCtrl, curve: Curves.easeOut));

    // Logo: scale 0.6→1.0 + fade, over 550ms with spring feel
    _logoCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 550));
    _logoScale = Tween<double>(begin: 0.60, end: 1.0)
        .animate(CurvedAnimation(parent: _logoCtrl, curve: Curves.easeOutBack));
    _logoOpacity = Tween<double>(begin: 0, end: 1)
        .animate(CurvedAnimation(parent: _logoCtrl, curve: const Interval(0.0, 0.6, curve: Curves.easeOut)));

    // Wordmark: subtle upward slide + fade over 450ms
    _wordmarkCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 450));
    _wordmarkOpacity = Tween<double>(begin: 0, end: 1)
        .animate(CurvedAnimation(parent: _wordmarkCtrl, curve: Curves.easeOut));
    _wordmarkSlide = Tween<Offset>(begin: const Offset(0, 0.2), end: Offset.zero)
        .animate(CurvedAnimation(parent: _wordmarkCtrl, curve: Curves.easeOutCubic));

    // Tagline: fade in after wordmark
    _taglineCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 400));
    _taglineOpacity = Tween<double>(begin: 0, end: 1)
        .animate(CurvedAnimation(parent: _taglineCtrl, curve: Curves.easeOut));

    _run();
  }

  Future<void> _run() async {
    // Start background immediately
    _bgCtrl.forward();

    await Future<void>.delayed(const Duration(milliseconds: 180));
    if (!mounted) return;

    // Logo reveal
    await _logoCtrl.forward();
    if (!mounted) return;

    // Wordmark follows
    await Future<void>.delayed(const Duration(milliseconds: 80));
    if (!mounted) return;
    _wordmarkCtrl.forward();

    // Tagline follows wordmark
    await Future<void>.delayed(const Duration(milliseconds: 200));
    if (!mounted) return;
    _taglineCtrl.forward();

    // Hold for legibility
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
    final isDark = context.isDark;

    return Scaffold(
      backgroundColor: c.bg,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // ── Atmospheric background motif ──────────────────
          FadeTransition(
            opacity: _bgOpacity,
            child: _AtmosphericBackground(isDark: isDark),
          ),

          // ── Center content ────────────────────────────────
          Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Logo mark
                ScaleTransition(
                  scale: _logoScale,
                  child: FadeTransition(
                    opacity: _logoOpacity,
                    child: _LogoMark(size: 76, colors: c),
                  ),
                ),

                const SizedBox(height: AppSpacing.s28),

                // Wordmark
                SlideTransition(
                  position: _wordmarkSlide,
                  child: FadeTransition(
                    opacity: _wordmarkOpacity,
                    child: Text(
                      'MemoryVerse',
                      style: GoogleFonts.poppins(
                        fontSize: 30,
                        fontWeight: FontWeight.w700,
                        letterSpacing: -1.0,
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
                      fontSize: 13,
                      fontWeight: FontWeight.w400,
                      letterSpacing: 0.2,
                      color: c.textMuted,
                    ),
                  ),
                ),
              ],
            ),
          ),

          // ── Bottom loading dots ───────────────────────────
          Positioned(
            bottom: 60,
            left: 0,
            right: 0,
            child: FadeTransition(
              opacity: _taglineOpacity,
              child: _PulsingDots(color: c.textMuted),
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Logo Mark
// ─────────────────────────────────────────────────────────────────

class _LogoMark extends StatelessWidget {
  final double size;
  final AppColors colors;
  const _LogoMark({required this.size, required this.colors});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: colors.primary,
        borderRadius: BorderRadius.circular(size * 0.28),
        boxShadow: [
          BoxShadow(
            color: colors.primary.withValues(alpha: 0.2),
            blurRadius: 24,
            spreadRadius: 0,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Stack(
        alignment: Alignment.center,
        children: [
          Positioned(
            right: size * 0.1,
            top: size * 0.1,
            child: Container(
              width: size * 0.28,
              height: size * 0.28,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: colors.primaryInverse.withValues(alpha: 0.1),
              ),
            ),
          ),
          Icon(
            Icons.auto_awesome_rounded,
            color: colors.primaryInverse,
            size: size * 0.44,
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// Atmospheric background — lightweight, no shaders
// ─────────────────────────────────────────────────────────────────

class _AtmosphericBackground extends StatelessWidget {
  final bool isDark;
  const _AtmosphericBackground({required this.isDark});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _BgPainter(isDark: isDark),
    );
  }
}

class _BgPainter extends CustomPainter {
  final bool isDark;
  const _BgPainter({required this.isDark});

  @override
  void paint(Canvas canvas, Size size) {
    final baseAlpha = isDark ? 0.06 : 0.04;

    // Subtle memory frame motifs — top-right corner
    final paint = Paint()
      ..color = (isDark ? Colors.white : Colors.black).withValues(alpha: baseAlpha)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;

    // Three concentric arcs — top right
    for (var i = 1; i <= 3; i++) {
      final r = 80.0 + (i * 55);
      canvas.drawArc(
        Rect.fromCenter(
          center: Offset(size.width, 0),
          width: r * 2,
          height: r * 2,
        ),
        0.5,
        1.0,
        false,
        paint,
      );
    }

    // Diagonal film strip dots — bottom-left
    final dotPaint = Paint()
      ..color = (isDark ? Colors.white : Colors.black).withValues(alpha: baseAlpha * 0.8)
      ..style = PaintingStyle.fill;

    for (var row = 0; row < 4; row++) {
      for (var col = 0; col < 4; col++) {
        canvas.drawCircle(
          Offset(30.0 + (col * 20), size.height - 80 + (row * 20)),
          1.5,
          dotPaint,
        );
      }
    }
  }

  @override
  bool shouldRepaint(_BgPainter old) => old.isDark != isDark;
}

// ─────────────────────────────────────────────────────────────────
// Pulsing loading indicator
// ─────────────────────────────────────────────────────────────────

class _PulsingDots extends StatefulWidget {
  final Color color;
  const _PulsingDots({required this.color});

  @override
  State<_PulsingDots> createState() => _PulsingDotsState();
}

class _PulsingDotsState extends State<_PulsingDots>
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
    _anim = Tween<double>(begin: 0.3, end: 1.0).animate(
      CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut),
    );
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
          // Stagger each dot
          final phase = (i / 3.0);
          final opacity = ((_anim.value + phase) % 1.0).clamp(0.25, 1.0);
          return Container(
            margin: const EdgeInsets.symmetric(horizontal: 3),
            width: 4,
            height: 4,
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
