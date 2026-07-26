import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:memory_verse/config/theme/app_theme.dart';
import 'package:memory_verse/app.dart';

// ─────────────────────────────────────────────
//  Splash Screen
// ─────────────────────────────────────────────

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  late AnimationController _logoController;
  late AnimationController _orbitController;
  late AnimationController _pulseController;

  late Animation<double> _logoScale;
  late Animation<double> _logoOpacity;
  late Animation<double> _taglineOpacity;
  late Animation<double> _taglineSlide;
  late Animation<double> _orbit;
  late Animation<double> _pulse;

  @override
  void initState() {
    super.initState();

    // Logo entrance
    _logoController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    );

    // Continuously rotating orbit rings
    _orbitController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 3000),
    )..repeat();

    // Gentle pulse on the glow
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);

    _logoScale = Tween<double>(begin: 0.6, end: 1.0).animate(
      CurvedAnimation(parent: _logoController, curve: Curves.easeOutBack),
    );
    _logoOpacity = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(
        parent: _logoController,
        curve: const Interval(0, 0.6, curve: Curves.easeIn),
      ),
    );
    _taglineOpacity = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(
        parent: _logoController,
        curve: const Interval(0.55, 1.0, curve: Curves.easeIn),
      ),
    );
    _taglineSlide = Tween<double>(begin: 12, end: 0).animate(
      CurvedAnimation(
        parent: _logoController,
        curve: const Interval(0.55, 1.0, curve: Curves.easeOut),
      ),
    );
    _orbit = Tween<double>(begin: 0, end: 2 * math.pi).animate(_orbitController);
    _pulse = Tween<double>(begin: 0.85, end: 1.0).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    _logoController.forward();

    // Navigate to welcome page after 2.2s
    Future.delayed(const Duration(milliseconds: 2200), () {
      if (mounted) {
        Navigator.pushReplacementNamed(context, AppRoutes.welcome);
      }
    });
  }

  @override
  void dispose() {
    _logoController.dispose();
    _orbitController.dispose();
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bgDark,
      body: Container(
        decoration: const BoxDecoration(gradient: AppColors.heroGradient),
        child: Stack(
          children: [
            // Background star particles
            const _StarField(),

            // Center content
            Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Logo + orbit rings
                  AnimatedBuilder(
                    animation: Listenable.merge([
                      _logoController,
                      _orbitController,
                      _pulseController,
                    ]),
                    builder: (context, _) {
                      return FadeTransition(
                        opacity: _logoOpacity,
                        child: ScaleTransition(
                          scale: _logoScale,
                          child: SizedBox(
                            width: 160,
                            height: 160,
                            child: Stack(
                              alignment: Alignment.center,
                              children: [
                                // Outer glow
                                Transform.scale(
                                  scale: _pulse.value,
                                  child: Container(
                                    width: 140,
                                    height: 140,
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      boxShadow: [
                                        BoxShadow(
                                          color: AppColors.primary
                                              .withValues(alpha: 0.35),
                                          blurRadius: 60,
                                          spreadRadius: 20,
                                        ),
                                      ],
                                    ),
                                  ),
                                ),

                                // Orbit ring 1
                                Transform.rotate(
                                  angle: _orbit.value,
                                  child: _OrbitRing(
                                    radius: 62,
                                    color: AppColors.primaryLight
                                        .withValues(alpha: 0.25),
                                    dotColor: AppColors.primaryLight,
                                    dotAngle: 0,
                                  ),
                                ),

                                // Orbit ring 2 (counter-rotating)
                                Transform.rotate(
                                  angle: -_orbit.value * 0.6,
                                  child: _OrbitRing(
                                    radius: 50,
                                    color: AppColors.secondary
                                        .withValues(alpha: 0.18),
                                    dotColor: AppColors.secondary,
                                    dotAngle: math.pi / 3,
                                  ),
                                ),

                                // Logo container
                                Container(
                                  width: 76,
                                  height: 76,
                                  decoration: BoxDecoration(
                                    shape: BoxShape.circle,
                                    gradient: const LinearGradient(
                                      colors: [
                                        AppColors.primary,
                                        AppColors.primaryLight,
                                      ],
                                      begin: Alignment.topLeft,
                                      end: Alignment.bottomRight,
                                    ),
                                    boxShadow: [
                                      BoxShadow(
                                        color: AppColors.primary
                                            .withValues(alpha: 0.6),
                                        blurRadius: 20,
                                        offset: const Offset(0, 6),
                                      ),
                                    ],
                                  ),
                                  child: const Icon(
                                    Icons.auto_awesome,
                                    size: 36,
                                    color: Colors.white,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      );
                    },
                  ),

                  const SizedBox(height: 28),

                  // App name
                  AnimatedBuilder(
                    animation: _logoController,
                    builder: (context, _) {
                      return FadeTransition(
                        opacity: _logoOpacity,
                        child: ShaderMask(
                          shaderCallback: (bounds) =>
                              AppColors.brandGradient.createShader(bounds),
                          child: const Text(
                            'MemoryVerse',
                            style: TextStyle(
                              fontSize: 34,
                              fontWeight: FontWeight.w800,
                              color: Colors.white,
                              letterSpacing: -0.5,
                            ),
                          ),
                        ),
                      );
                    },
                  ),

                  const SizedBox(height: 10),

                  // Tagline
                  AnimatedBuilder(
                    animation: _logoController,
                    builder: (context, _) {
                      return Transform.translate(
                        offset: Offset(0, _taglineSlide.value),
                        child: FadeTransition(
                          opacity: _taglineOpacity,
                          child: const Text(
                            'Your memories, intelligently woven.',
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w400,
                              color: AppColors.textSecondary,
                              letterSpacing: 0.2,
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),

            // Bottom loading indicator
            Positioned(
              bottom: 52,
              left: 0,
              right: 0,
              child: AnimatedBuilder(
                animation: _logoController,
                builder: (_, __) => FadeTransition(
                  opacity: _taglineOpacity,
                  child: Center(
                    child: SizedBox(
                      width: 32,
                      height: 32,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: AppColors.primary.withValues(alpha: 0.6),
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────
//  Orbit Ring Widget
// ─────────────────────────────────────────────

class _OrbitRing extends StatelessWidget {
  final double radius;
  final Color color;
  final Color dotColor;
  final double dotAngle;

  const _OrbitRing({
    required this.radius,
    required this.color,
    required this.dotColor,
    required this.dotAngle,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: radius * 2,
      height: radius * 2,
      child: CustomPaint(
        painter: _OrbitPainter(
          radius: radius,
          ringColor: color,
          dotColor: dotColor,
          dotAngle: dotAngle,
        ),
      ),
    );
  }
}

class _OrbitPainter extends CustomPainter {
  final double radius;
  final Color ringColor;
  final Color dotColor;
  final double dotAngle;

  _OrbitPainter({
    required this.radius,
    required this.ringColor,
    required this.dotColor,
    required this.dotAngle,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);

    // Ring
    final ringPaint = Paint()
      ..color = ringColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;
    canvas.drawCircle(center, radius - 2, ringPaint);

    // Dot
    final dotX = center.dx + (radius - 2) * math.cos(dotAngle);
    final dotY = center.dy + (radius - 2) * math.sin(dotAngle);
    final dotPaint = Paint()..color = dotColor;
    canvas.drawCircle(Offset(dotX, dotY), 3.5, dotPaint);
  }

  @override
  bool shouldRepaint(covariant _OrbitPainter old) =>
      old.dotAngle != dotAngle || old.ringColor != ringColor;
}

// ─────────────────────────────────────────────
//  Star Field Background
// ─────────────────────────────────────────────

class _StarField extends StatelessWidget {
  const _StarField();

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: MediaQuery.of(context).size,
      painter: _StarFieldPainter(),
    );
  }
}

class _StarFieldPainter extends CustomPainter {
  // Fixed pseudo-random star positions
  static final List<Offset> _stars = List.generate(
    70,
    (i) => Offset(
      (i * 137.5) % 400,
      (i * 73.1 + i * i * 0.3) % 900,
    ),
  );

  static final List<double> _sizes = List.generate(
    70,
    (i) => 0.8 + (i % 3) * 0.6,
  );

  static final List<double> _opacities = List.generate(
    70,
    (i) => 0.2 + (i % 5) * 0.07,
  );

  @override
  void paint(Canvas canvas, Size size) {
    for (int i = 0; i < _stars.length; i++) {
      final x = _stars[i].dx / 400 * size.width;
      final y = _stars[i].dy / 900 * size.height;
      final paint = Paint()
        ..color = Colors.white.withValues(alpha: _opacities[i]);
      canvas.drawCircle(Offset(x, y), _sizes[i], paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter old) => false;
}
