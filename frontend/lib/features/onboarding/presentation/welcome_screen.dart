import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:memory_verse/config/theme/app_theme.dart';
import 'package:memory_verse/app.dart';

// ─────────────────────────────────────────────
//  Welcome / Landing Screen
// ─────────────────────────────────────────────

class WelcomeScreen extends StatefulWidget {
  const WelcomeScreen({super.key});

  @override
  State<WelcomeScreen> createState() => _WelcomeScreenState();
}

class _WelcomeScreenState extends State<WelcomeScreen>
    with TickerProviderStateMixin {
  late AnimationController _entranceController;
  late AnimationController _floatController;
  late AnimationController _shimmerController;

  late Animation<double> _heroOpacity;
  late Animation<Offset> _heroSlide;
  late Animation<double> _cardsOpacity;
  late Animation<double> _cardsSlide;
  late Animation<double> _buttonsOpacity;
  late Animation<double> _buttonsSlide;
  late Animation<double> _float;
  late Animation<double> _shimmer;

  @override
  void initState() {
    super.initState();

    _entranceController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    );

    _floatController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 3500),
    )..repeat(reverse: true);

    _shimmerController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2200),
    )..repeat();

    _heroOpacity = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(
        parent: _entranceController,
        curve: const Interval(0, 0.5, curve: Curves.easeOut),
      ),
    );
    _heroSlide = Tween<Offset>(
      begin: const Offset(0, -0.1),
      end: Offset.zero,
    ).animate(
      CurvedAnimation(
        parent: _entranceController,
        curve: const Interval(0, 0.5, curve: Curves.easeOut),
      ),
    );
    _cardsOpacity = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(
        parent: _entranceController,
        curve: const Interval(0.3, 0.75, curve: Curves.easeOut),
      ),
    );
    _cardsSlide = Tween<double>(begin: 30, end: 0).animate(
      CurvedAnimation(
        parent: _entranceController,
        curve: const Interval(0.3, 0.75, curve: Curves.easeOut),
      ),
    );
    _buttonsOpacity = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(
        parent: _entranceController,
        curve: const Interval(0.65, 1.0, curve: Curves.easeOut),
      ),
    );
    _buttonsSlide = Tween<double>(begin: 24, end: 0).animate(
      CurvedAnimation(
        parent: _entranceController,
        curve: const Interval(0.65, 1.0, curve: Curves.easeOut),
      ),
    );
    _float = Tween<double>(begin: -6, end: 6).animate(
      CurvedAnimation(parent: _floatController, curve: Curves.easeInOut),
    );
    _shimmer = Tween<double>(begin: -2, end: 2).animate(_shimmerController);

    _entranceController.forward();
  }

  @override
  void dispose() {
    _entranceController.dispose();
    _floatController.dispose();
    _shimmerController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;

    return Scaffold(
      backgroundColor: AppColors.bgDark,
      body: Stack(
        children: [
          // Background gradient blobs
          Positioned(
            top: -80,
            left: -60,
            child: _GradientBlob(
              color: AppColors.primary.withValues(alpha: 0.22),
              size: 280,
            ),
          ),
          Positioned(
            top: size.height * 0.35,
            right: -80,
            child: _GradientBlob(
              color: AppColors.secondary.withValues(alpha: 0.15),
              size: 220,
            ),
          ),
          Positioned(
            bottom: 80,
            left: -40,
            child: _GradientBlob(
              color: AppColors.teal.withValues(alpha: 0.10),
              size: 180,
            ),
          ),

          // Main scrollable content
          SafeArea(
            child: Column(
              children: [
                Expanded(
                  child: SingleChildScrollView(
                    physics: const BouncingScrollPhysics(),
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 24),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.center,
                        children: [
                          const SizedBox(height: 36),

                          // ── Hero Section ──────────────────────────────────
                          AnimatedBuilder(
                            animation: _entranceController,
                            builder: (_, child) => FadeTransition(
                              opacity: _heroOpacity,
                              child: SlideTransition(
                                position: _heroSlide,
                                child: child,
                              ),
                            ),
                            child: _buildHeroSection(),
                          ),

                          const SizedBox(height: 40),

                          // ── Floating Memory Cards ─────────────────────────
                          AnimatedBuilder(
                            animation: _floatController,
                            builder: (_, __) {
                              return AnimatedBuilder(
                                animation: _entranceController,
                                builder: (_, child) => FadeTransition(
                                  opacity: _cardsOpacity,
                                  child: Transform.translate(
                                    offset: Offset(0, _cardsSlide.value),
                                    child: child,
                                  ),
                                ),
                                child: _buildMemoryCards(),
                              );
                            },
                          ),

                          const SizedBox(height: 40),

                          // ── Feature Pills ─────────────────────────────────
                          AnimatedBuilder(
                            animation: _entranceController,
                            builder: (_, child) => FadeTransition(
                              opacity: _cardsOpacity,
                              child: Transform.translate(
                                offset: Offset(0, _cardsSlide.value),
                                child: child,
                              ),
                            ),
                            child: _buildFeaturePills(),
                          ),

                          const SizedBox(height: 48),
                        ],
                      ),
                    ),
                  ),
                ),

                // ── CTA Buttons ───────────────────────────────────────
                AnimatedBuilder(
                  animation: _entranceController,
                  builder: (_, child) => FadeTransition(
                    opacity: _buttonsOpacity,
                    child: Transform.translate(
                      offset: Offset(0, _buttonsSlide.value),
                      child: child,
                    ),
                  ),
                  child: _buildCTAButtons(context),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── Hero Section ───────────────────────────────────────────────────

  Widget _buildHeroSection() {
    return Column(
      children: [
        // Logo badge
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
          decoration: BoxDecoration(
            color: AppColors.primary.withValues(alpha: 0.15),
            borderRadius: BorderRadius.circular(40),
            border: Border.all(
              color: AppColors.primaryLight.withValues(alpha: 0.35),
              width: 1,
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: const [
              Icon(Icons.auto_awesome, size: 14, color: AppColors.primaryLight),
              SizedBox(width: 6),
              Text(
                'AI-Powered Memory Vault',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                  color: AppColors.primaryLight,
                  letterSpacing: 0.3,
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 20),

        // Main headline
        ShaderMask(
          shaderCallback: (bounds) =>
              AppColors.brandGradient.createShader(bounds),
          child: const Text(
            'Relive Your\nBest Moments',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 40,
              fontWeight: FontWeight.w800,
              color: Colors.white,
              letterSpacing: -1.0,
              height: 1.1,
            ),
          ),
        ),

        const SizedBox(height: 16),

        // Subheadline
        const Text(
          'MemoryVerse uses AI to transform your scattered\nphotos into cinematic stories — automatically.',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 15,
            color: AppColors.textSecondary,
            height: 1.65,
          ),
        ),
      ],
    );
  }

  // ── Memory Cards ───────────────────────────────────────────────────

  Widget _buildMemoryCards() {
    return AnimatedBuilder(
      animation: _floatController,
      builder: (_, __) {
        return SizedBox(
          height: 190,
          child: Stack(
            alignment: Alignment.center,
            clipBehavior: Clip.none,
            children: [
              // Card: left (tilted back)
              Positioned(
                left: 12,
                child: Transform.rotate(
                  angle: -0.18,
                  child: Transform.translate(
                    offset: Offset(0, _float.value * 0.5),
                    child: _MemoryCard(
                      icon: Icons.beach_access_rounded,
                      label: 'Beach Trip',
                      date: 'Jun 2024',
                      color: const Color(0xFFFF7E5F),
                      tagColor: const Color(0xFFFF7E5F),
                      width: 140,
                      height: 170,
                    ),
                  ),
                ),
              ),

              // Card: right (tilted back)
              Positioned(
                right: 12,
                child: Transform.rotate(
                  angle: 0.16,
                  child: Transform.translate(
                    offset: Offset(0, -_float.value * 0.5),
                    child: _MemoryCard(
                      icon: Icons.celebration_rounded,
                      label: 'Birthday',
                      date: 'Mar 2025',
                      color: AppColors.gold,
                      tagColor: AppColors.gold,
                      width: 140,
                      height: 170,
                    ),
                  ),
                ),
              ),

              // Card: center (front, elevated)
              Transform.translate(
                offset: Offset(0, _float.value),
                child: _MemoryCard(
                  icon: Icons.landscape_rounded,
                  label: 'Mountain Hike',
                  date: 'Aug 2025',
                  color: AppColors.teal,
                  tagColor: AppColors.teal,
                  width: 160,
                  height: 185,
                  isFeatured: true,
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  // ── Feature Pills ──────────────────────────────────────────────────

  Widget _buildFeaturePills() {
    final features = [
      (Icons.psychology_rounded, 'Smart AI Tagging'),
      (Icons.group_rounded, 'Shared Vaults'),
      (Icons.movie_creation_rounded, 'Auto Story'),
      (Icons.lock_rounded, 'Private & Secure'),
    ];

    return Wrap(
      spacing: 10,
      runSpacing: 10,
      alignment: WrapAlignment.center,
      children: features.map((f) {
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          decoration: BoxDecoration(
            color: AppColors.bgCard,
            borderRadius: BorderRadius.circular(40),
            border: Border.all(
              color: AppColors.primaryLight.withValues(alpha: 0.15),
              width: 1,
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(f.$1, size: 15, color: AppColors.primaryLight),
              const SizedBox(width: 7),
              Text(
                f.$2,
                style: const TextStyle(
                  fontSize: 13,
                  color: AppColors.textSecondary,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }

  // ── CTA Buttons ────────────────────────────────────────────────────

  Widget _buildCTAButtons(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(24, 20, 24, 32),
      decoration: BoxDecoration(
        color: AppColors.bgDark,
        border: Border(
          top: BorderSide(
            color: AppColors.primaryLight.withValues(alpha: 0.1),
            width: 1,
          ),
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Get Started (Sign Up)
          _GradientButton(
            label: 'Get Started',
            icon: Icons.arrow_forward_rounded,
            onTap: () => Navigator.pushNamed(
              context,
              AppRoutes.auth,
              arguments: true, // open in sign-up mode
            ),
          ),

          const SizedBox(height: 12),

          // Sign In (outlined)
          _OutlineButton(
            label: 'I Already Have an Account',
            onTap: () => Navigator.pushNamed(
              context,
              AppRoutes.auth,
              arguments: false, // open in sign-in mode
            ),
          ),

          const SizedBox(height: 14),

          // Legal note
          const Text(
            'By continuing, you agree to our Terms & Privacy Policy.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 11,
              color: AppColors.textHint,
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────
//  Memory Card Widget
// ─────────────────────────────────────────────

class _MemoryCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final String date;
  final Color color;
  final Color tagColor;
  final double width;
  final double height;
  final bool isFeatured;

  const _MemoryCard({
    required this.icon,
    required this.label,
    required this.date,
    required this.color,
    required this.tagColor,
    required this.width,
    required this.height,
    this.isFeatured = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: width,
      height: height,
      decoration: BoxDecoration(
        color: AppColors.bgCard,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: isFeatured
              ? color.withValues(alpha: 0.55)
              : AppColors.primaryLight.withValues(alpha: 0.12),
          width: isFeatured ? 1.5 : 1,
        ),
        boxShadow: [
          BoxShadow(
            color: isFeatured
                ? color.withValues(alpha: 0.22)
                : Colors.black.withValues(alpha: 0.3),
            blurRadius: isFeatured ? 28 : 12,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Icon box
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.18),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, size: 22, color: color),
            ),

            const Spacer(),

            // Memory label
            Text(
              label,
              style: const TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.w600,
                color: AppColors.textPrimary,
              ),
            ),
            const SizedBox(height: 4),

            // Date
            Row(
              children: [
                Icon(Icons.calendar_today_rounded,
                    size: 11, color: AppColors.textHint),
                const SizedBox(width: 4),
                Text(
                  date,
                  style: const TextStyle(
                    fontSize: 11,
                    color: AppColors.textHint,
                  ),
                ),
              ],
            ),

            if (isFeatured) ...[
              const SizedBox(height: 10),
              // AI Story pill
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  gradient: AppColors.buttonGradient,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.auto_awesome, size: 11, color: Colors.white),
                    SizedBox(width: 4),
                    Text(
                      'Story Ready',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: Colors.white,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────
//  Gradient Button
// ─────────────────────────────────────────────

class _GradientButton extends StatefulWidget {
  final String label;
  final IconData icon;
  final VoidCallback onTap;

  const _GradientButton({
    required this.label,
    required this.icon,
    required this.onTap,
  });

  @override
  State<_GradientButton> createState() => _GradientButtonState();
}

class _GradientButtonState extends State<_GradientButton> {
  bool _pressed = false;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => setState(() => _pressed = true),
      onTapUp: (_) {
        setState(() => _pressed = false);
        widget.onTap();
      },
      onTapCancel: () => setState(() => _pressed = false),
      child: AnimatedScale(
        scale: _pressed ? 0.97 : 1.0,
        duration: const Duration(milliseconds: 100),
        child: Container(
          height: 54,
          decoration: BoxDecoration(
            gradient: AppColors.buttonGradient,
            borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                color: AppColors.primary.withValues(alpha: 0.45),
                blurRadius: 20,
                offset: const Offset(0, 8),
              ),
            ],
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                widget.label,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                  letterSpacing: 0.3,
                ),
              ),
              const SizedBox(width: 8),
              Icon(widget.icon, size: 18, color: Colors.white),
            ],
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────
//  Outline Button
// ─────────────────────────────────────────────

class _OutlineButton extends StatefulWidget {
  final String label;
  final VoidCallback onTap;

  const _OutlineButton({required this.label, required this.onTap});

  @override
  State<_OutlineButton> createState() => _OutlineButtonState();
}

class _OutlineButtonState extends State<_OutlineButton> {
  bool _pressed = false;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => setState(() => _pressed = true),
      onTapUp: (_) {
        setState(() => _pressed = false);
        widget.onTap();
      },
      onTapCancel: () => setState(() => _pressed = false),
      child: AnimatedScale(
        scale: _pressed ? 0.97 : 1.0,
        duration: const Duration(milliseconds: 100),
        child: Container(
          height: 54,
          decoration: BoxDecoration(
            color: AppColors.bgCard,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: AppColors.primaryLight.withValues(alpha: 0.35),
              width: 1.5,
            ),
          ),
          child: Center(
            child: Text(
              widget.label,
              style: const TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.w600,
                color: AppColors.textSecondary,
                letterSpacing: 0.2,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────
//  Gradient Blob
// ─────────────────────────────────────────────

class _GradientBlob extends StatelessWidget {
  final Color color;
  final double size;

  const _GradientBlob({required this.color, required this.size});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(
            color: color,
            blurRadius: size * 0.8,
            spreadRadius: size * 0.2,
          ),
        ],
      ),
    );
  }
}
