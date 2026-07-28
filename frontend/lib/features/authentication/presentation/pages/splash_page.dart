import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:go_router/go_router.dart';
import 'package:supabase_flutter/supabase_flutter.dart' as sb;
import 'package:frontend/shared/constants/app_spacing.dart';

class SplashPage extends StatefulWidget {
  const SplashPage({super.key});

  @override
  State<SplashPage> createState() => _SplashPageState();
}

class _SplashPageState extends State<SplashPage>
    with SingleTickerProviderStateMixin {
  static const Duration _kNavDelay = Duration(milliseconds: 2600);

  late final AnimationController _bgController;
  late final Animation<Alignment> _beginAlign;
  late final Animation<Alignment> _endAlign;

  @override
  void initState() {
    super.initState();

    _bgController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 6),
    )..repeat(reverse: true);

    _beginAlign = TweenSequence<Alignment>([
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.topLeft, end: Alignment.topRight),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.topRight, end: Alignment.topLeft),
        weight: 1,
      ),
    ]).animate(_bgController);

    _endAlign = TweenSequence<Alignment>([
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.bottomRight, end: Alignment.bottomLeft),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.bottomLeft, end: Alignment.bottomRight),
        weight: 1,
      ),
    ]).animate(_bgController);

    Future.delayed(_kNavDelay, () {
      if (!mounted) return;
      // If already authenticated, go directly to home.
      final session = sb.Supabase.instance.client.auth.currentSession;
      if (session != null) {
        context.go('/home');
      } else {
        context.go('/login');
      }
    });
  }

  @override
  void dispose() {
    _bgController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      body: AnimatedBuilder(
        animation: _bgController,
        builder: (context, child) => Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: _beginAlign.value,
              end: _endAlign.value,
              colors: const [
                Color(0xFF030712), // Cosmic Void
                Color(0xFF2E1065), // Nebula Violet
                Color(0xFF0F172A), // Deep Space Blue
                Color(0xFF083344), // Starlight Cyan Horizon
              ],
              stops: const [0.0, 0.35, 0.65, 1.0],
            ),
          ),
          child: child,
        ),
        child: SafeArea(
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // ── Logo ────────────────────────────────────────────────────
                Image.asset(
                  isDark
                      ? 'assets/icons/memoryverse_darktheme.png'
                      : 'assets/icons/memoryverse_lighttheme.png',
                  width: 100,
                  height: 100,
                  errorBuilder: (_, e, s) => Icon(
                    Icons.auto_stories_rounded,
                    size: 80,
                    color: cs.onPrimary,
                  ),
                )
                    .animate()
                    .fadeIn(duration: 800.ms)
                    .scale(
                      begin: const Offset(0.5, 0.5),
                      duration: 800.ms,
                      curve: Curves.easeOutBack,
                    ),

                const SizedBox(height: AppSpacing.lg),

                Text(
                  'MemoryVerse',
                  style: tt.displaySmall?.copyWith(
                    color: cs.onPrimary,
                    fontWeight: FontWeight.w800,
                    letterSpacing: -1.2,
                  ),
                )
                    .animate()
                    .fadeIn(duration: 600.ms, delay: 300.ms)
                    .slideY(begin: 0.3, end: 0, duration: 600.ms, delay: 300.ms, curve: Curves.easeOutQuint),

                const SizedBox(height: AppSpacing.xs),

                Text(
                  'Capture life\'s story, one verse at a time.',
                  style: tt.bodyLarge?.copyWith(
                    color: cs.onPrimary.withValues(alpha: 0.8),
                    letterSpacing: 0.1,
                  ),
                  textAlign: TextAlign.center,
                )
                    .animate()
                    .fadeIn(duration: 500.ms, delay: 500.ms),

                const SizedBox(height: AppSpacing.xxxl),

                SizedBox(
                  width: 24,
                  height: 24,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    valueColor: AlwaysStoppedAnimation<Color>(
                      cs.onPrimary.withValues(alpha: 0.65),
                    ),
                  ),
                ).animate().fadeIn(duration: 400.ms, delay: 900.ms),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
