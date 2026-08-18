import 'package:flutter/material.dart';

import 'package:memory_verse/core/theme/app_design_tokens.dart';
import 'package:memory_verse/core/presentation/widgets/stacked_carousel.dart';
import 'package:memory_verse/core/presentation/widgets/flow_button.dart';
import 'package:memory_verse/core/presentation/widgets/aurora_background.dart';

class AuroraHero extends StatelessWidget {
  final String appName;
  final String badgeText;
  final String headline;
  final String subheadline;
  final String primaryCtaLabel;
  final String secondaryCtaLabel;
  final VoidCallback? onPrimaryCta;
  final VoidCallback? onSecondaryCta;
  final VoidCallback? onSignIn;

  const AuroraHero({
    super.key,
    this.appName = 'MemoryVerse',
    this.badgeText = 'Now with shared timelines',
    this.headline = 'Every memory, worth revisiting',
    this.subheadline = 'Collect, relive, and share your story with the people who were there for it.',
    this.primaryCtaLabel = 'Start free',
    this.secondaryCtaLabel = 'See how it works',
    this.onPrimaryCta,
    this.onSecondaryCta,
    this.onSignIn,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          const Positioned.fill(child: AuroraBackground(isDark: true)),
          SafeArea(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(
                horizontal: AppSpacing.s28,
                vertical: AppSpacing.s20,
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _buildNav(),
                  const SizedBox(height: 56),
                  _buildHeroContent(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNav() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Row(
          children: [
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.15),
                borderRadius: BorderRadius.circular(AppRadius.md),
              ),
              child: const Icon(
                Icons.auto_awesome,
                size: 20,
                color: Colors.white,
              ),
            ),
            const SizedBox(width: AppSpacing.s12),
            Text(
              appName,
              style: const TextStyle(
                fontFamily: 'Inter',
                fontSize: 20,
                fontWeight: FontWeight.w700,
                color: Colors.white,
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildHeroContent() {
    return Column(
      children: [
        const SizedBox(height: AppSpacing.s20),
        Text(
          headline,
          textAlign: TextAlign.center,
          style: const TextStyle(
            fontFamily: 'Inter',
            fontSize: 36,
            fontWeight: FontWeight.w700,
            letterSpacing: -1.2,
            height: 1.05,
            color: Colors.white,
          ),
        ),
        const SizedBox(height: 14),
        Text(
          subheadline,
          textAlign: TextAlign.center,
          style: AppTextStyles.body.copyWith(color: AppColors.onDarkMuted),
        ),
        const SizedBox(height: AppSpacing.s32),
        StackedCarousel(),
        const SizedBox(height: AppSpacing.s48),
        Center(
          child: FlowButton(
            text: "Let's get started",
            isDark: true,
            onPressed: onPrimaryCta ?? () {},
          ),
        ),
      ],
    );
  }
}
