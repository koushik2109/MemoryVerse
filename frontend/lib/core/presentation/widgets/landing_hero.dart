import 'package:flutter/material.dart';

import '../../theme/app_design_tokens.dart';
import 'stacked_carousel.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;

class LandingStat {
  final String value;
  final String label;

  const LandingStat({required this.value, required this.label});
}

class LandingHero extends StatelessWidget {
  final String appName;
  final String headline;
  final String subheadline;
  final String primaryCtaLabel;
  final VoidCallback? onPrimaryCta;
  final List<CarouselItem> carouselItems;
  final List<LandingStat> stats;
  final void Function(CarouselItem item)? onCarouselItemTap;

  const LandingHero({
    super.key,
    this.appName = 'MemoryVerse',
    this.headline = 'Every memory, worth revisiting',
    this.subheadline =
        'Collect, relive, and share your story with the people who were there for it.',
    this.primaryCtaLabel = "Let's get started",
    this.onPrimaryCta,
    this.carouselItems = const [],
    this.stats = const [
      LandingStat(value: "10k+", label: "Memories"),
      LandingStat(value: "50+", label: "Vaults"),
      LandingStat(value: "4.9", label: "Rating"),
    ],
    this.onCarouselItemTap,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SizedBox.expand(
        child: Stack(
          children: [
            // Same constant the auth screens use — not a new gradient.
            const Positioned.fill(
              child: DecoratedBox(decoration: BoxDecoration(gradient: AppGradients.dark)),
            ),
            // Mandatory on the dark scheme — do not remove.
            const Positioned.fill(
              child: DecoratedBox(decoration: BoxDecoration(gradient: AppGradients.scrim)),
            ),
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
                    const SizedBox(height: AppSpacing.s48),
                    _buildHeroContent(),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildNav() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.start,
      children: [
        Container(
          width: 32,
          height: 32,
          decoration: BoxDecoration(
            color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.15),
            borderRadius: BorderRadius.circular(AppRadius.sm),
          ),
          child: const Icon(Icons.auto_awesome, size: 18, color: AppColors.onDarkPrimary),
        ),
        const SizedBox(width: AppSpacing.s8),
        Text(
          appName,
          style: AppTextStyles.h2.copyWith(
            fontWeight: FontWeight.w700,
            color: AppColors.onDarkPrimary,
          ),
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
          style: AppTextStyles.display.copyWith(
            color: AppColors.onDarkPrimary,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: AppSpacing.s16),
        Text(
          subheadline,
          textAlign: TextAlign.center,
          style: AppTextStyles.body.copyWith(
            color: AppColors.onDarkSecondary,
            fontWeight: FontWeight.w500,
          ),
        ),
        if (carouselItems.isNotEmpty) ...[
          const SizedBox(height: AppSpacing.s32),
          StackedCarousel(
            items: carouselItems,
            showIndicators: false, // browse mode, not onboarding
            onItemTap: onCarouselItemTap,
          ),
        ],
        const SizedBox(height: AppSpacing.s48),
        Center(
          child: ElevatedButton(
            onPressed: onPrimaryCta,
            style: AppTheme.primaryButtonOnDark.copyWith(
              padding: WidgetStateProperty.all(
                const EdgeInsets.symmetric(horizontal: 32, vertical: 20),
              ),
              textStyle: WidgetStateProperty.all(
                AppTextStyles.body.copyWith(fontWeight: FontWeight.w600, fontSize: 16),
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(primaryCtaLabel),
                const SizedBox(width: 8),
                const Icon(Icons.arrow_forward, size: 20),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
