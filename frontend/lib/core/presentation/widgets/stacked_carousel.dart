import 'package:cached_network_image/cached_network_image.dart';
import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;

class CarouselItem {
  final String? id;
  final String title;
  final String? subtitle;
  final String imageUrl;
  final String? badge;

  const CarouselItem({
    this.id,
    required this.title,
    this.subtitle,
    required this.imageUrl,
    this.badge,
  });
}

class StackedCarousel extends StatefulWidget {
  final List<CarouselItem> items;
  final bool showIndicators;
  final double height;
  final void Function(CarouselItem)? onItemTap;

  const StackedCarousel({
    super.key,
    required this.items,
    this.showIndicators = false,
    this.height = 210,
    this.onItemTap,
  });

  @override
  State<StackedCarousel> createState() => _StackedCarouselState();
}

class _StackedCarouselState extends State<StackedCarousel> {
  int _currentIndex = 0;

  @override
  Widget build(BuildContext context) {
    if (widget.items.isEmpty) {
      return SizedBox(height: widget.height);
    }

    final bool isSingleItem = widget.items.length == 1;

    Widget carousel = CarouselSlider.builder(
      itemCount: widget.items.length,
      options: CarouselOptions(
        height: widget.height,
        enlargeCenterPage: true,
        enlargeFactor: 0.18,
        viewportFraction: isSingleItem ? 0.92 : 0.82,
        enableInfiniteScroll: !isSingleItem,
        autoPlay: !isSingleItem,
        autoPlayCurve: Curves.fastOutSlowIn,
        autoPlayAnimationDuration: const Duration(milliseconds: 800),
        onPageChanged: (index, reason) {
          setState(() {
            _currentIndex = index;
          });
        },
      ),
      itemBuilder: (context, index, realIndex) {
        final item = widget.items[index];
        return GestureDetector(
          onTap: () => widget.onItemTap?.call(item),
          child: Container(
            margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 4),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(20),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.35),
                  blurRadius: 14,
                  offset: const Offset(0, 6),
                ),
              ],
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  // Photo with cached loading & error fallback
                  CachedNetworkImage(
                    imageUrl: item.imageUrl,
                    fit: BoxFit.cover,
                    placeholder: (context, url) => Container(
                      color: adt.AppColors.plum900,
                      child: const Center(
                        child: SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                      ),
                    ),
                    errorWidget: (context, url, error) => Container(
                      color: adt.AppColors.plum900,
                      child: Center(
                        child: Icon(
                          Icons.image_outlined,
                          size: 36,
                          color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.3),
                        ),
                      ),
                    ),
                  ),

                  // Bottom dark gradient overlay for crystal clear text readability
                  Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [
                          Colors.transparent,
                          Colors.black.withValues(alpha: 0.1),
                          Colors.black.withValues(alpha: 0.8),
                        ],
                        stops: const [0.35, 0.65, 1.0],
                      ),
                    ),
                  ),

                  // Subtle card border
                  Container(
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: Colors.white.withValues(alpha: 0.12),
                        width: 1,
                      ),
                    ),
                  ),

                  // Content Overlay
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        if (item.badge != null)
                          Align(
                            alignment: Alignment.topRight,
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 10,
                                vertical: 4,
                              ),
                              decoration: BoxDecoration(
                                color: Colors.black.withValues(alpha: 0.45),
                                borderRadius: BorderRadius.circular(16),
                                border: Border.all(
                                  color: Colors.white.withValues(alpha: 0.3),
                                ),
                              ),
                              child: Text(
                                item.badge!.toUpperCase(),
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 10,
                                  fontWeight: FontWeight.w700,
                                  letterSpacing: 1.0,
                                ),
                              ),
                            ),
                          ),
                        const Spacer(),
                        Text(
                          item.title,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 18,
                            fontWeight: FontWeight.w700,
                            letterSpacing: -0.2,
                            shadows: [
                              Shadow(
                                color: Colors.black54,
                                blurRadius: 6,
                                offset: Offset(0, 1),
                              ),
                            ],
                          ),
                        ),
                        if (item.subtitle != null && item.subtitle!.isNotEmpty) ...[
                          const SizedBox(height: 3),
                          Row(
                            children: [
                              Icon(
                                Icons.location_on_outlined,
                                size: 13,
                                color: Colors.white.withValues(alpha: 0.85),
                              ),
                              const SizedBox(width: 4),
                              Expanded(
                                child: Text(
                                  item.subtitle!,
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                  style: TextStyle(
                                    color: Colors.white.withValues(alpha: 0.85),
                                    fontSize: 12,
                                    fontWeight: FontWeight.w400,
                                    shadows: const [
                                      Shadow(
                                        color: Colors.black54,
                                        blurRadius: 4,
                                        offset: Offset(0, 1),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );

    if (!isSingleItem) {
      carousel = ShaderMask(
        shaderCallback: (Rect bounds) {
          return const LinearGradient(
            colors: [
              Colors.transparent,
              adt.AppColors.plum900,
              adt.AppColors.plum900,
              Colors.transparent,
            ],
            stops: [0.0, 0.05, 0.95, 1.0],
          ).createShader(bounds);
        },
        blendMode: BlendMode.dstIn,
        child: carousel,
      );
    }

    return Column(
      children: [
        carousel,
        if (widget.showIndicators && !isSingleItem) ...[
          const SizedBox(height: 14),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: widget.items.asMap().entries.map((entry) {
              return Container(
                width: 7.0,
                height: 7.0,
                margin: const EdgeInsets.symmetric(horizontal: 4.0),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _currentIndex == entry.key
                      ? Theme.of(context).colorScheme.primary
                      : Theme.of(context).colorScheme.primary.withValues(alpha: 0.3),
                ),
              );
            }).toList(),
          ),
        ]
      ],
    );
  }
}
