import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/core/navigation/router.dart';
import 'package:memory_verse/core/presentation/widgets/landing_hero.dart';
import 'package:memory_verse/core/presentation/widgets/stacked_carousel.dart';

class LandingScreen extends StatelessWidget {
  const LandingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return LandingHero(
      carouselItems: const [
        CarouselItem(
          title: "Mountain Trek",
          subtitle: "Scale new heights and embrace the hiker's journey.",
          imageUrl: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600&q=80",
          badge: "Adventure",
        ),
        CarouselItem(
          title: "River Rafting",
          subtitle: "Feel the adrenaline rush as you navigate the wild rapids.",
          imageUrl: "https://images.unsplash.com/photo-1504280390367-361c6d9f38f4?w=600&q=80",
          badge: "Extreme",
        ),
        CarouselItem(
          title: "Forest Walk",
          subtitle: "Deep dive into the silence of the ancient woods.",
          imageUrl: "https://images.unsplash.com/photo-1448375240586-882707db888b?w=600&q=80",
          badge: "Nature",
        ),
        CarouselItem(
          title: "Azure Beach",
          subtitle: "Unwind on the crystal clear shores of a tropical paradise.",
          imageUrl: "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=600&q=80",
          badge: "Paradise",
        ),
      ],
      onPrimaryCta: () {
        context.push(Routes.signUp);
      },
    );
  }
}
