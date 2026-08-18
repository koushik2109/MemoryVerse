import 'package:flutter/material.dart';
import 'package:carousel_slider/carousel_slider.dart';

class StackedCarousel extends StatelessWidget {
  final List<Map<String, String>> memories = [
    {
      "image": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600&q=80",
      "title": "Mountain Trek",
      "description": "Scale new heights and embrace the hiker's journey.",
      "badge": "Adventure",
    },
    {
      "image": "https://images.unsplash.com/photo-1504280390367-361c6d9f38f4?w=600&q=80",
      "title": "River Rafting",
      "description":
          "Feel the adrenaline rush as you navigate the wild rapids.",
      "badge": "Extreme",
    },
    {
      "image": "https://images.unsplash.com/photo-1448375240586-882707db888b?w=600&q=80",
      "title": "Forest Walk",
      "description": "Deep dive into the silence of the ancient woods.",
      "badge": "Nature",
    },
    {
      "image": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=600&q=80",
      "title": "Azure Beach",
      "description":
          "Unwind on the crystal clear shores of a tropical paradise.",
      "badge": "Paradise",
    },
  ];

  StackedCarousel({super.key});

  @override
  Widget build(BuildContext context) {
    return ShaderMask(
      shaderCallback: (Rect bounds) {
        return const LinearGradient(
          colors: [
            Colors.transparent,
            Colors.black,
            Colors.black,
            Colors.transparent,
          ],
          stops: [0.0, 0.08, 0.92, 1.0],
        ).createShader(bounds);
      },
      blendMode: BlendMode.dstIn,
      child: CarouselSlider.builder(
        itemCount: memories.length,
        options: CarouselOptions(
          height: 240,
          enlargeCenterPage: true,
          enlargeFactor: 0.2,
          viewportFraction: 0.7,
          enableInfiniteScroll: true,
          autoPlay: true,
          autoPlayCurve: Curves.fastOutSlowIn,
          autoPlayAnimationDuration: const Duration(milliseconds: 800),
        ),
        itemBuilder: (context, index, realIndex) {
          final memory = memories[index];
          return Container(
            margin: const EdgeInsets.symmetric(vertical: 10),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(24),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.3),
                  blurRadius: 15,
                  offset: const Offset(0, 10),
                ),
              ],
              image: DecorationImage(
                image: NetworkImage(memory['image']!),
                fit: BoxFit.cover,
              ),
            ),
            child: Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(24),
                gradient: LinearGradient(
                  begin: Alignment.bottomCenter,
                  end: Alignment.topCenter,
                  colors: [
                    Colors.black.withOpacity(0.9),
                    Colors.black.withOpacity(0.3),
                    Colors.transparent,
                  ],
                  stops: const [0.0, 0.4, 1.0],
                ),
              ),
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  Align(
                    alignment: Alignment.topRight,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(
                          color: Colors.white.withOpacity(0.4),
                        ),
                      ),
                      child: Text(
                        memory['badge']!.toUpperCase(),
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 10,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 1.2,
                        ),
                      ),
                    ),
                  ),
                  const Spacer(),
                  Text(
                    memory['title']!,
                    style: const TextStyle(
                      fontFamily: 'Inter',
                      color: Colors.white,
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    memory['description']!,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      fontFamily: 'Inter',
                      color: Colors.white.withOpacity(0.8),
                      fontSize: 13,
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
