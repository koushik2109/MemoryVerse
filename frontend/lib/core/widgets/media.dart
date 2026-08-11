import 'package:flutter/material.dart';
import 'package:memory_verse/core/design/tokens.dart';

class MediaGrid extends StatelessWidget {
  final List<String> imageUrls;
  final Function(int)? onMediaTap;

  const MediaGrid({
    super.key,
    required this.imageUrls,
    this.onMediaTap,
  });

  @override
  Widget build(BuildContext context) {
    if (imageUrls.isEmpty) return const SizedBox.shrink();

    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 3,
        crossAxisSpacing: 2,
        mainAxisSpacing: 2,
        childAspectRatio: 1,
      ),
      itemCount: imageUrls.length,
      itemBuilder: (context, index) {
        return GestureDetector(
          onTap: () => onMediaTap?.call(index),
          child: Container(
            color: context.colors.surfaceElevated,
            child: Image.network(
              imageUrls[index],
              fit: BoxFit.cover,
              errorBuilder: (_, __, ___) => Icon(Icons.image, color: context.colors.textMuted),
            ),
          ),
        );
      },
    );
  }
}

class VideoThumbnail extends StatelessWidget {
  final String imageUrl;
  final String? duration;
  final VoidCallback? onTap;

  const VideoThumbnail({
    super.key,
    required this.imageUrl,
    this.duration,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Stack(
        fit: StackFit.expand,
        children: [
          Image.network(
            imageUrl,
            fit: BoxFit.cover,
            errorBuilder: (_, __, ___) => Container(color: context.colors.surfaceElevated),
          ),
          Center(
            child: Container(
              padding: const EdgeInsets.all(AppSpacing.s12),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.4),
                shape: BoxShape.circle,
                border: Border.all(color: Colors.white.withValues(alpha: 0.2)),
              ),
              child: const Icon(Icons.play_arrow_rounded, color: Colors.white, size: 28),
            ),
          ),
          if (duration != null)
            Positioned(
              bottom: AppSpacing.s8,
              right: AppSpacing.s8,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s8, vertical: 2),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.6),
                  borderRadius: BorderRadius.circular(AppRadii.sm),
                ),
                child: Text(
                  duration!,
                  style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.w600),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class Avatar extends StatelessWidget {
  final String? url;
  final String? name;
  final double size;
  final bool hasBorder;

  const Avatar({
    super.key,
    this.url,
    this.name,
    this.size = 40,
    this.hasBorder = false,
  });

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final initial = (name ?? 'U').isNotEmpty ? (name ?? 'U')[0].toUpperCase() : 'U';

    Widget child;
    if (url != null && url!.isNotEmpty) {
      child = Image.network(
        url!,
        fit: BoxFit.cover,
        errorBuilder: (_, __, ___) => _fallback(c, initial),
      );
    } else {
      child = _fallback(c, initial);
    }

    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        border: hasBorder ? Border.all(color: c.bg, width: 2) : null,
      ),
      clipBehavior: Clip.antiAlias,
      child: child,
    );
  }

  Widget _fallback(AppColors c, String initial) => Container(
    color: c.surfaceElevated,
    alignment: Alignment.center,
    child: Text(
      initial,
      style: TextStyle(
        fontSize: size * 0.4,
        fontWeight: FontWeight.w600,
        color: c.text,
      ),
    ),
  );
}
