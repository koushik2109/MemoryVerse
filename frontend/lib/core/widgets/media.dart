import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;

/// High-performance network image loader with disk caching and low-memory decoding.
class AppNetworkImage extends StatelessWidget {
  final String imageUrl;
  final BoxFit fit;
  final double? width;
  final double? height;
  final int? memCacheWidth;

  const AppNetworkImage({
    super.key,
    required this.imageUrl,
    this.fit = BoxFit.cover,
    this.width,
    this.height,
    this.memCacheWidth = 480,
  });

  @override
  Widget build(BuildContext context) {
    if (imageUrl.isEmpty) {
      return Container(
        width: width,
        height: height,
        color: context.colors.surfaceElevated,
        child: Icon(Icons.image_not_supported_outlined, color: context.colors.textMuted, size: 24),
      );
    }

    return CachedNetworkImage(
      imageUrl: imageUrl,
      width: width,
      height: height,
      fit: fit,
      memCacheWidth: memCacheWidth,
      placeholder: (context, url) => Container(
        width: width,
        height: height,
        color: context.colors.surfaceElevated,
        child: const Center(
          child: SizedBox(
            width: 16,
            height: 16,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
        ),
      ),
      errorWidget: (context, url, error) => Container(
        width: width,
        height: height,
        color: context.colors.surfaceElevated,
        child: Icon(Icons.broken_image_outlined, color: context.colors.textMuted, size: 24),
      ),
    );
  }
}

class MediaGrid extends StatelessWidget {
  final List<String> imageUrls;
  final Function(int)? onMediaTap;

  const MediaGrid({super.key, required this.imageUrls, this.onMediaTap});

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
            child: AppNetworkImage(
              imageUrl: imageUrls[index],
              fit: BoxFit.cover,
              memCacheWidth: 400,
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
          AppNetworkImage(
            imageUrl: imageUrl,
            fit: BoxFit.cover,
            memCacheWidth: 400,
          ),
          Center(
            child: Container(
              padding: const EdgeInsets.all(AppSpacing.s12),
              decoration: BoxDecoration(
                color: adt.AppColors.plum900.withValues(alpha: 0.4),
                shape: BoxShape.circle,
                border: Border.all(color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.2)),
              ),
              child: const Icon(
                Icons.play_arrow_rounded,
                color: adt.AppColors.onDarkPrimary,
                size: 28,
              ),
            ),
          ),
          if (duration != null)
            Positioned(
              bottom: AppSpacing.s8,
              right: AppSpacing.s8,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppSpacing.s8,
                  vertical: 2,
                ),
                decoration: BoxDecoration(
                  color: adt.AppColors.plum900.withValues(alpha: 0.6),
                  borderRadius: BorderRadius.circular(AppRadii.sm),
                ),
                child: Text(
                  duration!,
                  style: const TextStyle(
                    color: adt.AppColors.onDarkPrimary,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                  ),
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

  bool _isLoadableUrl(String? value) {
    if (value == null || value.trim().isEmpty) return false;
    final uri = Uri.tryParse(value);
    return uri != null && (uri.scheme == 'http' || uri.scheme == 'https') && uri.host.isNotEmpty;
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final initial = (name ?? 'U').isNotEmpty
        ? (name ?? 'U')[0].toUpperCase()
        : 'U';

    Widget child;
    if (_isLoadableUrl(url)) {
      child = AppNetworkImage(
        imageUrl: url!,
        width: size,
        height: size,
        memCacheWidth: (size * 2).toInt(),
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
