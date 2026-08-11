import 'package:flutter/material.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:intl/intl.dart';

class MemoryCard extends StatelessWidget {
  final String imageUrl;
  final String title;
  final DateTime date;
  final int photoCount;
  final VoidCallback? onTap;

  const MemoryCard({
    super.key,
    required this.imageUrl,
    required this.title,
    required this.date,
    this.photoCount = 0,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    
    return GestureDetector(
      onTap: onTap,
      child: Container(
        height: 240,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(AppRadii.md),
          color: c.surface,
        ),
        clipBehavior: Clip.antiAlias,
        child: Stack(
          fit: StackFit.expand,
          children: [
            Image.network(
              imageUrl,
              fit: BoxFit.cover,
              errorBuilder: (_, __, ___) => Container(color: c.surfaceElevated),
            ),
            // Gradient Overlay
            DecoratedBox(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.transparent,
                    Colors.black.withValues(alpha: 0.7),
                  ],
                ),
              ),
            ),
            Positioned(
              left: AppSpacing.s16,
              right: AppSpacing.s16,
              bottom: AppSpacing.s16,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: context.text.titleLarge?.copyWith(color: Colors.white, fontWeight: FontWeight.w600),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      Text(
                        DateFormat('MMM d, yyyy').format(date),
                        style: context.text.labelMedium?.copyWith(color: Colors.white.withValues(alpha: 0.8)),
                      ),
                      const SizedBox(width: AppSpacing.s8),
                      Icon(Icons.photo_library_rounded, size: 12, color: Colors.white.withValues(alpha: 0.8)),
                      const SizedBox(width: AppSpacing.s4),
                      Text(
                        '$photoCount',
                        style: context.text.labelMedium?.copyWith(color: Colors.white.withValues(alpha: 0.8)),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class RoomCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final int memberCount;
  final VoidCallback? onTap;

  const RoomCard({
    super.key,
    required this.title,
    required this.subtitle,
    required this.memberCount,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(AppSpacing.s20),
        decoration: BoxDecoration(
          color: c.surface,
          borderRadius: BorderRadius.circular(AppRadii.md),
          border: Border.all(color: c.borderSubtle),
        ),
        child: Row(
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: c.borderSubtle,
                borderRadius: BorderRadius.circular(AppRadii.sm),
              ),
              child: Icon(Icons.shield_outlined, color: c.text, size: 24),
            ),
            const SizedBox(width: AppSpacing.s16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title, style: context.text.titleMedium),
                  const SizedBox(height: 2),
                  Text(subtitle, style: context.text.bodySmall),
                ],
              ),
            ),
            Row(
              children: [
                Icon(Icons.people_outline, size: 16, color: c.textMuted),
                const SizedBox(width: 4),
                Text('$memberCount', style: context.text.labelMedium),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class UploadProgressCard extends StatelessWidget {
  final String filename;
  final double progress;

  const UploadProgressCard({
    super.key,
    required this.filename,
    required this.progress,
  });

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Container(
      padding: const EdgeInsets.all(AppSpacing.s16),
      decoration: BoxDecoration(
        color: c.surface,
        borderRadius: BorderRadius.circular(AppRadii.sm),
        border: Border.all(color: c.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Text(
                  filename,
                  style: context.text.labelMedium?.copyWith(color: c.text),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              Text('${(progress * 100).toInt()}%', style: context.text.labelMedium),
            ],
          ),
          const SizedBox(height: AppSpacing.s12),
          LinearProgressIndicator(
            value: progress,
            backgroundColor: c.borderSubtle,
            color: c.primary,
            minHeight: 4,
            borderRadius: BorderRadius.circular(AppRadii.pill),
          ),
        ],
      ),
    );
  }
}

class MediaCard extends StatelessWidget {
  const MediaCard({
    super.key,
    required this.imageUrl,
    required this.title,
    required this.subtitle,
    this.isVideo = false,
    this.duration,
    this.onTap,
    this.width = 160,
  });

  final String imageUrl;
  final String title;
  final String subtitle;
  final bool isVideo;
  final String? duration;
  final VoidCallback? onTap;
  final double width;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return GestureDetector(
      onTap: onTap,
      child: SizedBox(
        width: width,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Thumbnail
            ClipRRect(
              borderRadius: BorderRadius.circular(AppRadii.md),
              child: AspectRatio(
                aspectRatio: 4 / 3,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Image.network(
                      imageUrl,
                      fit: BoxFit.cover,
                      errorBuilder: (_, __, ___) => Container(
                        color: c.surfaceElevated,
                        child: Icon(Icons.image_outlined, color: c.textMuted, size: 28),
                      ),
                    ),
                    if (isVideo) ...[
                      // Play icon overlay
                      Center(
                        child: Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: Colors.black.withValues(alpha: 0.5),
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(Icons.play_arrow_rounded, color: Colors.white, size: 20),
                        ),
                      ),
                      // Duration badge
                      if (duration != null)
                        Positioned(
                          bottom: AppSpacing.s6, right: AppSpacing.s6,
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                            decoration: BoxDecoration(
                              color: Colors.black.withValues(alpha: 0.7),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Text(
                              duration!,
                              style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.w500),
                            ),
                          ),
                        ),
                    ],
                  ],
                ),
              ),
            ),
            const SizedBox(height: AppSpacing.s8),
            // Title
            Text(
              title,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: context.text.titleMedium?.copyWith(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 2),
            // Subtitle
            Text(
              subtitle,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: context.text.bodySmall?.copyWith(color: c.textMuted),
            ),
          ],
        ),
      ),
    );
  }
}
