import 'package:flutter/material.dart';
import 'package:memory_verse/core/design/tokens.dart';

class SectionHeader extends StatelessWidget {
  final String title;
  final String? actionLabel;
  final VoidCallback? onAction;

  const SectionHeader({
    super.key,
    required this.title,
    this.actionLabel,
    this.onAction,
  });

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: AppSpacing.s12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(title, style: context.text.headlineSmall),
          if (actionLabel != null && onAction != null)
            GestureDetector(
              onTap: onAction,
              child: Text(
                actionLabel!,
                style: context.text.labelLarge?.copyWith(color: c.textMuted),
              ),
            ),
        ],
      ),
    );
  }
}

class TimelineMarker extends StatelessWidget {
  final String label;

  const TimelineMarker({super.key, required this.label});

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: AppSpacing.s24),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(color: c.text, shape: BoxShape.circle),
          ),
          const SizedBox(width: AppSpacing.s16),
          Text(
            label,
            style: context.text.titleLarge?.copyWith(
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}

class MemoryHeader extends StatelessWidget {
  final String title;
  final String subtitle;
  final VoidCallback? onBack;
  final VoidCallback? onOptions;

  const MemoryHeader({
    super.key,
    required this.title,
    required this.subtitle,
    this.onBack,
    this.onOptions,
  });

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Container(
      color: c.bg,
      padding: EdgeInsets.only(
        top: MediaQuery.paddingOf(context).top + AppSpacing.s8,
        bottom: AppSpacing.s16,
        left: AppSpacing.s16,
        right: AppSpacing.s16,
      ),
      child: Row(
        children: [
          if (onBack != null)
            IconButton(
              icon: Icon(
                Icons.arrow_back_ios_new_rounded,
                color: c.text,
                size: 20,
              ),
              onPressed: onBack,
            ),
          const SizedBox(width: AppSpacing.s8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: context.text.headlineSmall,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                Text(
                  subtitle,
                  style: context.text.bodySmall,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
          if (onOptions != null)
            IconButton(
              icon: Icon(Icons.more_horiz_rounded, color: c.text),
              onPressed: onOptions,
            ),
        ],
      ),
    );
  }
}
