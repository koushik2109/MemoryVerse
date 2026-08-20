import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/upload_controller.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;

class UploadProgressBanner extends ConsumerWidget {
  const UploadProgressBanner({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final tasks = ref.watch(uploadControllerProvider);

    if (tasks.isEmpty) return const SizedBox.shrink();

    final total = tasks.length;
    final completed = tasks
        .where((t) => t.status == UploadStatus.completed)
        .length;
    final uploading = tasks
        .where((t) => t.status == UploadStatus.uploading)
        .toList();
    final errors = tasks.where((t) => t.status == UploadStatus.error).length;

    // Auto hide if all completed or canceled
    if (completed == total ||
        tasks.every(
          (t) =>
              t.status == UploadStatus.completed ||
              t.status == UploadStatus.canceled,
        )) {
      // We could add a delay, but for now we'll just show it or provide a "dismiss" button in the queue
      // Actually, let's keep it visible if there are errors, otherwise auto-hide
      if (errors == 0 &&
          tasks.every((t) => t.status == UploadStatus.completed)) {
        Future.microtask(
          () => ref.read(uploadControllerProvider.notifier).clearAll(),
        );
        return const SizedBox.shrink();
      }
    }

    final c = context.colors;

    double overallProgress = 0.0;
    if (total > 0) {
      double totalProgress = 0.0;
      for (var task in tasks) {
        if (task.status == UploadStatus.completed) {
          totalProgress += 1.0;
        } else if (task.status == UploadStatus.uploading) {
          totalProgress += task.progress;
        }
      }
      overallProgress = totalProgress / total;
    }

    String statusText = 'Uploading $completed of $total...';
    if (errors > 0) statusText = '$errors failed, $completed uploaded';
    if (completed == total) statusText = 'Upload complete';

    return GestureDetector(
      onTap: () => _showQueueSheet(context, ref),
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: AppSpacing.s16,
          vertical: AppSpacing.s12,
        ),
        decoration: BoxDecoration(
          color: c.surfaceElevated,
          border: Border(bottom: BorderSide(color: c.border, width: 0.5)),
          boxShadow: [
            BoxShadow(
              color: adt.AppColors.plum900.withValues(alpha: 0.1),
              blurRadius: 4,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Row(
          children: [
            SizedBox(
              width: 24,
              height: 24,
              child: uploading.isNotEmpty || completed < total && errors == 0
                  ? CircularProgressIndicator(
                      value: overallProgress,
                      strokeWidth: 3,
                      color: c.primary,
                      backgroundColor: c.border,
                    )
                  : Icon(
                      errors > 0
                          ? Icons.error_outline
                          : Icons.check_circle_outline,
                      color: errors > 0 ? c.error : c.success,
                      size: 24,
                    ),
            ),
            const SizedBox(width: AppSpacing.s12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    statusText,
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: c.text,
                    ),
                  ),
                  if (uploading.isNotEmpty)
                    Text(
                      'Uploading: ${uploading.first.filename}',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(fontSize: 11, color: c.textMuted),
                    ),
                ],
              ),
            ),
            Icon(Icons.keyboard_arrow_up_rounded, color: c.textMuted),
          ],
        ),
      ),
    );
  }

  void _showQueueSheet(BuildContext context, WidgetRef ref) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => const _UploadQueueSheet(),
    );
  }
}

class _UploadQueueSheet extends ConsumerWidget {
  const _UploadQueueSheet();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final tasks = ref.watch(uploadControllerProvider);
    final c = context.colors;

    return Container(
      height: MediaQuery.of(context).size.height * 0.7,
      decoration: BoxDecoration(
        color: c.bg,
        borderRadius: const BorderRadius.vertical(
          top: Radius.circular(AppRadii.xl),
        ),
      ),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(AppSpacing.s16),
            decoration: BoxDecoration(
              color: c.surface,
              borderRadius: const BorderRadius.vertical(
                top: Radius.circular(AppRadii.xl),
              ),
              border: Border(bottom: BorderSide(color: c.border, width: 0.5)),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Upload Queue',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                    color: c.text,
                  ),
                ),
                Row(
                  children: [
                    TextButton(
                      onPressed: () => ref
                          .read(uploadControllerProvider.notifier)
                          .clearAll(),
                      child: Text(
                        'Clear',
                        style: TextStyle(color: c.textMuted),
                      ),
                    ),
                    IconButton(
                      icon: Icon(Icons.close_rounded, color: c.text),
                      onPressed: () => Navigator.pop(context),
                    ),
                  ],
                ),
              ],
            ),
          ),
          Expanded(
            child: ListView.separated(
              padding: const EdgeInsets.all(AppSpacing.s16),
              itemCount: tasks.length,
              separatorBuilder: (_, __) =>
                  const SizedBox(height: AppSpacing.s12),
              itemBuilder: (ctx, i) {
                final task = tasks[i];
                return Container(
                  padding: const EdgeInsets.all(AppSpacing.s12),
                  decoration: BoxDecoration(
                    color: c.surfaceElevated,
                    borderRadius: BorderRadius.circular(AppRadii.md),
                    border: Border.all(color: c.border),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 40,
                        height: 40,
                        decoration: BoxDecoration(
                          color: c.bg,
                          borderRadius: BorderRadius.circular(AppRadii.sm),
                        ),
                        child: task.mediaType == 'video'
                            ? Icon(Icons.movie_outlined, color: c.textMuted)
                            : ClipRRect(
                                borderRadius: BorderRadius.circular(
                                  AppRadii.sm,
                                ),
                                child: Image.file(task.file, fit: BoxFit.cover),
                              ),
                      ),
                      const SizedBox(width: AppSpacing.s12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              task.filename,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w500,
                                color: c.text,
                              ),
                            ),
                            const SizedBox(height: 4),
                            if (task.status == UploadStatus.uploading)
                              LinearProgressIndicator(
                                value: task.progress,
                                backgroundColor: c.border,
                                color: c.primary,
                                minHeight: 4,
                              )
                            else if (task.status == UploadStatus.error)
                              Text(
                                task.error ?? 'Upload failed',
                                style: TextStyle(fontSize: 11, color: c.error),
                              )
                            else if (task.status == UploadStatus.completed)
                              Text(
                                'Completed',
                                style: TextStyle(
                                  fontSize: 11,
                                  color: c.success,
                                ),
                              )
                            else if (task.status == UploadStatus.canceled)
                              Text(
                                'Canceled',
                                style: TextStyle(
                                  fontSize: 11,
                                  color: c.textMuted,
                                ),
                              )
                            else
                              Text(
                                'Waiting...',
                                style: TextStyle(
                                  fontSize: 11,
                                  color: c.textMuted,
                                ),
                              ),
                          ],
                        ),
                      ),
                      const SizedBox(width: AppSpacing.s12),
                      if (task.status == UploadStatus.error ||
                          task.status == UploadStatus.canceled)
                        IconButton(
                          icon: Icon(Icons.refresh_rounded, color: c.primary),
                          onPressed: () => ref
                              .read(uploadControllerProvider.notifier)
                              .retryUpload(task.id),
                        )
                      else if (task.status == UploadStatus.uploading ||
                          task.status == UploadStatus.waiting)
                        IconButton(
                          icon: Icon(Icons.close_rounded, color: c.textMuted),
                          onPressed: () => ref
                              .read(uploadControllerProvider.notifier)
                              .cancelUpload(task.id),
                        ),
                    ],
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
