import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/services/media/media_service.dart';
import 'package:memory_verse/core/services/media/media_service_provider.dart';
import 'package:memory_verse/core/services/media/windows_media_service.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;
import 'package:memory_verse/core/widgets/buttons.dart';

class MediaPermissionView extends ConsumerWidget {
  final VoidCallback? onManualUploadRequested;
  final VoidCallback? onPermissionGranted;

  const MediaPermissionView({
    super.key,
    this.onManualUploadRequested,
    this.onPermissionGranted,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final c = context.colors;
    final permStateAsync = ref.watch(mediaPermissionStateProvider);
    final mediaService = ref.watch(mediaServiceProvider);

    // If on Windows dev/testing environment, show the Windows Folder configuration card
    if (mediaService.isDesktopEnvironment) {
      final winService = mediaService as WindowsMediaService;
      final currentFolder = winService.activeCustomFolder;

      return Container(
        padding: const EdgeInsets.all(AppSpacing.s16),
        decoration: BoxDecoration(
          color: adt.AppColors.plum900.withValues(alpha: 0.6),
          borderRadius: BorderRadius.circular(AppRadii.lg),
          border: Border.all(
            color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.15),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(AppSpacing.s8),
                  decoration: BoxDecoration(
                    color: c.primary.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(AppRadii.md),
                  ),
                  child: Icon(Icons.laptop_windows_rounded, color: c.primary, size: 20),
                ),
                const SizedBox(width: AppSpacing.s12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Windows Dev Media Discovery',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          color: c.text,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        currentFolder != null
                            ? 'Folder: ...${currentFolder.length > 30 ? currentFolder.substring(currentFolder.length - 28) : currentFolder}'
                            : 'Scanning Pictures & Videos libraries',
                        style: TextStyle(fontSize: 12, color: c.textMuted),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ),
                TextButton.icon(
                  onPressed: () async {
                    await winService.selectCustomFolder();
                    // trigger rebuild
                    ref.invalidate(mediaServiceProvider);
                  },
                  icon: const Icon(Icons.folder_open_outlined, size: 16),
                  label: const Text('Change', style: TextStyle(fontSize: 12)),
                  style: TextButton.styleFrom(
                    foregroundColor: c.primary,
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  ),
                ),
              ],
            ),
          ],
        ),
      );
    }

    return permStateAsync.when(
      loading: () => const Center(
        child: Padding(
          padding: EdgeInsets.all(AppSpacing.s16),
          child: CircularProgressIndicator(),
        ),
      ),
      error: (err, _) => Container(
        padding: const EdgeInsets.all(AppSpacing.s16),
        decoration: BoxDecoration(
          color: c.error.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(AppRadii.lg),
          border: Border.all(color: c.error.withValues(alpha: 0.3)),
        ),
        child: Text('Error checking permissions: $err', style: TextStyle(color: c.error, fontSize: 13)),
      ),
      data: (status) {
        if (status == MediaPermissionState.granted) {
          return const SizedBox.shrink();
        }

        if (status == MediaPermissionState.limited) {
          return Container(
            padding: const EdgeInsets.all(AppSpacing.s16),
            decoration: BoxDecoration(
              color: Colors.amber.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(AppRadii.lg),
              border: Border.all(color: Colors.amber.withValues(alpha: 0.3)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Icon(Icons.info_outline, color: Colors.amber, size: 20),
                    const SizedBox(width: AppSpacing.s8),
                    Expanded(
                      child: Text(
                        'Limited Media Access',
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: c.text,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                Text(
                  'MemoryVerse currently has access to selected photos. To discover all memories, expand access in Settings.',
                  style: TextStyle(fontSize: 12, color: c.textMuted),
                ),
                const SizedBox(height: AppSpacing.s12),
                Row(
                  children: [
                    TextButton.icon(
                      onPressed: () => ref.read(mediaPermissionStateProvider.notifier).openSettings(),
                      icon: const Icon(Icons.settings, size: 16),
                      label: const Text('Open Settings'),
                    ),
                    if (onManualUploadRequested != null) ...[
                      const SizedBox(width: AppSpacing.s8),
                      TextButton.icon(
                        onPressed: onManualUploadRequested,
                        icon: const Icon(Icons.add_photo_alternate_outlined, size: 16),
                        label: const Text('Add Manually'),
                      ),
                    ],
                  ],
                ),
              ],
            ),
          );
        }

        if (status == MediaPermissionState.denied || status == MediaPermissionState.permanentlyDenied) {
          return Container(
            padding: const EdgeInsets.all(AppSpacing.s16),
            decoration: BoxDecoration(
              color: adt.AppColors.plum900.withValues(alpha: 0.8),
              borderRadius: BorderRadius.circular(AppRadii.lg),
              border: Border.all(color: c.error.withValues(alpha: 0.3)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.lock_outline_rounded, color: c.error, size: 20),
                    const SizedBox(width: AppSpacing.s8),
                    Expanded(
                      child: Text(
                        'Media Access Required',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          color: c.text,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: AppSpacing.s8),
                Text(
                  'MemoryVerse needs access to your photos and videos to automatically find relevant memories for your prompt.',
                  style: TextStyle(fontSize: 13, color: c.textMuted),
                ),
                const SizedBox(height: AppSpacing.s16),
                Row(
                  children: [
                    Expanded(
                      child: PrimaryButton(
                        label: status == MediaPermissionState.permanentlyDenied ? 'Open Settings' : 'Allow Access',
                        onPressed: () async {
                          if (status == MediaPermissionState.permanentlyDenied) {
                            await ref.read(mediaPermissionStateProvider.notifier).openSettings();
                          } else {
                            final res = await ref.read(mediaPermissionStateProvider.notifier).requestPermission();
                            if (res == MediaPermissionState.granted && onPermissionGranted != null) {
                              onPermissionGranted!();
                            }
                          }
                        },
                      ),
                    ),
                    if (onManualUploadRequested != null) ...[
                      const SizedBox(width: AppSpacing.s12),
                      Expanded(
                        child: OutlinedButton(
                          onPressed: onManualUploadRequested,
                          style: OutlinedButton.styleFrom(
                            foregroundColor: c.text,
                            side: BorderSide(color: c.border),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(AppRadii.pill),
                            ),
                          ),
                          child: const Text('Add Manually', style: TextStyle(fontSize: 13)),
                        ),
                      ),
                    ],
                  ],
                ),
              ],
            ),
          );
        }

        // notDetermined state
        return Container(
          padding: const EdgeInsets.all(AppSpacing.s16),
          decoration: BoxDecoration(
            color: adt.AppColors.plum900.withValues(alpha: 0.85),
            borderRadius: BorderRadius.circular(AppRadii.lg),
            border: Border.all(color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.15)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: c.primary.withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(AppRadii.md),
                    ),
                    child: Icon(Icons.auto_awesome, color: c.primary, size: 20),
                  ),
                  const SizedBox(width: AppSpacing.s12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Automatic Memory Discovery',
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: c.text,
                          ),
                        ),
                        Text(
                          'Allow access so AI can find your relevant photos.',
                          style: TextStyle(fontSize: 12, color: c.textMuted),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: AppSpacing.s12),
              PrimaryButton(
                label: 'Give Access to Media',
                onPressed: () async {
                  final res = await ref.read(mediaPermissionStateProvider.notifier).requestPermission();
                  if (res == MediaPermissionState.granted && onPermissionGranted != null) {
                    onPermissionGranted!();
                  }
                },
              ),
            ],
          ),
        );
      },
    );
  }
}
