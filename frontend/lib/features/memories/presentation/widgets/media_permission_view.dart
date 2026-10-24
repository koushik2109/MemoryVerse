import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/services/media/media_service.dart';
import 'package:memory_verse/core/services/media/media_service_provider.dart';
import 'package:memory_verse/core/services/media/windows_media_service.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;
import 'package:memory_verse/core/widgets/buttons.dart';

class MediaPermissionView extends ConsumerStatefulWidget {
  final VoidCallback? onManualUploadRequested;
  final VoidCallback? onPermissionGranted;

  const MediaPermissionView({
    super.key,
    this.onManualUploadRequested,
    this.onPermissionGranted,
  });

  @override
  ConsumerState<MediaPermissionView> createState() =>
      _MediaPermissionViewState();
}

class _MediaPermissionViewState extends ConsumerState<MediaPermissionView>
    with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      ref.read(mediaPermissionStateProvider.notifier).checkPermission();
    }
  }

  @override
  Widget build(BuildContext context) {
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
                  child: Icon(Icons.laptop_windows_rounded,
                      color: c.primary, size: 20),
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
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
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
        child: Text('Error checking permissions: $err',
            style: TextStyle(color: c.error, fontSize: 13)),
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
                    const Icon(Icons.info_outline,
                        color: Colors.amber, size: 20),
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
                  'MemoryVerse currently has access to selected photos and videos. Use the selected media now, manage the selection, or expand access in Settings.',
                  style: TextStyle(fontSize: 12, color: c.textMuted),
                ),
                const SizedBox(height: AppSpacing.s12),
                Wrap(
                  spacing: AppSpacing.s8,
                  runSpacing: AppSpacing.s4,
                  children: [
                    TextButton.icon(
                      onPressed: () => ref
                          .read(mediaPermissionStateProvider.notifier)
                          .openLimitedAccessPicker(),
                      icon: const Icon(Icons.photo_library_outlined, size: 16),
                      label: const Text('Manage Selection'),
                    ),
                    TextButton.icon(
                      onPressed: () => ref
                          .read(mediaPermissionStateProvider.notifier)
                          .openSettings(),
                      icon: const Icon(Icons.settings, size: 16),
                      label: const Text('Open Settings'),
                    ),
                    if (widget.onPermissionGranted != null)
                      TextButton.icon(
                        onPressed: widget.onPermissionGranted,
                        icon: const Icon(Icons.check, size: 16),
                        label: const Text('Use Selected Media'),
                      ),
                    if (widget.onManualUploadRequested != null)
                      TextButton.icon(
                        onPressed: widget.onManualUploadRequested,
                        icon: const Icon(Icons.add_photo_alternate_outlined,
                            size: 16),
                        label: const Text('Add Manually'),
                      ),
                  ],
                ),
              ],
            ),
          );
        }

        if (status == MediaPermissionState.denied ||
            status == MediaPermissionState.permanentlyDenied ||
            status == MediaPermissionState.restricted) {
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
                        label:
                            status == MediaPermissionState.permanentlyDenied ||
                                    status == MediaPermissionState.restricted
                                ? 'Open Settings'
                                : 'Allow Access',
                        onPressed: () async {
                          if (status ==
                                  MediaPermissionState.permanentlyDenied ||
                              status == MediaPermissionState.restricted) {
                            await ref
                                .read(mediaPermissionStateProvider.notifier)
                                .openSettings();
                          } else {
                            final res = await ref
                                .read(mediaPermissionStateProvider.notifier)
                                .requestPermission();
                            if (res.hasMediaAccess &&
                                widget.onPermissionGranted != null) {
                              widget.onPermissionGranted!();
                            }
                          }
                        },
                      ),
                    ),
                    if (widget.onManualUploadRequested != null) ...[
                      const SizedBox(width: AppSpacing.s12),
                      Expanded(
                        child: OutlinedButton(
                          onPressed: widget.onManualUploadRequested,
                          style: OutlinedButton.styleFrom(
                            foregroundColor: c.text,
                            side: BorderSide(color: c.border),
                            shape: RoundedRectangleBorder(
                              borderRadius:
                                  BorderRadius.circular(AppRadii.pill),
                            ),
                          ),
                          child: const Text('Add Manually',
                              style: TextStyle(fontSize: 13)),
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
            border: Border.all(
                color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.15)),
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
                          'Private On-Device Memory Discovery',
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: c.text,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          "MemoryVerse can look through your photos and videos to find moments that match what you're creating — so you don't have to hunt for them yourself. This scan happens on your device; nothing is sent anywhere unless you choose to add it to a memory. You can allow full access, select specific albums, or skip this and upload manually.",
                          style: TextStyle(
                              fontSize: 12, color: c.textMuted, height: 1.35),
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
                  final res = await ref
                      .read(mediaPermissionStateProvider.notifier)
                      .requestPermission();
                  if (res.hasMediaAccess &&
                      widget.onPermissionGranted != null) {
                    widget.onPermissionGranted!();
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
