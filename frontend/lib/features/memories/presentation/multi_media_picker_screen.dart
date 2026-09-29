
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:file_picker/file_picker.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/upload_controller.dart';
import 'package:memory_verse/core/services/media/media_service.dart';
import 'package:memory_verse/core/services/media/media_service_provider.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;
import 'package:memory_verse/core/widgets/media.dart';

enum _MediaFilter { all, photos, videos }

class MultiMediaPickerScreen extends ConsumerStatefulWidget {
  final String memoryId;
  final String? vaultId;

  const MultiMediaPickerScreen({
    super.key,
    required this.memoryId,
    this.vaultId,
  });

  static void open(
    BuildContext context, {
    required String memoryId,
    String? vaultId,
  }) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) =>
            MultiMediaPickerScreen(memoryId: memoryId, vaultId: vaultId),
        fullscreenDialog: true,
      ),
    );
  }

  @override
  ConsumerState<MultiMediaPickerScreen> createState() =>
      _MultiMediaPickerScreenState();
}

class _MultiMediaPickerScreenState
    extends ConsumerState<MultiMediaPickerScreen> {
  bool _isLoading = true;
  List<DiscoveredMediaItem> _discoveredItems = [];
  final List<DiscoveredMediaItem> _selectedItems = [];
  _MediaFilter _activeFilter = _MediaFilter.all;
  int? _coverIndex;

  @override
  void initState() {
    super.initState();
    _loadDeviceMedia();
  }

  Future<void> _loadDeviceMedia() async {
    setState(() => _isLoading = true);
    try {
      final mediaService = ref.read(mediaServiceProvider);
      final perm = await mediaService.requestPermission();
      if (perm == MediaPermissionState.granted ||
          perm == MediaPermissionState.limited) {
        final items = await mediaService.discoverMedia(limit: 400);
        if (mounted) {
          setState(() {
            _discoveredItems = items;
            _isLoading = false;
          });
        }
      } else {
        if (mounted) {
          setState(() => _isLoading = false);
        }
      }
    } catch (e) {
      debugPrint('Error discovering device media: $e');
      if (mounted) setState(() => _isLoading = false);
    }
  }

  void _toggleSelection(DiscoveredMediaItem item) {
    setState(() {
      final existingIndex =
          _selectedItems.indexWhere((x) => x.mediaReference == item.mediaReference);
      if (existingIndex >= 0) {
        _selectedItems.removeAt(existingIndex);
        if (_coverIndex == existingIndex) {
          _coverIndex = _selectedItems.isNotEmpty ? 0 : null;
        } else if (_coverIndex != null && _coverIndex! > existingIndex) {
          _coverIndex = _coverIndex! - 1;
        }
      } else {
        _selectedItems.add(item);
        if (_coverIndex == null && _selectedItems.isNotEmpty) {
          _coverIndex = 0;
        }
      }
    });
  }

  Future<void> _pickExternalFiles() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['jpg', 'jpeg', 'png', 'webp', 'mp4', 'mov', 'heic'],
        allowMultiple: true,
      );
      if (result != null && result.files.isNotEmpty && mounted) {
        setState(() {
          for (final f in result.files) {
            if (f.path != null) {
              final file = File(f.path!);
              final ext = (f.extension ?? '').toLowerCase();
              final isVideo = ['mp4', 'mov', '3gp'].contains(ext);
              final item = DiscoveredMediaItem(
                id: 'external_${f.path.hashCode}',
                type: isVideo ? MediaItemType.video : MediaItemType.image,
                mediaReference: f.path!,
                timestamp: DateTime.now(),
                file: file,
                metadata: {'filename': f.name, 'file_size': f.size},
              );
              if (!_selectedItems.any((x) => x.mediaReference == item.mediaReference)) {
                _selectedItems.add(item);
              }
            }
          }
          if (_coverIndex == null && _selectedItems.isNotEmpty) {
            _coverIndex = 0;
          }
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error browsing files: $e')),
        );
      }
    }
  }

  void _confirmAndUpload() {
    if (_selectedItems.isEmpty) return;

    final files = <File>[];
    final mediaTypes = <String>[];

    for (final item in _selectedItems) {
      if (item.file != null) {
        files.add(item.file!);
        mediaTypes.add(item.isVideo ? 'video' : 'image');
      } else {
        files.add(File(item.mediaReference));
        mediaTypes.add(item.isVideo ? 'video' : 'image');
      }
    }

    ref.read(uploadControllerProvider.notifier).addUploads(
          files: files,
          mediaTypes: mediaTypes,
          memoryId: widget.memoryId,
          vaultId: widget.vaultId,
          coverIndex: _coverIndex,
        );

    Navigator.pop(context);
  }

  void _previewItem(DiscoveredMediaItem item) {
    if (item.file == null && !File(item.mediaReference).existsSync()) return;
    final file = item.file ?? File(item.mediaReference);

    showDialog(
      context: context,
      builder: (ctx) => Dialog(
        backgroundColor: Colors.transparent,
        insetPadding: EdgeInsets.zero,
        child: Stack(
          fit: StackFit.expand,
          children: [
            InteractiveViewer(
              child: AppFileImage(file: file, fit: BoxFit.contain),
            ),
            Positioned(
              top: AppSpacing.s32,
              right: AppSpacing.s16,
              child: IconButton(
                icon: const Icon(
                  Icons.close_rounded,
                  color: adt.AppColors.onDarkPrimary,
                  size: 32,
                ),
                onPressed: () => Navigator.pop(ctx),
              ),
            ),
          ],
        ),
      ),
    );
  }

  List<DiscoveredMediaItem> get _filteredItems {
    switch (_activeFilter) {
      case _MediaFilter.photos:
        return _discoveredItems.where((x) => !x.isVideo).toList();
      case _MediaFilter.videos:
        return _discoveredItems.where((x) => x.isVideo).toList();
      case _MediaFilter.all:
        return _discoveredItems;
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final selectedCount = _selectedItems.length;
    final displayItems = _filteredItems;

    return Scaffold(
      backgroundColor: c.bg,
      appBar: AppBar(
        backgroundColor: c.surface,
        elevation: 0,
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Add Media',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w700,
                color: c.text,
              ),
            ),
            Text(
              'Device Gallery',
              style: TextStyle(fontSize: 12, color: c.textMuted),
            ),
          ],
        ),
        leading: IconButton(
          icon: Icon(Icons.close_rounded, color: c.text),
          onPressed: () => Navigator.pop(context),
        ),
        actions: [
          IconButton(
            tooltip: 'Browse Files / Drive',
            icon: Icon(Icons.folder_open_rounded, color: c.primary),
            onPressed: _pickExternalFiles,
          ),
          if (selectedCount > 0)
            Padding(
              padding: const EdgeInsets.only(right: AppSpacing.s12),
              child: TextButton(
                onPressed: _confirmAndUpload,
                child: Text(
                  'Add ($selectedCount)',
                  style: TextStyle(
                    color: c.primary,
                    fontWeight: FontWeight.w700,
                    fontSize: 15,
                  ),
                ),
              ),
            ),
        ],
      ),
      body: Column(
        children: [
          // Filter Tabs
          Container(
            padding: const EdgeInsets.symmetric(
              horizontal: AppSpacing.s16,
              vertical: AppSpacing.s8,
            ),
            color: c.surface,
            child: Row(
              children: [
                _buildFilterChip('All', _MediaFilter.all),
                const SizedBox(width: AppSpacing.s8),
                _buildFilterChip('Photos', _MediaFilter.photos),
                const SizedBox(width: AppSpacing.s8),
                _buildFilterChip('Videos', _MediaFilter.videos),
                const Spacer(),
                Text(
                  '${displayItems.length} found',
                  style: TextStyle(fontSize: 12, color: c.textMuted),
                ),
              ],
            ),
          ),

          // Main Gallery Grid
          Expanded(
            child: _isLoading
                ? const Center(child: CircularProgressIndicator())
                : displayItems.isEmpty
                    ? Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.photo_library_outlined,
                              size: 64,
                              color: c.textMuted.withValues(alpha: 0.4),
                            ),
                            const SizedBox(height: AppSpacing.s16),
                            Text(
                              'No media found on device',
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.w600,
                                color: c.text,
                              ),
                            ),
                            const SizedBox(height: AppSpacing.s8),
                            TextButton.icon(
                              icon: const Icon(Icons.folder_open),
                              label: const Text('Browse Files or Storage'),
                              onPressed: _pickExternalFiles,
                            ),
                          ],
                        ),
                      )
                    : GridView.builder(
                        padding: const EdgeInsets.all(4),
                        gridDelegate:
                            const SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: 3,
                          crossAxisSpacing: 3,
                          mainAxisSpacing: 3,
                          childAspectRatio: 1.0,
                        ),
                        itemCount: displayItems.length,
                        itemBuilder: (context, index) {
                          final item = displayItems[index];
                          final file =
                              item.file ?? File(item.mediaReference);
                          final selIdx = _selectedItems.indexWhere(
                            (x) => x.mediaReference == item.mediaReference,
                          );
                          final isSelected = selIdx >= 0;

                          return GestureDetector(
                            key: ValueKey(item.mediaReference),
                            onTap: () => _toggleSelection(item),
                            onLongPress: () => _previewItem(item),
                            child: Stack(
                              fit: StackFit.expand,
                              children: [
                                // Thumbnail Image
                                Container(
                                  color: c.surfaceElevated,
                                  child: AppFileImage(
                                    file: file,
                                    fit: BoxFit.cover,
                                    memCacheWidth: 260,
                                  ),
                                ),

                                // Video Indicator
                                if (item.isVideo)
                                  Positioned(
                                    bottom: 6,
                                    left: 6,
                                    child: Container(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 5,
                                        vertical: 2,
                                      ),
                                      decoration: BoxDecoration(
                                        color: adt.AppColors.plum900
                                            .withValues(alpha: 0.75),
                                        borderRadius:
                                            BorderRadius.circular(4),
                                      ),
                                      child: const Row(
                                        mainAxisSize: MainAxisSize.min,
                                        children: [
                                          Icon(
                                            Icons.videocam_rounded,
                                            size: 14,
                                            color: adt.AppColors.onDarkPrimary,
                                          ),
                                        ],
                                      ),
                                    ),
                                  ),

                                // Dark overlay if selected
                                if (isSelected)
                                  Container(
                                    color: c.primary.withValues(alpha: 0.28),
                                  ),

                                // Numbered Selection Badge
                                Positioned(
                                  top: 6,
                                  right: 6,
                                  child: Container(
                                    width: 26,
                                    height: 26,
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      color: isSelected
                                          ? c.primary
                                          : adt.AppColors.plum900
                                              .withValues(alpha: 0.45),
                                      border: Border.all(
                                        color: isSelected
                                            ? c.primary
                                            : adt.AppColors.onDarkPrimary
                                                .withValues(alpha: 0.7),
                                        width: 1.8,
                                      ),
                                    ),
                                    child: Center(
                                      child: isSelected
                                          ? Text(
                                              '${selIdx + 1}',
                                              style: TextStyle(
                                                color: c.primaryInverse,
                                                fontSize: 12,
                                                fontWeight: FontWeight.w700,
                                              ),
                                            )
                                          : null,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          );
                        },
                      ),
          ),

          // Bottom Bar
          if (selectedCount > 0)
            Container(
              padding: const EdgeInsets.symmetric(
                horizontal: AppSpacing.s20,
                vertical: AppSpacing.s12,
              ),
              decoration: BoxDecoration(
                color: c.surfaceElevated,
                border: Border(top: BorderSide(color: c.border, width: 0.5)),
              ),
              child: SafeArea(
                top: false,
                child: Row(
                  children: [
                    Text(
                      'Selected: $selectedCount',
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                        color: c.text,
                      ),
                    ),
                    const Spacer(),
                    FilledButton.icon(
                      icon: const Icon(Icons.check_rounded, size: 18),
                      label: Text('Add to Memory ($selectedCount)'),
                      onPressed: _confirmAndUpload,
                      style: FilledButton.styleFrom(
                        backgroundColor: c.primary,
                        foregroundColor: c.primaryInverse,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(AppRadii.pill),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildFilterChip(String label, _MediaFilter filter) {
    final c = context.colors;
    final isSelected = _activeFilter == filter;

    return ChoiceChip(
      label: Text(label),
      selected: isSelected,
      onSelected: (_) => setState(() => _activeFilter = filter),
      selectedColor: c.primary.withValues(alpha: 0.2),
      backgroundColor: c.surfaceElevated,
      labelStyle: TextStyle(
        fontSize: 13,
        fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
        color: isSelected ? c.primary : c.textMuted,
      ),
      side: BorderSide(
        color: isSelected ? c.primary : c.border,
        width: 1,
      ),
    );
  }
}
