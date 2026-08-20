import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/upload_controller.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;

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
  final ImagePicker _picker = ImagePicker();
  final List<File> _selectedFiles = [];
  final List<String> _mediaTypes = []; // 'image' or 'video'
  int? _coverIndex;

  Future<void> _pickImages() async {
    final images = await _picker.pickMultiImage(imageQuality: 80);
    if (images.isNotEmpty) {
      setState(() {
        _selectedFiles.addAll(images.map((x) => File(x.path)));
        _mediaTypes.addAll(List.generate(images.length, (_) => 'image'));
        if (_coverIndex == null && _selectedFiles.isNotEmpty) _coverIndex = 0;
      });
    }
  }

  Future<void> _pickVideo() async {
    final video = await _picker.pickVideo(source: ImageSource.gallery);
    if (video != null) {
      setState(() {
        _selectedFiles.add(File(video.path));
        _mediaTypes.add('video');
        if (_coverIndex == null && _selectedFiles.isNotEmpty) _coverIndex = 0;
      });
    }
  }

  Future<void> _pickFiles() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi', 'heic'],
      allowMultiple: true,
    );
    if (result != null && result.files.isNotEmpty) {
      setState(() {
        for (var file in result.files) {
          if (file.path != null) {
            final ext = (file.extension ?? '').toLowerCase();
            final isVideo = ['mp4', 'mov', 'avi'].contains(ext);
            _selectedFiles.add(File(file.path!));
            _mediaTypes.add(isVideo ? 'video' : 'image');
          }
        }
        if (_coverIndex == null && _selectedFiles.isNotEmpty) _coverIndex = 0;
      });
    }
  }

  void _startUpload() {
    if (_selectedFiles.isEmpty) return;

    ref
        .read(uploadControllerProvider.notifier)
        .addUploads(
          files: _selectedFiles,
          mediaTypes: _mediaTypes,
          memoryId: widget.memoryId,
          vaultId: widget.vaultId,
          coverIndex: _coverIndex,
        );

    Navigator.pop(context); // Close the picker immediately
  }

  void _previewItem(int index) {
    if (_mediaTypes[index] == 'video') {
      // For local video preview we can't use VideoPlayerScreen easily if it expects network url.
      // But we will just show a snackbar or implement a local player if needed.
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Video preview coming soon')),
      );
    } else {
      showDialog(
        context: context,
        builder: (ctx) => Dialog(
          backgroundColor: Colors.transparent,
          insetPadding: EdgeInsets.zero,
          child: Stack(
            fit: StackFit.expand,
            children: [
              InteractiveViewer(
                child: Image.file(_selectedFiles[index], fit: BoxFit.contain),
              ),
              Positioned(
                top: AppSpacing.s32,
                right: AppSpacing.s16,
                child: IconButton(
                  icon: const Icon(Icons.close, color: adt.AppColors.onDarkPrimary, size: 32),
                  onPressed: () => Navigator.pop(ctx),
                ),
              ),
            ],
          ),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;

    return Scaffold(
      backgroundColor: c.bg,
      appBar: AppBar(
        backgroundColor: c.surface,
        elevation: 0,
        title: Text(
          'Select Media',
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w600,
            color: c.text,
          ),
        ),
        leading: IconButton(
          icon: Icon(Icons.close_rounded, color: c.text),
          onPressed: () => Navigator.pop(context),
        ),
        actions: [
          if (_selectedFiles.isNotEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: AppSpacing.s16,
                vertical: AppSpacing.s8,
              ),
              child: ElevatedButton(
                onPressed: _startUpload,
                style: ElevatedButton.styleFrom(
                  backgroundColor: c.primary,
                  foregroundColor: c.primaryInverse,
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(AppRadii.pill),
                  ),
                ),
                child: Text(
                  'Upload (${_selectedFiles.length})',
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
              ),
            ),
        ],
      ),
      body: Column(
        children: [
          Container(
            padding: const EdgeInsets.symmetric(
              horizontal: AppSpacing.s16,
              vertical: AppSpacing.s12,
            ),
            decoration: BoxDecoration(
              color: c.surface,
              border: Border(bottom: BorderSide(color: c.border, width: 0.5)),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _ActionBtn(
                  icon: Icons.photo_library_outlined,
                  label: 'Photos',
                  onTap: _pickImages,
                  colors: c,
                ),
                _ActionBtn(
                  icon: Icons.videocam_outlined,
                  label: 'Video',
                  onTap: _pickVideo,
                  colors: c,
                ),
                _ActionBtn(
                  icon: Icons.folder_open_outlined,
                  label: 'Files',
                  onTap: _pickFiles,
                  colors: c,
                ),
              ],
            ),
          ),

          Expanded(
            child: _selectedFiles.isEmpty
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.perm_media_outlined,
                          size: 64,
                          color: c.textMuted.withValues(alpha: 0.3),
                        ),
                        const SizedBox(height: AppSpacing.s16),
                        Text(
                          'No media selected',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                            color: c.text,
                          ),
                        ),
                        const SizedBox(height: AppSpacing.s8),
                        Text(
                          'Tap above to select photos or videos',
                          style: TextStyle(color: c.textMuted),
                        ),
                      ],
                    ),
                  )
                : ReorderableListView.builder(
                    padding: const EdgeInsets.all(AppSpacing.s16),
                    itemCount: _selectedFiles.length,
                    onReorderItem: (oldIndex, newIndex) {
                      setState(() {
                        final file = _selectedFiles.removeAt(oldIndex);
                        final type = _mediaTypes.removeAt(oldIndex);
                        _selectedFiles.insert(newIndex, file);
                        _mediaTypes.insert(newIndex, type);

                        // Update cover index if affected
                        if (_coverIndex == oldIndex) {
                          _coverIndex = newIndex;
                        } else if (_coverIndex != null) {
                          if (oldIndex < _coverIndex! &&
                              newIndex >= _coverIndex!) {
                            _coverIndex = _coverIndex! - 1;
                          } else if (oldIndex > _coverIndex! &&
                              newIndex <= _coverIndex!) {
                            _coverIndex = _coverIndex! + 1;
                          }
                        }
                      });
                    },
                    itemBuilder: (ctx, i) {
                      final file = _selectedFiles[i];
                      final isVideo = _mediaTypes[i] == 'video';
                      final isCover = _coverIndex == i;

                      return Container(
                        key: ValueKey(file.path),
                        margin: const EdgeInsets.only(bottom: AppSpacing.s12),
                        decoration: BoxDecoration(
                          color: c.surfaceElevated,
                          borderRadius: BorderRadius.circular(AppRadii.lg),
                          border: Border.all(
                            color: isCover ? c.primary : c.border,
                            width: isCover ? 2 : 1,
                          ),
                        ),
                        child: ListTile(
                          contentPadding: const EdgeInsets.all(AppSpacing.s8),
                          leading: GestureDetector(
                            onTap: () => _previewItem(i),
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(AppRadii.md),
                              child: SizedBox(
                                width: 64,
                                height: 64,
                                child: Stack(
                                  fit: StackFit.expand,
                                  children: [
                                    if (isVideo)
                                      Container(
                                        color: adt.AppColors.plum900.withValues(alpha: 0.87),
                                        child: const Icon(
                                          Icons.movie_outlined,
                                          color: adt.AppColors.onDarkPrimary,
                                        ),
                                      )
                                    else
                                      Image.file(file, fit: BoxFit.cover),

                                    if (isVideo)
                                      Container(
                                        color: adt.AppColors.plum900.withValues(
                                          alpha: 0.3,
                                        ),
                                        child: const Center(
                                          child: Icon(
                                            Icons.play_arrow_rounded,
                                            color: adt.AppColors.onDarkPrimary,
                                          ),
                                        ),
                                      ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                          title: Text(
                            file.path.split('/').last,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w500,
                              color: c.text,
                            ),
                          ),
                          subtitle: Padding(
                            padding: const EdgeInsets.only(top: 4),
                            child: Row(
                              children: [
                                if (isCover)
                                  Container(
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: 6,
                                      vertical: 2,
                                    ),
                                    decoration: BoxDecoration(
                                      color: c.primary.withValues(alpha: 0.1),
                                      borderRadius: BorderRadius.circular(4),
                                    ),
                                    child: Text(
                                      'COVER',
                                      style: TextStyle(
                                        fontSize: 10,
                                        fontWeight: FontWeight.w700,
                                        color: c.primary,
                                      ),
                                    ),
                                  )
                                else
                                  GestureDetector(
                                    onTap: () =>
                                        setState(() => _coverIndex = i),
                                    child: Text(
                                      'Set as cover',
                                      style: TextStyle(
                                        fontSize: 12,
                                        color: c.primary,
                                      ),
                                    ),
                                  ),
                              ],
                            ),
                          ),
                          trailing: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              IconButton(
                                icon: Icon(
                                  Icons.close_rounded,
                                  color: c.textMuted,
                                ),
                                onPressed: () {
                                  setState(() {
                                    _selectedFiles.removeAt(i);
                                    _mediaTypes.removeAt(i);
                                    if (_coverIndex == i) {
                                      _coverIndex = _selectedFiles.isEmpty
                                          ? null
                                          : 0;
                                    } else if (_coverIndex != null &&
                                        _coverIndex! > i) {
                                      _coverIndex = _coverIndex! - 1;
                                    }
                                  });
                                },
                              ),
                              Icon(
                                Icons.drag_handle_rounded,
                                color: c.textMuted.withValues(alpha: 0.5),
                              ),
                            ],
                          ),
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

class _ActionBtn extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final AppColors colors;

  const _ActionBtn({
    required this.icon,
    required this.label,
    required this.onTap,
    required this.colors,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: AppSpacing.s20,
          vertical: AppSpacing.s12,
        ),
        decoration: BoxDecoration(
          color: colors.bg,
          borderRadius: BorderRadius.circular(AppRadii.lg),
        ),
        child: Column(
          children: [
            Icon(icon, size: 24, color: colors.text),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w500,
                color: colors.text,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
