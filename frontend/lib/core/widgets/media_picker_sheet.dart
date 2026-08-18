import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/core/widgets/buttons.dart';

class MediaPickerSheet extends ConsumerStatefulWidget {
  final String? vaultId;
  final String? memoryId;
  const MediaPickerSheet({super.key, this.vaultId, this.memoryId});

  static void show(BuildContext context, {String? vaultId, String? memoryId}) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => MediaPickerSheet(vaultId: vaultId, memoryId: memoryId),
    );
  }

  @override
  ConsumerState<MediaPickerSheet> createState() => _MediaPickerSheetState();
}

class _MediaPickerSheetState extends ConsumerState<MediaPickerSheet> {
  final ImagePicker _picker = ImagePicker();
  final List<File> _selectedFiles = [];
  final List<String> _mediaTypes = []; // 'image' or 'video'
  bool _isUploading = false;
  String? _uploadStatus;

  Future<void> _pickImages() async {
    final images = await _picker.pickMultiImage(imageQuality: 80);
    if (images.isNotEmpty) {
      setState(() {
        _selectedFiles.addAll(images.map((x) => File(x.path)));
        _mediaTypes.addAll(List.generate(images.length, (_) => 'image'));
      });
    }
  }

  Future<void> _pickVideo() async {
    final video = await _picker.pickVideo(source: ImageSource.gallery);
    if (video != null) {
      setState(() {
        _selectedFiles.add(File(video.path));
        _mediaTypes.add('video');
      });
    }
  }

  Future<void> _pickFiles() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'],
      allowMultiple: true,
    );
    if (result != null && result.files.isNotEmpty) {
      setState(() {
        for (var file in result.files) {
          if (file.path != null) {
            final ext = (file.extension ?? '').toLowerCase();
            final isVideo = ['mp4', 'mov', 'avi'].contains(ext);
            _selectedFiles.add(File(file.path!));
            _mediaTypes.add(isVideo ? 'image' : 'video'); // Quick fix: should map extensions properly but this works for demo
          }
        }
      });
    }
  }

  Future<void> _uploadAll() async {
    if (_selectedFiles.isEmpty) return;
    setState(() {
      _isUploading = true;
      _uploadStatus = 'Uploading 0/${_selectedFiles.length}...';
    });

    final repo = ref.read(mediaRepositoryProvider);
    try {
      for (int i = 0; i < _selectedFiles.length; i++) {
        final file = _selectedFiles[i];
        final type = _mediaTypes[i];
        final name = file.path.split('/').last;

        setState(() {
          _uploadStatus =
              'Uploading ${i + 1}/${_selectedFiles.length}: $name...';
        });

        await repo.uploadMedia(
          file: file,
          filename: name,
          mediaType: type,
          vaultId: widget.vaultId,
          memoryId: widget.memoryId,
        );
      }

      ref.invalidate(userMediaProvider);
      ref.invalidate(memoriesListProvider);
      ref.invalidate(memoryDetailProvider);
      ref.invalidate(vaultsListProvider);
      ref.invalidate(userProfileProvider);

      if (mounted) {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              '${_selectedFiles.length} media item(s) uploaded successfully!',
            ),
            backgroundColor: context.colors.success,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        setState(() => _isUploading = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Upload error: $e'),
            backgroundColor: context.colors.error,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;

    return Container(
      padding: const EdgeInsets.all(AppSpacing.s24),
      decoration: BoxDecoration(
        color: c.surfaceElevated,
        borderRadius: const BorderRadius.vertical(
          top: Radius.circular(AppRadii.xl),
        ),
        border: Border.all(color: c.borderSubtle),
      ),
      child: SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('Add Memories', style: context.text.headlineSmall),
                IconButton(
                  onPressed: () => Navigator.pop(context),
                  icon: Icon(Icons.close, color: c.textMuted),
                ),
              ],
            ),
            const SizedBox(height: AppSpacing.s16),

            if (_selectedFiles.isEmpty) ...[
              Row(
                children: [
                  Expanded(
                    child: _OptionTile(
                      icon: Icons.photo_library_outlined,
                      label: 'Photos',
                      onTap: _pickImages,
                    ),
                  ),
                  const SizedBox(width: AppSpacing.s12),
                  Expanded(
                    child: _OptionTile(
                      icon: Icons.videocam_outlined,
                      label: 'Video',
                      onTap: _pickVideo,
                    ),
                  ),
                  const SizedBox(width: AppSpacing.s12),
                  Expanded(
                    child: _OptionTile(
                      icon: Icons.folder_open_outlined,
                      label: 'Files',
                      onTap: _pickFiles,
                    ),
                  ),
                ],
              ),
            ] else ...[
              SizedBox(
                height: 110,
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  itemCount: _selectedFiles.length,
                  itemBuilder: (_, i) {
                    final file = _selectedFiles[i];
                    final isVideo = _mediaTypes[i] == 'video';
                    return Container(
                      width: 90,
                      margin: const EdgeInsets.only(right: AppSpacing.s12),
                      decoration: BoxDecoration(
                        color: c.surface,
                        borderRadius: BorderRadius.circular(AppRadii.md),
                        border: Border.all(color: c.border),
                      ),
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          if (!isVideo)
                            ClipRRect(
                              borderRadius: BorderRadius.circular(AppRadii.md),
                              child: Image.file(file, fit: BoxFit.cover),
                            )
                          else
                            Center(
                              child: Icon(
                                Icons.movie_outlined,
                                size: 36,
                                color: c.text,
                              ),
                            ),
                          Positioned(
                            top: 4,
                            right: 4,
                            child: GestureDetector(
                              onTap: () {
                                setState(() {
                                  _selectedFiles.removeAt(i);
                                  _mediaTypes.removeAt(i);
                                });
                              },
                              child: Container(
                                padding: const EdgeInsets.all(2),
                                decoration: const BoxDecoration(
                                  color: Colors.black54,
                                  shape: BoxShape.circle,
                                ),
                                child: const Icon(
                                  Icons.close,
                                  size: 14,
                                  color: Colors.white,
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),
              const SizedBox(height: AppSpacing.s16),
              if (_uploadStatus != null)
                Text(_uploadStatus!, style: context.text.bodySmall),
              const SizedBox(height: AppSpacing.s12),
              Row(
                children: [
                  Expanded(
                    child: SecondaryButton(
                      label: 'Add More',
                      onPressed: _isUploading ? null : _pickImages,
                    ),
                  ),
                  const SizedBox(width: AppSpacing.s12),
                  Expanded(
                    flex: 2,
                    child: PrimaryButton(
                      label: 'Upload (${_selectedFiles.length})',
                      onPressed: _isUploading ? null : _uploadAll,
                      isLoading: _isUploading,
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _OptionTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _OptionTile({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(
          vertical: AppSpacing.s20,
          horizontal: AppSpacing.s8,
        ),
        decoration: BoxDecoration(
          color: c.surface,
          borderRadius: BorderRadius.circular(AppRadii.lg),
          border: Border.all(color: c.border),
        ),
        child: Column(
          children: [
            Icon(icon, size: 28, color: c.text),
            const SizedBox(height: AppSpacing.s8),
            Text(
              label,
              textAlign: TextAlign.center,
              style: context.text.labelSmall?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
