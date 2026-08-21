import 'dart:io';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as design;
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/providers/upload_controller.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/core/widgets/buttons.dart';
import 'package:memory_verse/core/widgets/inputs.dart';
import 'package:memory_verse/features/memories/presentation/memory_detail_screen.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;

class CreateMemorySheet extends ConsumerStatefulWidget {
  final String? vaultId;
  const CreateMemorySheet({super.key, this.vaultId});

  static void show(BuildContext context, {String? vaultId}) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      barrierColor: adt.AppColors.plum900.withValues(alpha: 0.2),
      builder: (ctx) => Theme(
        data: Theme.of(ctx).copyWith(extensions: const [AppColors.dark]),
        child: CreateMemorySheet(vaultId: vaultId),
      ),
    );
  }

  @override
  ConsumerState<CreateMemorySheet> createState() => _CreateMemorySheetState();
}

class _CreateMemorySheetState extends ConsumerState<CreateMemorySheet> {
  final _titleCtrl = TextEditingController();
  final _descCtrl = TextEditingController();
  final _locCtrl = TextEditingController();
  final ImagePicker _picker = ImagePicker();
  final List<File> _selectedFiles = [];
  final List<String> _mediaTypes = [];
  DateTime _date = DateTime.now();
  bool _isLoading = false;
  String? _error;

  @override
  void dispose() {
    _titleCtrl.dispose();
    _descCtrl.dispose();
    _locCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickImages() async {
    try {
      final images = await _picker.pickMultiImage(imageQuality: 80);
      if (images.isNotEmpty && mounted) {
        setState(() {
          _selectedFiles.addAll(images.map((x) => File(x.path)));
          _mediaTypes.addAll(List.generate(images.length, (_) => 'image'));
        });
      }
    } catch (e) {
      setState(() => _error = 'Failed to pick images: $e');
    }
  }

  Future<void> _pickVideo() async {
    try {
      final video = await _picker.pickVideo(source: ImageSource.gallery);
      if (video != null && mounted) {
        setState(() {
          _selectedFiles.add(File(video.path));
          _mediaTypes.add('video');
        });
      }
    } catch (e) {
      setState(() => _error = 'Failed to pick video: $e');
    }
  }

  Future<void> _pickDate() async {
    final c = context.colors;
    final picked = await showDatePicker(
      context: context,
      initialDate: _date,
      firstDate: DateTime(1900),
      lastDate: DateTime.now(),
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: ColorScheme.light(
              primary: c.primary,
              onPrimary: c.primaryInverse,
              surface: c.surface,
              onSurface: c.text,
            ),
          ),
          child: child!,
        );
      },
    );
    if (picked != null && mounted) {
      setState(() => _date = picked);
    }
  }

  Future<void> _submit() async {
    if (_titleCtrl.text.trim().isEmpty) {
      setState(() => _error = 'Please enter a title for your memory.');
      return;
    }
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final repo = ref.read(memoryRepositoryProvider);
      final memory = await repo.createMemory(
        title: _titleCtrl.text.trim(),
        description: _descCtrl.text.trim().isNotEmpty
            ? _descCtrl.text.trim()
            : null,
        vaultId: widget.vaultId,
        locationName: _locCtrl.text.trim().isNotEmpty
            ? _locCtrl.text.trim()
            : null,
        memoryDate: _date,
      );

      // Enqueue multi-image upload if files were selected
      if (_selectedFiles.isNotEmpty) {
        ref.read(uploadControllerProvider.notifier).addUploads(
              files: _selectedFiles,
              mediaTypes: _mediaTypes,
              memoryId: memory.id,
              vaultId: widget.vaultId,
              coverIndex: 0,
            );
      }

      // Invalidate providers
      ref.invalidate(memoriesListProvider);
      ref.invalidate(timelineProvider);
      ref.invalidate(vaultsListProvider);

      if (mounted) {
        Navigator.pop(context);
        MemoryDetailScreen.open(context, memory);
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _error = e.toString().replaceAll('Exception: ', '');
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return ClipRRect(
      borderRadius: const BorderRadius.vertical(
        top: Radius.circular(AppRadii.xl),
      ),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 64, sigmaY: 64),
        child: Container(
          padding: EdgeInsets.only(
            bottom: MediaQuery.of(context).viewInsets.bottom,
          ),
          decoration: BoxDecoration(
            color: design.AppColors.plum900.withValues(alpha: 0.85),
            border: Border(
              top: BorderSide(
                color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.15),
                width: 1,
              ),
            ),
          ),
          child: SafeArea(
            top: false,
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(
                horizontal: AppSpacing.s24,
                vertical: AppSpacing.s24,
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'Create Memory',
                        style: TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.w700,
                          letterSpacing: -0.5,
                          color: c.text,
                        ),
                      ),
                      IconButton(
                        icon: Icon(Icons.close_rounded, color: c.text),
                        onPressed: () => Navigator.pop(context),
                      ),
                    ],
                  ),
                  const SizedBox(height: AppSpacing.s20),
                  AppTextField(
                    label: 'Title (required)',
                    hint: 'e.g. Goa Trip 2026',
                    controller: _titleCtrl,
                  ),
                  const SizedBox(height: AppSpacing.s16),
                  AppTextField(
                    label: 'Description (optional)',
                    hint: 'What made this memorable?',
                    controller: _descCtrl,
                    maxLines: 3,
                  ),
                  const SizedBox(height: AppSpacing.s16),
                  Row(
                    children: [
                      Expanded(
                        child: AppTextField(
                          label: 'Location (optional)',
                          hint: 'e.g. Goa',
                          controller: _locCtrl,
                        ),
                      ),
                      const SizedBox(width: AppSpacing.s16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Date',
                              style: Theme.of(context).textTheme.labelSmall
                                  ?.copyWith(color: c.textMuted),
                            ),
                            const SizedBox(height: AppSpacing.s8),
                            GestureDetector(
                              onTap: _pickDate,
                              child: Container(
                                height: 52,
                                padding: const EdgeInsets.symmetric(
                                  horizontal: AppSpacing.s16,
                                ),
                                decoration: BoxDecoration(
                                  color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.06),
                                  borderRadius: BorderRadius.circular(AppRadii.lg),
                                  border: Border.all(color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.15)),
                                ),
                                child: Row(
                                  children: [
                                    Icon(
                                      Icons.calendar_today_rounded,
                                      size: 16,
                                      color: c.textMuted,
                                    ),
                                    const SizedBox(width: AppSpacing.s8),
                                    Expanded(
                                      child: Text(
                                        DateFormat('MMM d, yyyy').format(_date),
                                        style: TextStyle(color: c.text, fontSize: 14),
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: AppSpacing.s20),

                  // Media selection section
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'Photos & Media (${_selectedFiles.length})',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          color: c.text,
                        ),
                      ),
                      Row(
                        children: [
                          TextButton.icon(
                            onPressed: _pickImages,
                            icon: const Icon(Icons.add_photo_alternate_outlined, size: 18),
                            label: const Text('Add Photos'),
                          ),
                          TextButton.icon(
                            onPressed: _pickVideo,
                            icon: const Icon(Icons.videocam_outlined, size: 18),
                            label: const Text('Video'),
                          ),
                        ],
                      ),
                    ],
                  ),
                  if (_selectedFiles.isNotEmpty) ...[
                    const SizedBox(height: AppSpacing.s8),
                    SizedBox(
                      height: 80,
                      child: ListView.builder(
                        scrollDirection: Axis.horizontal,
                        itemCount: _selectedFiles.length,
                        itemBuilder: (ctx, i) {
                          final file = _selectedFiles[i];
                          final isVideo = _mediaTypes[i] == 'video';
                          return Container(
                            width: 80,
                            margin: const EdgeInsets.only(right: AppSpacing.s8),
                            decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(AppRadii.md),
                              border: Border.all(color: c.border),
                            ),
                            child: Stack(
                              fit: StackFit.expand,
                              children: [
                                ClipRRect(
                                  borderRadius: BorderRadius.circular(AppRadii.md),
                                  child: isVideo
                                      ? Container(
                                          color: adt.AppColors.plum900,
                                          child: const Icon(Icons.videocam, color: Colors.white),
                                        )
                                      : Image.file(file, fit: BoxFit.cover),
                                ),
                                Positioned(
                                  top: 2,
                                  right: 2,
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
                  ],

                  if (_error != null) ...[
                    const SizedBox(height: AppSpacing.s16),
                    Text(_error!, style: TextStyle(color: c.error, fontSize: 13)),
                  ],
                  const SizedBox(height: AppSpacing.s24),
                  PrimaryButton(
                    label: _selectedFiles.isEmpty
                        ? 'Create Memory'
                        : 'Create & Upload (${_selectedFiles.length})',
                    isLoading: _isLoading,
                    onPressed: _submit,
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
