import 'dart:io';
import 'dart:ui';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/providers/upload_controller.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/core/services/media/media_service.dart';
import 'package:memory_verse/core/services/media/media_service_provider.dart';
import 'package:memory_verse/core/services/rag/rag_retrieval_service.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;
import 'package:memory_verse/core/widgets/buttons.dart';
import 'package:memory_verse/core/widgets/inputs.dart';
import 'package:memory_verse/core/widgets/media.dart';
import 'package:memory_verse/features/memories/presentation/memory_detail_screen.dart';
import 'package:memory_verse/features/memories/presentation/widgets/media_permission_view.dart';
import 'package:memory_verse/features/memories/presentation/widgets/memory_rag_review_view.dart';

enum _SheetMode {
  promptEntry,
  discovering,
  review,
  manualFallback,
}

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
  _SheetMode _mode = _SheetMode.promptEntry;
  final _promptCtrl = TextEditingController();

  // Manual fallback controllers
  final _titleCtrl = TextEditingController();
  final _descCtrl = TextEditingController();
  final _locCtrl = TextEditingController();
  final ImagePicker _picker = ImagePicker();
  final List<File> _selectedFiles = [];
  final List<String> _mediaTypes = [];
  DateTime _date = DateTime.now();

  RagRetrievalResult? _ragResult;
  String _discoveryStepText = 'Accessing media library...';
  bool _isLoading = false;
  String? _error;

  final List<String> _quickPrompts = [
    'Goa trip with friends',
    'Graduation celebration',
    'Family weekend getaway',
    'Birthday party memories',
  ];

  @override
  void dispose() {
    _promptCtrl.dispose();
    _titleCtrl.dispose();
    _descCtrl.dispose();
    _locCtrl.dispose();
    super.dispose();
  }

  Future<void> _startDiscoveryFlow() async {
    final prompt = _promptCtrl.text.trim();
    if (prompt.isEmpty) {
      setState(() => _error = 'Please enter a prompt describing your memory.');
      return;
    }

    setState(() {
      _mode = _SheetMode.discovering;
      _discoveryStepText = 'Checking media access permissions...';
      _error = null;
    });

    try {
      // 1. Check or request permission
      final mediaService = ref.read(mediaServiceProvider);
      var permState = await mediaService.checkPermission();
      if (permState == MediaPermissionState.notDetermined) {
        setState(() => _discoveryStepText = 'Requesting access to photos & videos...');
        permState = await mediaService.requestPermission();
      }

      if (permState == MediaPermissionState.denied ||
          permState == MediaPermissionState.permanentlyDenied) {
        setState(() {
          _mode = _SheetMode.promptEntry;
          _error = 'Media permission was not granted. You can grant access or add photos manually.';
        });
        return;
      }

      // 2. Discover media & query RAG engine
      setState(() => _discoveryStepText = 'Discovering accessible media...');
      await Future.delayed(const Duration(milliseconds: 300));

      setState(() => _discoveryStepText = 'RAG filtering memories matching "$prompt"...');
      final ragService = ref.read(ragRetrievalServiceProvider);
      final result = await ragService.retrieveMemoriesForPrompt(prompt: prompt);

      if (!mounted) return;

      setState(() {
        _ragResult = result;
        _mode = _SheetMode.review;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _mode = _SheetMode.promptEntry;
        _error = 'Discovery error: $e';
      });
    }
  }

  Future<void> _confirmAndCreateMemory({
    required String title,
    required String? description,
    required DateTime date,
    required String? location,
    required List<DiscoveredMediaItem> selectedMedia,
    required int coverIndex,
    required bool autoGenerateVideo,
  }) async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final repo = ref.read(memoryRepositoryProvider);
      final memory = await repo.createMemory(
        title: title,
        description: description,
        vaultId: widget.vaultId,
        locationName: location,
        memoryDate: date,
      );

      // Extract physical files for upload
      final filesToUpload = <File>[];
      final mediaTypes = <String>[];

      for (final item in selectedMedia) {
        if (item.file != null && item.file!.existsSync()) {
          filesToUpload.add(item.file!);
          mediaTypes.add(item.isVideo ? 'video' : 'image');
        } else if (item.mediaReference.isNotEmpty) {
          final f = File(item.mediaReference);
          if (f.existsSync()) {
            filesToUpload.add(f);
            mediaTypes.add(item.isVideo ? 'video' : 'image');
          }
        }
      }

      // Enqueue upload of only the curated memories
      if (filesToUpload.isNotEmpty) {
        ref.read(uploadControllerProvider.notifier).addUploads(
              files: filesToUpload,
              mediaTypes: mediaTypes,
              memoryId: memory.id,
              vaultId: widget.vaultId,
              coverIndex: coverIndex < filesToUpload.length ? coverIndex : 0,
            );
      }

      // Auto-generate video reel if selected
      if (autoGenerateVideo && filesToUpload.length >= 2) {
        try {
          await ref.read(mediaRepositoryProvider).generateVideo(memory.id);
        } catch (_) {}
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

  // ── Manual Fallback Methods ──

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

  Future<void> _pickFiles() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['jpg', 'jpeg', 'png', 'mp4', 'mov', 'heic'],
      allowMultiple: true,
    );
    if (result != null && result.files.isNotEmpty && mounted) {
      setState(() {
        for (var file in result.files) {
          if (file.path != null) {
            final ext = (file.extension ?? '').toLowerCase();
            final isVideo = ['mp4', 'mov'].contains(ext);
            _selectedFiles.add(File(file.path!));
            _mediaTypes.add(isVideo ? 'video' : 'image');
          }
        }
      });
    }
  }

  Future<void> _submitManual() async {
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
        description: _descCtrl.text.trim().isNotEmpty ? _descCtrl.text.trim() : null,
        vaultId: widget.vaultId,
        locationName: _locCtrl.text.trim().isNotEmpty ? _locCtrl.text.trim() : null,
        memoryDate: _date,
      );

      if (_selectedFiles.isNotEmpty) {
        ref.read(uploadControllerProvider.notifier).addUploads(
              files: _selectedFiles,
              mediaTypes: _mediaTypes,
              memoryId: memory.id,
              vaultId: widget.vaultId,
              coverIndex: 0,
            );
      }

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
            color: adt.AppColors.plum900.withValues(alpha: 0.88),
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
              child: _buildContent(context, c),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildContent(BuildContext context, AppColors c) {
    switch (_mode) {
      case _SheetMode.discovering:
        return _buildDiscoveringState(c);
      case _SheetMode.review:
        if (_ragResult != null) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Review Memories',
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
              const SizedBox(height: AppSpacing.s16),
              MemoryRagReviewView(
                ragResult: _ragResult!,
                isLoading: _isLoading,
                onAddManualMedia: () {
                  _pickFiles();
                },
                onConfirm: ({
                  required title,
                  required description,
                  required date,
                  required location,
                  required selectedMedia,
                  required coverIndex,
                  required autoGenerateVideo,
                }) {
                  _confirmAndCreateMemory(
                    title: title,
                    description: description,
                    date: date,
                    location: location,
                    selectedMedia: selectedMedia,
                    coverIndex: coverIndex,
                    autoGenerateVideo: autoGenerateVideo,
                  );
                },
              ),
              if (_error != null) ...[
                const SizedBox(height: AppSpacing.s16),
                Text(_error!, style: TextStyle(color: c.error, fontSize: 13)),
              ],
            ],
          );
        }
        return _buildPromptEntry(c);

      case _SheetMode.manualFallback:
        return _buildManualFallbackView(c);

      case _SheetMode.promptEntry:
        return _buildPromptEntry(c);
    }
  }

  Widget _buildPromptEntry(AppColors c) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(AppSpacing.s8),
                  decoration: BoxDecoration(
                    color: c.primary.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(AppRadii.md),
                  ),
                  child: Icon(Icons.auto_awesome, color: c.primary, size: 20),
                ),
                const SizedBox(width: AppSpacing.s12),
                Text(
                  'Create Memory',
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w700,
                    letterSpacing: -0.5,
                    color: c.text,
                  ),
                ),
              ],
            ),
            IconButton(
              icon: Icon(Icons.close_rounded, color: c.text),
              onPressed: () => Navigator.pop(context),
            ),
          ],
        ),
        const SizedBox(height: AppSpacing.s8),
        Text(
          'Describe what you want to remember. MemoryVerse will automatically discover and organize the right photos and videos.',
          style: TextStyle(color: c.textMuted, fontSize: 13, height: 1.4),
        ),
        const SizedBox(height: AppSpacing.s20),

        // Prompt Input
        AppTextField(
          label: 'Memory Prompt',
          hint: 'e.g. My Goa trip with my friends in 2025',
          controller: _promptCtrl,
          maxLines: 2,
        ),
        const SizedBox(height: AppSpacing.s12),

        // Quick Suggestion Chips
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: _quickPrompts.map((p) {
            return GestureDetector(
              onTap: () {
                _promptCtrl.text = p;
              },
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.06),
                  borderRadius: BorderRadius.circular(AppRadii.pill),
                  border: Border.all(
                    color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.12),
                  ),
                ),
                child: Text(
                  '+ $p',
                  style: TextStyle(fontSize: 12, color: c.textMuted),
                ),
              ),
            );
          }).toList(),
        ),
        const SizedBox(height: AppSpacing.s20),

        // Media Permission / Windows Folder view
        MediaPermissionView(
          onManualUploadRequested: () {
            setState(() => _mode = _SheetMode.manualFallback);
          },
          onPermissionGranted: () {
            if (_promptCtrl.text.trim().isNotEmpty) {
              _startDiscoveryFlow();
            }
          },
        ),

        if (_error != null) ...[
          const SizedBox(height: AppSpacing.s16),
          Text(_error!, style: TextStyle(color: c.error, fontSize: 13)),
        ],
        const SizedBox(height: AppSpacing.s24),

        // Primary Action: Discover & Filter Memories
        PrimaryButton(
          label: 'Discover Memories with AI',
          icon: Icons.auto_awesome,
          isLoading: _isLoading,
          onPressed: _startDiscoveryFlow,
        ),

        const SizedBox(height: AppSpacing.s12),
        Center(
          child: TextButton.icon(
            onPressed: () {
              setState(() => _mode = _SheetMode.manualFallback);
            },
            icon: const Icon(Icons.upload_file_outlined, size: 16),
            label: const Text('Add Photos Manually Instead'),
            style: TextButton.styleFrom(foregroundColor: c.textMuted),
          ),
        ),
      ],
    );
  }

  Widget _buildDiscoveringState(AppColors c) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: AppSpacing.s32),
      alignment: Alignment.center,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 64,
            height: 64,
            decoration: BoxDecoration(
              color: c.primary.withValues(alpha: 0.15),
              shape: BoxShape.circle,
            ),
            child: Center(
              child: SizedBox(
                width: 32,
                height: 32,
                child: CircularProgressIndicator(
                  color: c.primary,
                  strokeWidth: 3,
                ),
              ),
            ),
          ),
          const SizedBox(height: AppSpacing.s24),
          Text(
            'Finding Memories...',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w700,
              color: c.text,
            ),
          ),
          const SizedBox(height: AppSpacing.s8),
          Text(
            _discoveryStepText,
            textAlign: TextAlign.center,
            style: TextStyle(color: c.textMuted, fontSize: 13),
          ),
          const SizedBox(height: AppSpacing.s24),
          OutlinedButton(
            onPressed: () {
              setState(() => _mode = _SheetMode.promptEntry);
            },
            style: OutlinedButton.styleFrom(
              foregroundColor: c.textMuted,
              side: BorderSide(color: c.border),
            ),
            child: const Text('Cancel'),
          ),
        ],
      ),
    );
  }

  Widget _buildManualFallbackView(AppColors c) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              children: [
                IconButton(
                  icon: const Icon(Icons.arrow_back_rounded),
                  onPressed: () => setState(() => _mode = _SheetMode.promptEntry),
                ),
                Text(
                  'Manual Upload',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                    color: c.text,
                  ),
                ),
              ],
            ),
            IconButton(
              icon: Icon(Icons.close_rounded, color: c.text),
              onPressed: () => Navigator.pop(context),
            ),
          ],
        ),
        const SizedBox(height: AppSpacing.s16),
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
          maxLines: 2,
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
              child: GestureDetector(
                onTap: () async {
                  final picked = await showDatePicker(
                    context: context,
                    initialDate: _date,
                    firstDate: DateTime(1900),
                    lastDate: DateTime.now(),
                  );
                  if (picked != null) setState(() => _date = picked);
                },
                child: Container(
                  height: 52,
                  padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s16),
                  decoration: BoxDecoration(
                    color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.06),
                    borderRadius: BorderRadius.circular(AppRadii.lg),
                    border: Border.all(color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.15)),
                  ),
                  child: Row(
                    children: [
                      Icon(Icons.calendar_today_rounded, size: 16, color: c.textMuted),
                      const SizedBox(width: AppSpacing.s8),
                      Expanded(
                        child: Text(
                          DateFormat('MMM d, yyyy').format(_date),
                          style: TextStyle(color: c.text, fontSize: 13),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: AppSpacing.s20),

        // Media selection buttons
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Selected Files (${_selectedFiles.length})',
              style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: c.text),
            ),
            Row(
              children: [
                TextButton.icon(
                  onPressed: _pickImages,
                  icon: const Icon(Icons.add_photo_alternate_outlined, size: 18),
                  label: const Text('Photos'),
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
                            : AppFileImage(file: file, fit: BoxFit.cover),
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
                            child: const Icon(Icons.close, size: 14, color: Colors.white),
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
          label: _selectedFiles.isEmpty ? 'Create Memory' : 'Create & Upload (${_selectedFiles.length})',
          isLoading: _isLoading,
          onPressed: _submitManual,
        ),
      ],
    );
  }
}
