import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/services/media/media_service.dart';
import 'package:memory_verse/core/services/rag/rag_retrieval_service.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;
import 'package:memory_verse/core/widgets/buttons.dart';
import 'package:memory_verse/core/widgets/inputs.dart';
import 'package:memory_verse/core/widgets/media.dart';

class MemoryRagReviewView extends StatefulWidget {
  final RagRetrievalResult ragResult;
  final VoidCallback onAddManualMedia;
  final Function({
    required String title,
    required String? description,
    required DateTime date,
    required String? location,
    required List<DiscoveredMediaItem> selectedMedia,
    required int coverIndex,
    required bool autoGenerateVideo,
  }) onConfirm;
  final bool isLoading;

  const MemoryRagReviewView({
    super.key,
    required this.ragResult,
    required this.onAddManualMedia,
    required this.onConfirm,
    this.isLoading = false,
  });

  @override
  State<MemoryRagReviewView> createState() => _MemoryRagReviewViewState();
}

class _MemoryRagReviewViewState extends State<MemoryRagReviewView> {
  late TextEditingController _titleCtrl;
  late TextEditingController _descCtrl;
  late TextEditingController _locCtrl;
  late DateTime _selectedDate;
  late List<DiscoveredMediaItem> _items;
  final Set<String> _selectedItemIds = {};
  int _coverIndex = 0;
  bool _autoGenerateVideo = false;

  @override
  void initState() {
    super.initState();
    _titleCtrl = TextEditingController(text: widget.ragResult.suggestedTitle);
    _descCtrl = TextEditingController(text: widget.ragResult.suggestedDescription ?? '');
    _locCtrl = TextEditingController(text: widget.ragResult.suggestedLocation ?? '');
    _selectedDate = widget.ragResult.suggestedDate ?? DateTime.now();
    
    _items = widget.ragResult.matchedItems;
    // ONLY pre-select items that the AI recommended as relevant
    for (final match in widget.ragResult.matches) {
      if (match.recommended) {
        _selectedItemIds.add(match.item.id);
      }
    }
  }

  @override
  void dispose() {
    _titleCtrl.dispose();
    _descCtrl.dispose();
    _locCtrl.dispose();
    super.dispose();
  }

  List<DiscoveredMediaItem> get _selectedItems =>
      _items.where((it) => _selectedItemIds.contains(it.id)).toList();

  Future<void> _pickDate() async {
    final c = context.colors;
    final picked = await showDatePicker(
      context: context,
      initialDate: _selectedDate,
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
    if (picked != null) {
      setState(() => _selectedDate = picked);
    }
  }

  void _previewItem(DiscoveredMediaItem item) {
    if (item.file == null || !item.file!.existsSync()) return;

    if (item.isVideo) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Video ready for memory inclusion')),
      );
      return;
    }

    showDialog(
      context: context,
      builder: (ctx) => Dialog(
        backgroundColor: Colors.transparent,
        insetPadding: EdgeInsets.zero,
        child: Stack(
          fit: StackFit.expand,
          children: [
            InteractiveViewer(
              child: AppFileImage(file: item.file!, fit: BoxFit.contain),
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

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final selectedCount = _selectedItems.length;
    final recCount = widget.ragResult.totalRecommended;
    final totalCount = widget.ragResult.totalScanned;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        // 1. Discovery Banner
        Container(
          padding: const EdgeInsets.all(AppSpacing.s16),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [
                c.primary.withValues(alpha: 0.25),
                adt.AppColors.plum900.withValues(alpha: 0.9),
              ],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(AppRadii.lg),
            border: Border.all(color: c.primary.withValues(alpha: 0.3)),
          ),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: c.primary.withValues(alpha: 0.2),
                  shape: BoxShape.circle,
                ),
                child: Icon(Icons.auto_awesome, color: c.primary, size: 22),
              ),
              const SizedBox(width: AppSpacing.s12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      recCount > 0
                          ? 'AI Curation ($recCount of $totalCount relevant)'
                          : 'AI Media Discovery',
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                        color: c.text,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      recCount > 0
                          ? 'AI found $recCount photos matching "${widget.ragResult.prompt}". Tap any to customize.'
                          : 'Scanned $totalCount photos for "${widget.ragResult.prompt}". Select any photos to include.',
                      style: TextStyle(fontSize: 12, color: c.textMuted),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: AppSpacing.s16),

        // 2. Metadata Editor
        AppTextField(
          label: 'Memory Title',
          controller: _titleCtrl,
        ),
        const SizedBox(height: AppSpacing.s12),
        Row(
          children: [
            Expanded(
              child: AppTextField(
                label: 'Location (optional)',
                controller: _locCtrl,
                hint: 'e.g. Goa',
              ),
            ),
            const SizedBox(width: AppSpacing.s12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Date',
                    style: Theme.of(context).textTheme.labelSmall?.copyWith(color: c.textMuted),
                  ),
                  const SizedBox(height: AppSpacing.s8),
                  GestureDetector(
                    onTap: _pickDate,
                    child: Container(
                      height: 52,
                      padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s12),
                      decoration: BoxDecoration(
                        color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.06),
                        borderRadius: BorderRadius.circular(AppRadii.lg),
                        border: Border.all(color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.15)),
                      ),
                      child: Row(
                        children: [
                          Icon(Icons.calendar_today_rounded, size: 16, color: c.textMuted),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              DateFormat('MMM d, yyyy').format(_selectedDate),
                              style: TextStyle(color: c.text, fontSize: 13),
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
        const SizedBox(height: AppSpacing.s16),

        // 3. Media Grid Header & Actions
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Selected Media ($selectedCount / ${_items.length})',
              style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: c.text),
            ),
            Row(
              children: [
                TextButton(
                  onPressed: () {
                    setState(() {
                      if (_selectedItemIds.length == _items.length) {
                        _selectedItemIds.clear();
                      } else {
                        _selectedItemIds.addAll(_items.map((e) => e.id));
                      }
                    });
                  },
                  child: Text(
                    _selectedItemIds.length == _items.length ? 'Deselect All' : 'Select All',
                    style: const TextStyle(fontSize: 12),
                  ),
                ),
                TextButton.icon(
                  onPressed: widget.onAddManualMedia,
                  icon: const Icon(Icons.add, size: 16),
                  label: const Text('Add More', style: TextStyle(fontSize: 12)),
                ),
              ],
            ),
          ],
        ),
        const SizedBox(height: AppSpacing.s8),

        // 4. Media Grid
        if (_items.isEmpty)
          Container(
            padding: const EdgeInsets.all(AppSpacing.s24),
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: c.surfaceElevated,
              borderRadius: BorderRadius.circular(AppRadii.lg),
            ),
            child: Column(
              children: [
                Icon(Icons.image_not_supported_outlined, size: 40, color: c.textMuted),
                const SizedBox(height: 8),
                Text('No media found matching this prompt', style: TextStyle(color: c.textMuted, fontSize: 13)),
                const SizedBox(height: 8),
                TextButton.icon(
                  onPressed: widget.onAddManualMedia,
                  icon: const Icon(Icons.add_photo_alternate_outlined),
                  label: const Text('Add Photos Manually'),
                ),
              ],
            ),
          )
        else
          SizedBox(
            height: 140,
            child: Builder(
              builder: (context) {
                final matchMap = {for (final m in widget.ragResult.matches) m.item.id: m};
                return ListView.builder(
                  scrollDirection: Axis.horizontal,
                  itemCount: _items.length,
                  itemBuilder: (ctx, i) {
                    final item = _items[i];
                    final isSelected = _selectedItemIds.contains(item.id);
                    final isCover = _coverIndex == i;
                    final match = matchMap[item.id];
                    final isRecommended = match?.recommended ?? false;
                    final scorePct = match != null ? (match.relevanceScore * 100).toInt() : null;

                    return GestureDetector(
                      onTap: () {
                        setState(() {
                          if (isSelected) {
                            _selectedItemIds.remove(item.id);
                          } else {
                            _selectedItemIds.add(item.id);
                          }
                        });
                      },
                      child: Container(
                        width: 110,
                        margin: const EdgeInsets.only(right: AppSpacing.s10),
                        decoration: BoxDecoration(
                          color: c.surfaceElevated,
                          borderRadius: BorderRadius.circular(AppRadii.md),
                          border: Border.all(
                            color: isCover
                                ? c.primary
                                : isSelected
                                    ? c.primary.withValues(alpha: 0.6)
                                    : c.border.withValues(alpha: 0.4),
                            width: isCover ? 2.5 : isSelected ? 1.5 : 1,
                          ),
                        ),
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            // Image or Video Thumbnail
                            ClipRRect(
                              borderRadius: BorderRadius.circular(AppRadii.md),
                              child: item.file != null && (kIsWeb || item.file!.existsSync())
                                  ? (item.isVideo
                                      ? Container(
                                          color: adt.AppColors.plum900,
                                          child: const Center(
                                            child: Icon(Icons.videocam_rounded, color: Colors.white, size: 28),
                                          ),
                                        )
                                      : AppFileImage(file: item.file!, fit: BoxFit.cover))
                                  : Container(
                                      color: adt.AppColors.plum900,
                                      child: Icon(Icons.broken_image, color: c.textMuted),
                                    ),
                            ),

                            // Dimmer if not selected
                            if (!isSelected)
                              Container(
                                decoration: BoxDecoration(
                                  color: Colors.black.withValues(alpha: 0.6),
                                  borderRadius: BorderRadius.circular(AppRadii.md),
                                ),
                              ),

                            // AI Match badge
                            if (isRecommended && scorePct != null)
                              Positioned(
                                top: 4,
                                left: 24,
                                child: Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                                  decoration: BoxDecoration(
                                    color: Colors.green.shade700,
                                    borderRadius: BorderRadius.circular(4),
                                  ),
                                  child: Text(
                                    'AI MATCH $scorePct%',
                                    style: const TextStyle(
                                      fontSize: 7.5,
                                      fontWeight: FontWeight.w700,
                                      color: Colors.white,
                                    ),
                                  ),
                                ),
                              ),

                            // Selection Checkbox Badge
                            Positioned(
                              top: 4,
                              right: 4,
                              child: Container(
                                padding: const EdgeInsets.all(2),
                                decoration: BoxDecoration(
                                  color: isSelected ? c.primary : Colors.black54,
                                  shape: BoxShape.circle,
                                ),
                                child: Icon(
                                  isSelected ? Icons.check : Icons.circle_outlined,
                                  size: 14,
                                  color: Colors.white,
                                ),
                              ),
                            ),

                            // Cover Badge / Button
                            Positioned(
                              bottom: 4,
                              left: 4,
                              child: isCover
                                  ? Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
                                      decoration: BoxDecoration(
                                        color: c.primary,
                                        borderRadius: BorderRadius.circular(4),
                                      ),
                                      child: const Text(
                                        'COVER',
                                        style: TextStyle(
                                          fontSize: 9,
                                          fontWeight: FontWeight.w800,
                                          color: Colors.white,
                                        ),
                                      ),
                                    )
                                  : GestureDetector(
                                      onTap: () => setState(() => _coverIndex = i),
                                      child: Container(
                                        padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
                                        decoration: BoxDecoration(
                                          color: Colors.black54,
                                          borderRadius: BorderRadius.circular(4),
                                        ),
                                        child: const Text(
                                          'Set Cover',
                                          style: TextStyle(
                                            fontSize: 9,
                                            color: Colors.white,
                                          ),
                                        ),
                                      ),
                                    ),
                            ),

                            // Preview Tap
                            Positioned(
                              top: 4,
                              left: 4,
                              child: GestureDetector(
                                onTap: () => _previewItem(item),
                                child: Container(
                                  padding: const EdgeInsets.all(3),
                                  decoration: const BoxDecoration(
                                    color: Colors.black54,
                                    shape: BoxShape.circle,
                                  ),
                                  child: const Icon(Icons.fullscreen, size: 12, color: Colors.white),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                );
              },
            ),
          ),
        const SizedBox(height: AppSpacing.s16),

        // 5. Video reel option
        Container(
          padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s12, vertical: AppSpacing.s8),
          decoration: BoxDecoration(
            color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.04),
            borderRadius: BorderRadius.circular(AppRadii.md),
            border: Border.all(color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.08)),
          ),
          child: Row(
            children: [
              Icon(Icons.movie_creation_outlined, color: c.primary, size: 20),
              const SizedBox(width: AppSpacing.s12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Stitch into Cinematic Video Reel',
                      style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: c.text),
                    ),
                    Text(
                      'Automatically compile these memories into an MP4 video',
                      style: TextStyle(fontSize: 11, color: c.textMuted),
                    ),
                  ],
                ),
              ),
              Switch(
                value: _autoGenerateVideo,
                activeThumbColor: c.primary,
                onChanged: (val) => setState(() => _autoGenerateVideo = val),
              ),
            ],
          ),
        ),
        const SizedBox(height: AppSpacing.s24),

        // 6. Action Button
        PrimaryButton(
          label: selectedCount == 0
              ? 'Select at least 1 memory'
              : 'Create Memory ($selectedCount items)',
          isLoading: widget.isLoading,
          onPressed: selectedCount == 0
              ? null
              : () {
                  widget.onConfirm(
                    title: _titleCtrl.text.trim().isNotEmpty
                        ? _titleCtrl.text.trim()
                        : widget.ragResult.suggestedTitle,
                    description: _descCtrl.text.trim().isNotEmpty
                        ? _descCtrl.text.trim()
                        : widget.ragResult.suggestedDescription,
                    date: _selectedDate,
                    location: _locCtrl.text.trim().isNotEmpty ? _locCtrl.text.trim() : null,
                    selectedMedia: _selectedItems,
                    coverIndex: _coverIndex.clamp(0, _selectedItems.length - 1),
                    autoGenerateVideo: _autoGenerateVideo,
                  );
                },
        ),
      ],
    );
  }
}
