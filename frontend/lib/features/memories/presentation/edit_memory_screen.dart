import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/features/memories/presentation/multi_media_picker_screen.dart';

class EditMemoryScreen extends ConsumerStatefulWidget {
  final MemoryModel memory;
  const EditMemoryScreen({super.key, required this.memory});

  static void open(BuildContext context, MemoryModel memory) {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => EditMemoryScreen(memory: memory)),
    );
  }

  @override
  ConsumerState<EditMemoryScreen> createState() => _EditMemoryScreenState();
}

class _EditMemoryScreenState extends ConsumerState<EditMemoryScreen> {
  final _formKey = GlobalKey<FormState>();

  late TextEditingController _titleController;
  late TextEditingController _descController;
  late TextEditingController _locController;
  late DateTime _selectedDate;

  late List<MediaModel> _mediaList;
  String? _coverMediaId;

  bool _isSaving = false;

  @override
  void initState() {
    super.initState();
    _titleController = TextEditingController(text: widget.memory.title);
    _descController = TextEditingController(
      text: widget.memory.description ?? '',
    );
    _locController = TextEditingController(
      text: widget.memory.locationName ?? '',
    );
    _selectedDate = widget.memory.memoryDate;
    _mediaList = List.from(widget.memory.media);
    _coverMediaId = widget.memory.coverMediaId;
  }

  @override
  void dispose() {
    _titleController.dispose();
    _descController.dispose();
    _locController.dispose();
    super.dispose();
  }

  Future<void> _pickDate() async {
    final date = await showDatePicker(
      context: context,
      initialDate: _selectedDate,
      firstDate: DateTime(1900),
      lastDate: DateTime.now().add(const Duration(days: 365)),
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: ColorScheme.dark(
              primary: context.colors.primary,
              onPrimary: context.colors.primaryInverse,
              surface: context.colors.surface,
              onSurface: context.colors.text,
            ),
          ),
          child: child!,
        );
      },
    );
    if (date != null) {
      setState(() => _selectedDate = date);
    }
  }

  void _onReorderItem(int oldIndex, int newIndex) {
    setState(() {
      final item = _mediaList.removeAt(oldIndex);
      _mediaList.insert(newIndex, item);
    });
  }

  void _removeMedia(int index) {
    setState(() {
      final item = _mediaList[index];
      if (_coverMediaId == item.id) _coverMediaId = null;
      _mediaList.removeAt(index);
      // Note: we don't delete from server until save, or we can delete immediately?
      // Better to just delete it on the server right now to avoid complex diff logic for saves.
      // Wait, the prompt says "Save changes to Supabase... Flow: Save -> database update -> state update"
      // If we don't delete on server immediately, we must diff and delete during Save.
    });
  }

  Future<void> _saveChanges() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _isSaving = true);

    try {
      final mRepo = ref.read(memoryRepositoryProvider);
      final mediaRepo = ref.read(mediaRepositoryProvider);

      // 1. Calculate deleted media
      final originalIds = widget.memory.media.map((m) => m.id).toSet();
      final currentIds = _mediaList.map((m) => m.id).toSet();
      final deletedIds = originalIds.difference(currentIds);

      // Delete removed media
      for (final id in deletedIds) {
        await mediaRepo.deleteMedia(id);
      }

      // 2. Update memory fields
      await mRepo.updateMemory(
        widget.memory.id,
        title: _titleController.text.trim(),
        description: _descController.text.trim().isEmpty
            ? null
            : _descController.text.trim(),
        locationName: _locController.text.trim().isEmpty
            ? null
            : _locController.text.trim(),
        memoryDate: _selectedDate,
        coverMediaId: _coverMediaId,
      );

      // 3. Reorder media if changed
      // (Even if not changed visually, sending the order ensures it matches)
      if (currentIds.isNotEmpty) {
        await mediaRepo.reorderMedia(currentIds.toList());
      }

      // 4. Invalidate providers
      ref.invalidate(memoryDetailProvider(widget.memory.id));
      ref.invalidate(memoriesListProvider);
      ref.invalidate(timelineProvider);

      if (mounted) {
        Navigator.pop(context); // Go back to detail view
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to save: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isSaving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;

    return Scaffold(
      backgroundColor: c.bg,
      appBar: AppBar(
        backgroundColor: c.surface,
        title: Text(
          'Edit Memory',
          style: TextStyle(
            color: c.text,
            fontWeight: FontWeight.w600,
            fontSize: 16,
          ),
        ),
        leading: IconButton(
          icon: Icon(Icons.close, color: c.text),
          onPressed: () => Navigator.pop(context),
        ),
        actions: [
          _isSaving
              ? const Padding(
                  padding: EdgeInsets.all(16.0),
                  child: SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                )
              : TextButton(
                  onPressed: _saveChanges,
                  child: Text(
                    'Save',
                    style: TextStyle(
                      color: c.primary,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
        ],
      ),
      body: Form(
        key: _formKey,
        child: CustomScrollView(
          slivers: [
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.all(AppSpacing.s24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Title
                    TextFormField(
                      controller: _titleController,
                      style: TextStyle(
                        color: c.text,
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                      ),
                      decoration: InputDecoration(
                        labelText: 'Title',
                        labelStyle: TextStyle(color: c.textMuted),
                        filled: true,
                        fillColor: c.surfaceElevated,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(AppRadii.md),
                          borderSide: BorderSide.none,
                        ),
                      ),
                      validator: (val) => val == null || val.trim().isEmpty
                          ? 'Title is required'
                          : null,
                    ),
                    const SizedBox(height: AppSpacing.s16),

                    // Date & Location Row
                    Row(
                      children: [
                        Expanded(
                          child: InkWell(
                            onTap: _pickDate,
                            borderRadius: BorderRadius.circular(AppRadii.md),
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 16,
                                vertical: 16,
                              ),
                              decoration: BoxDecoration(
                                color: c.surfaceElevated,
                                borderRadius: BorderRadius.circular(
                                  AppRadii.md,
                                ),
                              ),
                              child: Row(
                                children: [
                                  Icon(
                                    Icons.calendar_today,
                                    size: 18,
                                    color: c.textMuted,
                                  ),
                                  const SizedBox(width: 12),
                                  Text(
                                    DateFormat('MMM d, yyyy')
                                        .format(_selectedDate),
                                    style: TextStyle(color: c.text),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: AppSpacing.s16),
                        Expanded(
                          child: TextFormField(
                            controller: _locController,
                            style: TextStyle(color: c.text),
                            decoration: InputDecoration(
                              hintText: 'Location (optional)',
                              hintStyle: TextStyle(color: c.textMuted),
                              prefixIcon: Icon(
                                Icons.location_on_outlined,
                                size: 18,
                                color: c.textMuted,
                              ),
                              filled: true,
                              fillColor: c.surfaceElevated,
                              border: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(
                                  AppRadii.md,
                                ),
                                borderSide: BorderSide.none,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: AppSpacing.s16),

                    // Description
                    TextFormField(
                      controller: _descController,
                      style: TextStyle(color: c.text),
                      maxLines: 3,
                      decoration: InputDecoration(
                        labelText: 'Description (optional)',
                        labelStyle: TextStyle(color: c.textMuted),
                        filled: true,
                        fillColor: c.surfaceElevated,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(AppRadii.md),
                          borderSide: BorderSide.none,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            // Media Reorder Section
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'Media',
                      style: TextStyle(
                        color: c.text,
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    TextButton.icon(
                      icon: const Icon(Icons.add, size: 18),
                      label: const Text('Add Media'),
                      onPressed: () {
                        // Launch multimedia picker. Background uploads will happen.
                        // Ideally we should reload this view when background uploads finish,
                        // but since the user hasn't saved yet, it might be tricky.
                        // We will just show the standard picker.
                        MultiMediaPickerScreen.open(
                          context,
                          memoryId: widget.memory.id,
                        );
                      },
                    ),
                  ],
                ),
              ),
            ),

            if (_mediaList.isEmpty)
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.all(AppSpacing.s32),
                  child: Center(
                    child: Text(
                      'No media attached.',
                      style: TextStyle(color: c.textMuted),
                    ),
                  ),
                ),
              )
            else
              SliverReorderableList(
                itemBuilder: (context, index) {
                  final media = _mediaList[index];
                  final isCover = _coverMediaId == media.id;

                  return Container(
                    key: ValueKey(media.id),
                    margin: const EdgeInsets.symmetric(
                      horizontal: AppSpacing.s24,
                      vertical: AppSpacing.s4,
                    ),
                    decoration: BoxDecoration(
                      color: c.surfaceElevated,
                      borderRadius: BorderRadius.circular(AppRadii.sm),
                      border: isCover
                          ? Border.all(color: c.primary, width: 2)
                          : null,
                    ),
                    child: ListTile(
                      contentPadding: const EdgeInsets.only(left: 8, right: 0),
                      leading: ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: Image.network(
                          media.thumbnailUrl ?? media.url,
                          width: 48,
                          height: 48,
                          fit: BoxFit.cover,
                          errorBuilder: (_, __, ___) => Container(
                            width: 48,
                            height: 48,
                            color: c.surface,
                          ),
                        ),
                      ),
                      title: Text(
                        media.mediaType == 'video' ? 'Video' : 'Photo',
                        style: TextStyle(color: c.text, fontSize: 14),
                      ),
                      subtitle: isCover
                          ? Text(
                              'Cover',
                              style: TextStyle(color: c.primary, fontSize: 12),
                            )
                          : null,
                      trailing: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          PopupMenuButton<String>(
                            icon: Icon(Icons.more_vert, color: c.textMuted),
                            color: c.surface,
                            onSelected: (val) {
                              if (val == 'cover') {
                                setState(() => _coverMediaId = media.id);
                              }
                              if (val == 'remove') _removeMedia(index);
                            },
                            itemBuilder: (_) => [
                              if (media.mediaType == 'image')
                                const PopupMenuItem(
                                  value: 'cover',
                                  child: Text('Set as Cover'),
                                ),
                              const PopupMenuItem(
                                value: 'remove',
                                child: Text(
                                  'Remove',
                                  style: TextStyle(color: Colors.red),
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(width: 8),
                          Icon(Icons.drag_handle, color: c.textMuted),
                          const SizedBox(width: 16),
                        ],
                      ),
                    ),
                  );
                },
                itemCount: _mediaList.length,
                onReorderItem: _onReorderItem,
              ),

            const SliverToBoxAdapter(child: SizedBox(height: 100)),
          ],
        ),
      ),
    );
  }
}
