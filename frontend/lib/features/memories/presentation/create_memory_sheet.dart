import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as design;
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/core/widgets/buttons.dart';
import 'package:memory_verse/core/widgets/inputs.dart';
import 'package:memory_verse/features/memories/presentation/memory_detail_screen.dart';

class CreateMemorySheet extends ConsumerStatefulWidget {
  final String? vaultId;
  const CreateMemorySheet({super.key, this.vaultId});

  static void show(BuildContext context, {String? vaultId}) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      barrierColor: Colors.black.withValues(alpha: 0.2),
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
      );

      // Invalidate the memories list
      ref.invalidate(memoriesListProvider);

      if (mounted) {
        Navigator.pop(context);
        MemoryDetailScreen.open(context, memory);
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _error = 'Failed to create memory. Please try again.';
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
            color: design.AppColors.plum900.withValues(alpha: 0.5),
            border: Border(
              top: BorderSide(
                color: Colors.white.withValues(alpha: 0.15),
                width: 1,
              ),
            ),
          ),
      child: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.symmetric(
            horizontal: AppSpacing.s24,
            vertical: AppSpacing.s32,
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
                      fontFamily: 'Inter',
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
              const SizedBox(height: AppSpacing.s24),
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
                              color: Colors.white.withValues(alpha: 0.06),
                              borderRadius: BorderRadius.circular(AppRadii.lg),
                              border: Border.all(color: Colors.white.withValues(alpha: 0.15)),
                            ),
                            child: Row(
                              children: [
                                Icon(
                                  Icons.calendar_today_rounded,
                                  size: 16,
                                  color: c.textMuted,
                                ),
                                const SizedBox(width: AppSpacing.s8),
                                Text(
                                  DateFormat('MMM d, yyyy').format(_date),
                                  style: TextStyle(color: c.text, fontSize: 14),
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
              if (_error != null) ...[
                const SizedBox(height: AppSpacing.s16),
                Text(_error!, style: TextStyle(color: c.error, fontSize: 13)),
              ],
              const SizedBox(height: AppSpacing.s32),
              PrimaryButton(
                label: 'Create',
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
