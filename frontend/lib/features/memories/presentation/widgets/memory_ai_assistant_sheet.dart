import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/navigation/router.dart';
import 'package:memory_verse/core/providers/home_assistant_controller.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;

class MemoryAiAssistantSheet extends ConsumerStatefulWidget {
  final MemoryModel memory;

  const MemoryAiAssistantSheet({
    super.key,
    required this.memory,
  });

  static Future<void> show(
    BuildContext context,
    MemoryModel memory, {
    required MemoryAssistantActionCallback onExecuteAction,
  }) async {
    final result = await showModalBottomSheet<_MemoryAssistantSheetResult>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      barrierColor: adt.AppColors.plum900.withValues(alpha: 0.5),
      builder: (ctx) => Padding(
        padding: EdgeInsets.only(
          bottom: MediaQuery.of(ctx).viewInsets.bottom,
        ),
        child: MemoryAiAssistantSheet(memory: memory),
      ),
    );

    if (!context.mounted || result == null) return;
    if (result.openGeneralChat) {
      context.push(Routes.ai);
    } else if (result.action != null) {
      onExecuteAction(result.action!, result.actionParams);
    }
  }

  @override
  ConsumerState<MemoryAiAssistantSheet> createState() =>
      _MemoryAiAssistantSheetState();
}

typedef MemoryAssistantActionCallback =
    void Function(String action, Map<String, dynamic> params);

class _MemoryAssistantSheetResult {
  final String? action;
  final Map<String, dynamic> actionParams;
  final bool openGeneralChat;

  const _MemoryAssistantSheetResult.action(this.action, this.actionParams)
      : openGeneralChat = false;

  const _MemoryAssistantSheetResult.openGeneralChat()
      : action = null,
        actionParams = const {},
        openGeneralChat = true;
}

class _MemoryAiAssistantSheetState
    extends ConsumerState<MemoryAiAssistantSheet> {
  final TextEditingController _controller = TextEditingController();
  final FocusNode _focusNode = FocusNode();

  static const _purple500 = Color(0xFF8B5CF6);
  static const _purple400 = Color(0xFFA78BFA);
  static const _purple300 = Color(0xFFC4B5FD);
  static const _purple200 = Color(0xFFDDD6FE);
  static const _purple600 = Color(0xFF7C3AED);
  static const _coral500 = Color(0xFFEF4444);
  static const _coral400 = Color(0xFFF87171);

  List<String> get _memoryChips => [
        'Check Readiness',
        'What should I add?',
        'Generate Video',
        'Summarize Memory',
        'Open AI Director',
      ];

  @override
  void dispose() {
    _controller.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  void _submit(String text) {
    final trimmed = text.trim();
    if (trimmed.isEmpty) return;
    _controller.clear();
    _focusNode.unfocus();
    ref
        .read(memoryAssistantProvider(widget.memory.id).notifier)
        .submitMemoryQuery(trimmed);
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(memoryAssistantProvider(widget.memory.id));

    return ClipRRect(
      borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 16, sigmaY: 16),
        child: Container(
          decoration: BoxDecoration(
            color: adt.AppColors.plum900.withValues(alpha: 0.94),
            borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
            border: Border.all(
              color: _purple500.withValues(alpha: 0.35),
              width: 1.2,
            ),
          ),
          child: SafeArea(
            top: false,
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(20, 12, 20, 20),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Drag Handle
                  Center(
                    child: Container(
                      width: 40,
                      height: 4,
                      margin: const EdgeInsets.only(bottom: 16),
                      decoration: BoxDecoration(
                        color: adt.AppColors.onDarkSecondary.withValues(alpha: 0.4),
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),

                  // Header Row
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(8),
                        decoration: const BoxDecoration(
                          shape: BoxShape.circle,
                          gradient: LinearGradient(
                            colors: [
                              _purple500,
                              _purple300,
                            ],
                          ),
                        ),
                        child: const Icon(
                          Icons.auto_awesome,
                          color: adt.AppColors.onDarkPrimary,
                          size: 18,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'AI Memory Assistant',
                              style: TextStyle(
                                color: adt.AppColors.onDarkPrimary,
                                fontSize: 16,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            const SizedBox(height: 2),
                            Row(
                              children: [
                                Container(
                                  width: 6,
                                  height: 6,
                                  decoration: const BoxDecoration(
                                    color: adt.AppColors.accent,
                                    shape: BoxShape.circle,
                                  ),
                                ),
                                const SizedBox(width: 6),
                                Flexible(
                                  child: Text(
                                    widget.memory.title,
                                    style: TextStyle(
                                      color: _purple200.withValues(alpha: 0.9),
                                      fontSize: 12,
                                      fontWeight: FontWeight.w500,
                                    ),
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                      IconButton(
                        icon: const Icon(Icons.close_rounded),
                        color: adt.AppColors.onDarkSecondary,
                        onPressed: () => Navigator.pop(context),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),

                  // Context Suggestion Chips
                  SingleChildScrollView(
                    scrollDirection: Axis.horizontal,
                    physics: const BouncingScrollPhysics(),
                    child: Row(
                      children: _memoryChips.map((chipText) {
                        return Padding(
                          padding: const EdgeInsets.only(right: 8),
                          child: ActionChip(
                            label: Text(
                              chipText,
                              style: TextStyle(
                                color: _purple200.withValues(alpha: 0.95),
                                fontSize: 12,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                            backgroundColor:
                                adt.AppColors.plum800.withValues(alpha: 0.7),
                            side: BorderSide(
                              color: _purple500.withValues(alpha: 0.3),
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                            onPressed: () => _submit(chipText),
                          ),
                        );
                      }).toList(),
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Input row
                  Container(
                    decoration: BoxDecoration(
                      color: adt.AppColors.plum800.withValues(alpha: 0.8),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(
                        color: _purple500.withValues(alpha: 0.3),
                      ),
                    ),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 14,
                      vertical: 4,
                    ),
                    child: Row(
                      children: [
                        Expanded(
                          child: TextField(
                            controller: _controller,
                            focusNode: _focusNode,
                            style: const TextStyle(
                              color: adt.AppColors.onDarkPrimary,
                              fontSize: 14,
                            ),
                            decoration: InputDecoration(
                              hintText: 'Ask about "${widget.memory.title}"...',
                              hintStyle: TextStyle(
                                color: adt.AppColors.onDarkSecondary
                                    .withValues(alpha: 0.6),
                                fontSize: 14,
                              ),
                              border: InputBorder.none,
                              isDense: true,
                              contentPadding:
                                  const EdgeInsets.symmetric(vertical: 10),
                            ),
                            textInputAction: TextInputAction.send,
                            onSubmitted: _submit,
                          ),
                        ),
                        if (state.isLoading)
                          const Padding(
                            padding: EdgeInsets.symmetric(horizontal: 8),
                            child: SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                valueColor: AlwaysStoppedAnimation<Color>(
                                  _purple400,
                                ),
                              ),
                            ),
                          )
                        else
                          IconButton(
                            icon: const Icon(
                              Icons.arrow_upward_rounded,
                              size: 20,
                            ),
                            color: _purple300,
                            onPressed: () => _submit(_controller.text),
                            tooltip: 'Send',
                          ),
                      ],
                    ),
                  ),

                  // Response area if available
                  if (state.responseMessage != null || state.error != null) ...[
                    const SizedBox(height: 16),
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: adt.AppColors.plum800.withValues(alpha: 0.9),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(
                          color: state.error != null
                              ? _coral500.withValues(alpha: 0.4)
                              : _purple500.withValues(alpha: 0.25),
                        ),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Icon(
                                state.error != null
                                    ? Icons.error_outline_rounded
                                    : Icons.smart_toy_outlined,
                                size: 18,
                                color: state.error != null
                                    ? _coral500
                                    : _purple300,
                              ),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Text(
                                  state.error ?? state.responseMessage ?? '',
                                  style: TextStyle(
                                    color: state.error != null
                                        ? _coral400
                                        : adt.AppColors.onDarkPrimary
                                            .withValues(alpha: 0.95),
                                    fontSize: 13.5,
                                    height: 1.45,
                                  ),
                                ),
                              ),
                            ],
                          ),
                          if (state.action != null) ...[
                            const SizedBox(height: 12),
                            Align(
                              alignment: Alignment.centerRight,
                              child: ElevatedButton.icon(
                                icon: Icon(
                                  _getActionIcon(state.action!),
                                  size: 16,
                                  color: adt.AppColors.onDarkPrimary,
                                ),
                                label: Text(
                                  _getActionLabel(state.action!),
                                  style: const TextStyle(
                                    fontSize: 13,
                                    fontWeight: FontWeight.w600,
                                    color: adt.AppColors.onDarkPrimary,
                                  ),
                                ),
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: _purple600,
                                  foregroundColor: adt.AppColors.onDarkPrimary,
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 16,
                                    vertical: 8,
                                  ),
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                ),
                                onPressed: () {
                                  final action = state.action!;
                                  final actionParams = state.actionParams;
                                  Navigator.pop(
                                    context,
                                    _MemoryAssistantSheetResult.action(
                                      action,
                                      actionParams,
                                    ),
                                  );
                                },
                              ),
                            ),
                          ],
                        ],
                      ),
                    ),
                  ],

                  const SizedBox(height: 16),

                  // Full AI Chat Link Button
                  OutlinedButton.icon(
                    icon: const Icon(Icons.chat_bubble_outline_rounded, size: 16),
                    label: const Text('Open General AI Chat'),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: _purple200,
                      side: BorderSide(
                        color: _purple500.withValues(alpha: 0.35),
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                    ),
                    onPressed: () => Navigator.pop(
                      context,
                      const _MemoryAssistantSheetResult.openGeneralChat(),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  IconData _getActionIcon(String action) {
    switch (action) {
      case 'ADD_MEDIA':
        return Icons.add_photo_alternate_rounded;
      case 'OPEN_MEMORY':
        return Icons.photo_library_outlined;
      case 'CREATE_MEMORY':
        return Icons.add_photo_alternate_outlined;
      case 'WATCH_VIDEO_JOB':
        return Icons.slow_motion_video_rounded;
      case 'OPEN_VIDEO':
        return Icons.play_circle_fill_rounded;
      case 'OPEN_AI_DIRECTOR':
        return Icons.movie_filter_rounded;
      case 'SHOW_READINESS':
        return Icons.fact_check_rounded;
      default:
        return Icons.arrow_forward_rounded;
    }
  }

  String _getActionLabel(String action) {
    switch (action) {
      case 'ADD_MEDIA':
        return 'Add Moments';
      case 'OPEN_MEMORY':
        return 'Open Memory';
      case 'CREATE_MEMORY':
        return 'Create Memory';
      case 'WATCH_VIDEO_JOB':
        return 'View Generation';
      case 'OPEN_VIDEO':
        return 'Play Reel';
      case 'OPEN_AI_DIRECTOR':
        return 'Open AI Director';
      case 'SHOW_READINESS':
        return 'Check Details';
      default:
        return 'Proceed';
    }
  }
}
