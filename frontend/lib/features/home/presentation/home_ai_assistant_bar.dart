import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/home_assistant_controller.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as design;

class HomeAiAssistantBar extends ConsumerStatefulWidget {
  const HomeAiAssistantBar({super.key});

  @override
  ConsumerState<HomeAiAssistantBar> createState() => _HomeAiAssistantBarState();
}

class _HomeAiAssistantBarState extends ConsumerState<HomeAiAssistantBar> {
  final TextEditingController _textController = TextEditingController();
  final FocusNode _focusNode = FocusNode();

  static const _purple500 = Color(0xFF8B5CF6);
  static const _purple400 = Color(0xFFA78BFA);
  static const _purple300 = Color(0xFFC4B5FD);
  static const _purple200 = Color(0xFFDDD6FE);
  static const _purple600 = Color(0xFF7C3AED);
  static const _purple900 = Color(0xFF4C1D95);

  @override
  void dispose() {
    _textController.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  void _submit(String text) {
    if (text.trim().isEmpty) return;
    _textController.clear();
    _focusNode.unfocus();
    ref.read(homeAssistantProvider.notifier).submitQuery(text);
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(homeAssistantProvider);

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: AppSpacing.s24),
      decoration: BoxDecoration(
        color: design.AppColors.plum800.withValues(alpha: 0.85),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: _purple500.withValues(alpha: 0.35),
          width: 1.2,
        ),
        boxShadow: [
          BoxShadow(
            color: _purple900.withValues(alpha: 0.25),
            blurRadius: 16,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          // Input field row
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 8, 8),
            child: Row(
              children: [
                ShaderMask(
                  shaderCallback: (bounds) => const LinearGradient(
                    colors: [Color(0xFF9D4EDD), Color(0xFFE0AAFF)],
                  ).createShader(bounds),
                  child: const Icon(
                    Icons.auto_awesome,
                    color: Colors.white,
                    size: 22,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: TextField(
                    controller: _textController,
                    focusNode: _focusNode,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                    ),
                    decoration: InputDecoration(
                      hintText: 'Ask MemoryVerse or run command...',
                      hintStyle: TextStyle(
                        color: Colors.white.withValues(alpha: 0.45),
                        fontSize: 14,
                      ),
                      border: InputBorder.none,
                      isDense: true,
                      contentPadding: const EdgeInsets.symmetric(vertical: 8),
                    ),
                    textInputAction: TextInputAction.send,
                    onSubmitted: _submit,
                  ),
                ),
                if (state.isLoading)
                  const Padding(
                    padding: EdgeInsets.symmetric(horizontal: 12),
                    child: SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        valueColor: AlwaysStoppedAnimation<Color>(_purple400),
                      ),
                    ),
                  )
                else
                  IconButton(
                    icon: const Icon(Icons.arrow_upward_rounded, size: 20),
                    color: _purple300,
                    onPressed: () => _submit(_textController.text),
                    tooltip: 'Send command',
                  ),
              ],
            ),
          ),

          // Suggestion chips
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 0, 12, 10),
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              physics: const BouncingScrollPhysics(),
              child: Row(
                children: state.suggestedChips.map((chipText) {
                  return Padding(
                    padding: const EdgeInsets.only(right: 8),
                    child: ActionChip(
                      label: Text(
                        chipText,
                        style: TextStyle(
                          color: _purple200.withValues(alpha: 0.9),
                          fontSize: 12,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      backgroundColor: design.AppColors.plum900.withValues(alpha: 0.6),
                      side: BorderSide(
                        color: _purple500.withValues(alpha: 0.25),
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
          ),

          // Expanded Assistant response card
          if (state.isExpanded && (state.responseMessage != null || state.error != null)) ...[
            const Divider(color: Colors.white12, height: 1),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: design.AppColors.plum900.withValues(alpha: 0.85),
                borderRadius: const BorderRadius.only(
                  bottomLeft: Radius.circular(18),
                  bottomRight: Radius.circular(18),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Icon(
                        Icons.smart_toy_outlined,
                        size: 18,
                        color: _purple300,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          state.error ?? state.responseMessage ?? '',
                          style: TextStyle(
                            color: state.error != null
                                ? Colors.redAccent
                                : Colors.white.withValues(alpha: 0.92),
                            fontSize: 13.5,
                            height: 1.45,
                          ),
                        ),
                      ),
                      IconButton(
                        icon: const Icon(Icons.close, size: 16),
                        color: Colors.white54,
                        padding: EdgeInsets.zero,
                        constraints: const BoxConstraints(),
                        onPressed: () {
                          ref.read(homeAssistantProvider.notifier).clear();
                        },
                      ),
                    ],
                  ),
                  if (state.action != null) ...[
                    const SizedBox(height: 12),
                    Align(
                      alignment: Alignment.centerRight,
                      child: ElevatedButton.icon(
                        icon: Icon(
                          _getIconForAction(state.action!),
                          size: 16,
                          color: Colors.white,
                        ),
                        label: Text(
                          _getLabelForAction(state.action!),
                          style: const TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: Colors.white,
                          ),
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: _purple600,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        onPressed: () {
                          ref
                              .read(homeAssistantProvider.notifier)
                              .executeAction(context);
                        },
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  IconData _getIconForAction(String action) {
    switch (action) {
      case 'OPEN_MEMORY':
        return Icons.photo_library_outlined;
      case 'CREATE_MEMORY':
        return Icons.add_photo_alternate_outlined;
      case 'OPEN_VAULT':
      case 'CREATE_VAULT':
        return Icons.folder_shared_outlined;
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

  String _getLabelForAction(String action) {
    switch (action) {
      case 'OPEN_MEMORY':
        return 'Open Memory';
      case 'CREATE_MEMORY':
        return 'Create Memory';
      case 'OPEN_VAULT':
        return 'Open Room';
      case 'CREATE_VAULT':
        return 'Create Room';
      case 'WATCH_VIDEO_JOB':
        return 'View Generation';
      case 'OPEN_VIDEO':
        return 'Play Reel';
      case 'OPEN_AI_DIRECTOR':
        return 'Open AI Director';
      case 'SHOW_READINESS':
        return 'Check Details';
      default:
        return 'View';
    }
  }
}
