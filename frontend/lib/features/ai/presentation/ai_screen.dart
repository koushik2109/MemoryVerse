import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/widgets/media.dart';
import 'package:memory_verse/features/memories/presentation/video_player_screen.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;

class AiScreen extends ConsumerStatefulWidget {
  const AiScreen({super.key});

  @override
  ConsumerState<AiScreen> createState() => _AiScreenState();
}

class _AiScreenState extends ConsumerState<AiScreen> {
  final _msgController = TextEditingController();
  final _scrollController = ScrollController();
  final List<AiMessageModel> _messages = [];
  String? _currentConvoId;
  bool _sending = false;

  static const _suggestions = [
    'Summarize my last trip',
    'Find happy memories',
    'Show memories from 2024',
    'Create a highlight reel',
  ];

  @override
  void dispose() {
    _msgController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final text = _msgController.text.trim();
    if (text.isEmpty || _sending) return;

    setState(() {
      _sending = true;
      _messages.add(
        AiMessageModel(
          id: 'local-${DateTime.now().millisecondsSinceEpoch}',
          conversationId: _currentConvoId ?? '',
          role: 'user',
          content: text,
          createdAt: DateTime.now(),
        ),
      );
    });
    _msgController.clear();
    _scrollToBottom();

    try {
      final repo = ref.read(aiRepositoryProvider);
      final response = await repo.chat(
        conversationId: _currentConvoId,
        message: text,
      );
      if (mounted) {
        final asstMsg = response['assistant_message'] as Map<String, dynamic>?;
        final asstModel = asstMsg != null
            ? AiMessageModel.fromJson(asstMsg)
            : AiMessageModel(
                id: response['id'] as String? ?? 'resp-${DateTime.now().millisecondsSinceEpoch}',
                conversationId: response['conversation_id'] as String? ?? '',
                role: 'assistant',
                content: response['content'] as String? ?? 'I am here to help you explore your memories!',
                createdAt: DateTime.now(),
              );

        setState(() {
          _currentConvoId = response['conversation_id'] as String?;
          _messages.add(asstModel);
          _sending = false;
        });
        _scrollToBottom();
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _messages.add(
            AiMessageModel(
              id: 'err-${DateTime.now().millisecondsSinceEpoch}',
              conversationId: _currentConvoId ?? '',
              role: 'assistant',
              content: 'Sorry, I couldn\'t process that right now. Please check backend logs or connection.',
              createdAt: DateTime.now(),
            ),
          );
          _sending = false;
        });
      }
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: AppMotion.normal,
          curve: AppMotion.curve,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final profileAsync = ref.watch(userProfileProvider);

    return Scaffold(
      backgroundColor: c.bg,
      appBar: AppBar(
        backgroundColor: c.bg,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded),
          onPressed: () => context.pop(),
        ),
        title: const Text('AI Assistant'),
      ),
      body: Column(
        children: [
          // Messages
          Expanded(
            child: _messages.isEmpty
                ? _WelcomeView(
                    c: c,
                    name:
                        profileAsync.valueOrNull?.fullName?.split(' ').first ??
                        'there',
                    onSuggestion: (s) {
                      _msgController.text = s;
                      _send();
                    },
                  )
                : ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.all(AppSpacing.s20),
                    itemCount: _messages.length + (_sending ? 1 : 0),
                    itemBuilder: (_, i) {
                      if (i == _messages.length && _sending) {
                        return _TypingIndicator(c: c);
                      }
                      return _MessageBubble(message: _messages[i]);
                    },
                  ),
          ),

          // Input bar
          Container(
            padding: EdgeInsets.fromLTRB(
              AppSpacing.s16,
              AppSpacing.s12,
              AppSpacing.s12,
              AppSpacing.s12 + MediaQuery.of(context).padding.bottom,
            ),
            decoration: BoxDecoration(
              color: c.surface,
              border: Border(top: BorderSide(color: c.border)),
            ),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _msgController,
                    style: Theme.of(context).textTheme.bodyMedium,
                    decoration: InputDecoration(
                      hintText: 'Ask anything...',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(AppRadii.xl),
                        borderSide: BorderSide(color: c.border),
                      ),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(AppRadii.xl),
                        borderSide: BorderSide(color: c.border),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(AppRadii.xl),
                        borderSide: BorderSide(color: c.primary),
                      ),
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: AppSpacing.s16,
                        vertical: AppSpacing.s12,
                      ),
                      filled: true,
                      fillColor: c.surfaceElevated,
                    ),
                    textInputAction: TextInputAction.send,
                    onSubmitted: (_) => _send(),
                  ),
                ),
                const SizedBox(width: AppSpacing.s8),
                GestureDetector(
                  onTap: _send,
                  child: Container(
                    width: 44,
                    height: 44,
                    decoration: BoxDecoration(
                      color: c.primary,
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons.send_rounded,
                      color: adt.AppColors.onDarkPrimary,
                      size: 20,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _WelcomeView extends StatelessWidget {
  const _WelcomeView({
    required this.c,
    required this.name,
    required this.onSuggestion,
  });
  final AppColors c;
  final String name;
  final ValueChanged<String> onSuggestion;

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(AppSpacing.s24),
      child: Column(
        children: [
          const SizedBox(height: AppSpacing.s40),
          // AI Avatar
          Container(
            width: 64,
            height: 64,
            decoration: BoxDecoration(
              color: c.surfaceElevated,
              shape: BoxShape.circle,
            ),
            child: Icon(Icons.auto_awesome_rounded, size: 30, color: c.primary),
          ),
          const SizedBox(height: AppSpacing.s20),
          Text(
            'Hi $name! 👋',
            style: Theme.of(context).textTheme.headlineMedium,
          ),
          const SizedBox(height: AppSpacing.s8),
          Text(
            'How can I help you with your memories today?',
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.bodyMedium
                ?.copyWith(color: c.textMuted),
          ),
          const SizedBox(height: AppSpacing.s32),
          ..._AiScreenState._suggestions.map(
            (s) => Padding(
              padding: const EdgeInsets.only(bottom: AppSpacing.s10),
              child: GestureDetector(
                onTap: () => onSuggestion(s),
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(
                    horizontal: AppSpacing.s16,
                    vertical: AppSpacing.s14,
                  ),
                  decoration: BoxDecoration(
                    color: c.surface,
                    borderRadius: BorderRadius.circular(AppRadii.md),
                    border: Border.all(color: c.border),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        Icons.auto_awesome_outlined,
                        size: 18,
                        color: c.primary,
                      ),
                      const SizedBox(width: AppSpacing.s12),
                      Expanded(
                        child: Text(
                          s,
                          style: Theme.of(context).textTheme.bodyMedium,
                        ),
                      ),
                      Icon(
                        Icons.arrow_forward_ios_rounded,
                        size: 14,
                        color: c.textMuted,
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _MessageBubble extends StatelessWidget {
  const _MessageBubble({required this.message});
  final AiMessageModel message;

  String _cleanContent(String content) {
    // Strip markdown table rows if any (e.g. | AR... | ... | and |---|---|)
    final lines = content.split('\n');
    final cleaned = <String>[];
    for (final l in lines) {
      final trimmed = l.trim();
      if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
        continue;
      }
      cleaned.add(l);
    }
    return cleaned.join('\n').trim();
  }

  void _showImagePreview(BuildContext context, MediaModel media) {
    showDialog(
      context: context,
      barrierColor: Colors.black87,
      builder: (ctx) => Dialog(
        backgroundColor: Colors.transparent,
        insetPadding: const EdgeInsets.all(AppSpacing.s16),
        child: Stack(
          alignment: Alignment.center,
          children: [
            InteractiveViewer(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(AppRadii.lg),
                child: AppNetworkImage(
                  imageUrl: media.url,
                  fit: BoxFit.contain,
                ),
              ),
            ),
            Positioned(
              top: 8,
              right: 8,
              child: IconButton(
                icon: const Icon(Icons.close_rounded, color: Colors.white),
                onPressed: () => Navigator.of(ctx).pop(),
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
    final isUser = message.isUser;
    final cleanText = _cleanContent(message.content);
    final hasMedia = !isUser && message.relatedMedia.isNotEmpty;

    return Padding(
      padding: const EdgeInsets.only(bottom: AppSpacing.s12),
      child: Align(
        alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
        child: Container(
          constraints: BoxConstraints(
            maxWidth: MediaQuery.of(context).size.width * 0.86,
          ),
          padding: const EdgeInsets.symmetric(
            horizontal: AppSpacing.s16,
            vertical: AppSpacing.s12,
          ),
          decoration: BoxDecoration(
            color: isUser ? c.primary : c.surface,
            borderRadius: BorderRadius.only(
              topLeft: const Radius.circular(AppRadii.lg),
              topRight: const Radius.circular(AppRadii.lg),
              bottomLeft: Radius.circular(isUser ? AppRadii.lg : AppSpacing.s4),
              bottomRight: Radius.circular(
                isUser ? AppSpacing.s4 : AppRadii.lg,
              ),
            ),
            border: isUser ? null : Border.all(color: c.border),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (cleanText.isNotEmpty)
                Text(
                  cleanText,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: isUser ? c.primaryInverse : c.text,
                        height: 1.5,
                        fontWeight: isUser ? FontWeight.w500 : FontWeight.w400,
                      ),
                ),
              if (hasMedia) ...[
                if (cleanText.isNotEmpty) const SizedBox(height: AppSpacing.s12),
                _MediaGalleryAttachment(
                  mediaList: message.relatedMedia,
                  onTapMedia: (media) {
                    if (media.isVideo) {
                      VideoPlayerScreen.open(context, media);
                    } else {
                      _showImagePreview(context, media);
                    }
                  },
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _MediaGalleryAttachment extends StatelessWidget {
  const _MediaGalleryAttachment({
    required this.mediaList,
    required this.onTapMedia,
  });

  final List<MediaModel> mediaList;
  final ValueChanged<MediaModel> onTapMedia;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Gallery Header Badge
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration(
            color: c.surfaceElevated,
            borderRadius: BorderRadius.circular(AppRadii.sm),
            border: Border.all(color: c.border.withValues(alpha: 0.5)),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.photo_library_rounded, size: 13, color: c.primary),
              const SizedBox(width: 5),
              Text(
                '${mediaList.length} ${mediaList.length == 1 ? "memory photo" : "memory photos & videos"}',
                style: TextStyle(
                  color: c.textMuted,
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: AppSpacing.s8),

        // Horizontal media slider
        SizedBox(
          height: 120,
          child: ListView.separated(
            scrollDirection: Axis.horizontal,
            itemCount: mediaList.length,
            separatorBuilder: (_, __) => const SizedBox(width: AppSpacing.s8),
            itemBuilder: (context, idx) {
              final item = mediaList[idx];
              final imgUrl = item.thumbnailUrl ?? item.url;

              return GestureDetector(
                onTap: () => onTapMedia(item),
                child: Container(
                  width: 120,
                  decoration: BoxDecoration(
                    color: c.surfaceElevated,
                    borderRadius: BorderRadius.circular(AppRadii.md),
                    border: Border.all(color: c.border),
                  ),
                  clipBehavior: Clip.antiAlias,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      AppNetworkImage(
                        imageUrl: imgUrl,
                        fit: BoxFit.cover,
                      ),
                      if (item.isVideo)
                        Container(
                          color: Colors.black26,
                          child: const Center(
                            child: Icon(
                              Icons.play_circle_fill_rounded,
                              color: Colors.white,
                              size: 32,
                            ),
                          ),
                        ),
                      // Bottom filename banner
                      Positioned(
                        left: 0,
                        right: 0,
                        bottom: 0,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 4,
                            vertical: 2,
                          ),
                          decoration: const BoxDecoration(
                            gradient: LinearGradient(
                              begin: Alignment.bottomCenter,
                              end: Alignment.topCenter,
                              colors: [Colors.black87, Colors.transparent],
                            ),
                          ),
                          child: Text(
                            item.filename,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 10,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}

class _TypingIndicator extends StatelessWidget {
  const _TypingIndicator({required this.c});
  final AppColors c;

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: AppSpacing.s12),
        padding: const EdgeInsets.all(AppSpacing.s16),
        decoration: BoxDecoration(
          color: c.surface,
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(AppRadii.lg),
            topRight: Radius.circular(AppRadii.lg),
            bottomRight: Radius.circular(AppRadii.lg),
            bottomLeft: Radius.circular(AppSpacing.s4),
          ),
          border: Border.all(color: c.border),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: List.generate(
            3,
            (i) => Padding(
              padding: EdgeInsets.only(right: i < 2 ? 4 : 0),
              child: _Dot(delay: i * 200, c: c),
            ),
          ),
        ),
      ),
    );
  }
}

class _Dot extends StatefulWidget {
  const _Dot({required this.delay, required this.c});
  final int delay;
  final AppColors c;

  @override
  State<_Dot> createState() => _DotState();
}

class _DotState extends State<_Dot> with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    )..repeat(reverse: true);
    Future.delayed(Duration(milliseconds: widget.delay), () {
      if (mounted) _ctrl.forward();
    });
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _ctrl,
      builder: (_, __) => Container(
        width: 8,
        height: 8,
        decoration: BoxDecoration(
          color: widget.c.textMuted.withValues(alpha: 0.3 + 0.5 * _ctrl.value),
          shape: BoxShape.circle,
        ),
      ),
    );
  }
}
