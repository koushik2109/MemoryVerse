import 'dart:io';
import 'dart:math' as math;
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/providers/ai_director_controller.dart';
import 'package:memory_verse/core/providers/video_job_controller.dart';
import 'package:memory_verse/features/memories/presentation/video_player_screen.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

class AiDirectorScreen extends ConsumerStatefulWidget {
  final MemoryModel memory;
  final String? initialPrompt;

  const AiDirectorScreen({
    super.key,
    required this.memory,
    this.initialPrompt,
  });

  static void open(BuildContext context, MemoryModel memory, {String? initialPrompt}) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => AiDirectorScreen(
          memory: memory,
          initialPrompt: initialPrompt,
        ),
      ),
    );
  }

  @override
  ConsumerState<AiDirectorScreen> createState() => _AiDirectorScreenState();
}

class _AiDirectorScreenState extends ConsumerState<AiDirectorScreen> {
  late final TextEditingController _textController;
  late final ScrollController _scrollController;
  final FocusNode _inputFocusNode = FocusNode();
  final List<XFile> _attachedMedia = [];
  bool _isDownloading = false;
  String? _downloadedVideoPath;

  @override
  void initState() {
    super.initState();
    _textController = TextEditingController(text: widget.initialPrompt ?? '');
    _scrollController = ScrollController();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(aiDirectorActiveProvider.notifier).state = true;
      ref.read(videoJobControllerProvider.notifier).setDirectorOpen(true);
      ref.read(aiDirectorControllerProvider.notifier).initSession(
            widget.memory.id,
            initialPrompt: widget.initialPrompt,
          );
    });
  }

  @override
  void dispose() {
    ref.read(aiDirectorActiveProvider.notifier).state = false;
    ref.read(videoJobControllerProvider.notifier).setDirectorOpen(false);
    _textController.dispose();
    _scrollController.dispose();
    _inputFocusNode.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent + 200,
          duration: const Duration(milliseconds: 350),
          curve: Curves.easeOutCubic,
        );
      }
    });
  }

  Future<void> _pickMedia() async {
    try {
      final picker = ImagePicker();
      final files = await picker.pickMultiImage();
      if (files.isNotEmpty) {
        setState(() {
          _attachedMedia.addAll(files);
        });
        _scrollToBottom();
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to pick images: $e')),
      );
    }
  }

  Future<String?> _downloadVideo(VideoJobModel job) async {
    if (job.resultUrl == null || _isDownloading) return _downloadedVideoPath;
    setState(() => _isDownloading = true);
    try {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Downloading video for offline viewing & sharing...'),
          duration: Duration(seconds: 2),
        ),
      );
      final dir = await getApplicationDocumentsDirectory();
      final sanitizedTitle = widget.memory.title.replaceAll(RegExp(r'[^a-zA-Z0-9]'), '_');
      final filename = 'MemoryVerse_${sanitizedTitle}_${job.id.substring(0, math.min(6, job.id.length))}.mp4';
      final savePath = '${dir.path}/$filename';

      final dio = Dio();
      await dio.download(job.resultUrl!, savePath);
      _downloadedVideoPath = savePath;

      String? publicSavedFolder;
      try {
        if (Platform.isAndroid) {
          final publicMoviesDir = Directory('/storage/emulated/0/Movies');
          if (publicMoviesDir.existsSync()) {
            final publicFile = File('${publicMoviesDir.path}/$filename');
            await File(savePath).copy(publicFile.path);
            publicSavedFolder = 'Movies';
          } else {
            final extMoviesDirs = await getExternalStorageDirectories(type: StorageDirectory.movies);
            if (extMoviesDirs != null && extMoviesDirs.isNotEmpty) {
              final publicFile = File('${extMoviesDirs.first.path}/$filename');
              await File(savePath).copy(publicFile.path);
              publicSavedFolder = 'Movies';
            }
          }
        } else {
          final dlDir = await getDownloadsDirectory();
          if (dlDir != null) {
            final publicFile = File('${dlDir.path}/$filename');
            await File(savePath).copy(publicFile.path);
            publicSavedFolder = 'Downloads';
          }
        }
      } catch (copyErr) {
        debugPrint('Could not copy to public folder: $copyErr');
      }

      if (!mounted) return _downloadedVideoPath;
      setState(() => _isDownloading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(publicSavedFolder != null
              ? 'Video saved in app and in offline $publicSavedFolder folder: $filename'
              : 'Video saved locally: $filename'),
          backgroundColor: const Color(0xFF4CAF50),
          duration: const Duration(seconds: 4),
          action: SnackBarAction(
            label: 'Share',
            textColor: Colors.white,
            onPressed: () {
              Share.shareXFiles([XFile(savePath)], text: 'Watch my MemoryVerse movie: ${widget.memory.title}');
            },
          ),
        ),
      );
      return savePath;
    } catch (e) {
      if (!mounted) return null;
      setState(() => _isDownloading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Download failed: $e'),
          backgroundColor: Colors.red,
        ),
      );
      return null;
    }
  }

  void _submitMessage(String text) {
    if (text.trim().isEmpty && _attachedMedia.isEmpty) return;
    String submission = text.trim();
    if (_attachedMedia.isNotEmpty) {
      final mediaNote = ' [Attached ${_attachedMedia.length} photo${_attachedMedia.length > 1 ? 's' : ''}]';
      submission = submission.isEmpty ? mediaNote.trim() : '$submission$mediaNote';
      setState(() {
        _attachedMedia.clear();
      });
    }
    _textController.clear();
    ref.read(aiDirectorControllerProvider.notifier).sendMessage(submission);
    _scrollToBottom();
  }

  void _onSuggestionTap(String suggestion) {
    if (suggestion.toLowerCase().contains('generate')) {
      ref.read(aiDirectorControllerProvider.notifier).approveAndGenerate();
    } else {
      _submitMessage(suggestion);
    }
  }

  void _playResultVideo(VideoJobModel job) {
    if (job.resultUrl != null) {
      final mediaItem = MediaModel(
        id: job.resultMediaId ?? job.id,
        memoryId: widget.memory.id,
        ownerId: job.userId ?? '',
        filename: 'memory_movie_${job.id}.mp4',
        storagePath: 'reels/${job.id}.mp4',
        url: job.resultUrl!,
        thumbnailUrl: null,
        mediaType: 'video',
        fileSize: 0,
        createdAt: DateTime.now(),
      );
      VideoPlayerScreen.open(context, mediaItem);
    }
  }

  @override
  Widget build(BuildContext context) {
    ref.listen<AiDirectorState>(aiDirectorControllerProvider, (prev, next) {
      if (prev?.session?.messages.length != next.session?.messages.length ||
          prev?.activeJobId != next.activeJobId ||
          prev?.isGenerating != next.isGenerating) {
        _scrollToBottom();
      }
    });

    ref.listen<VideoJobUIState>(videoJobControllerProvider, (prev, next) {
      if (prev?.job?.overallProgress != next.job?.overallProgress ||
          prev?.job?.status != next.job?.status ||
          prev?.job?.currentTask != next.job?.currentTask) {
        _scrollToBottom();
      }
    });

    final directorState = ref.watch(aiDirectorControllerProvider);
    final jobState = ref.watch(videoJobControllerProvider);
    final session = directorState.session;
    final activeJob = jobState.job;

    // Check if user has entered conversation mode (more than initial greeting)
    final hasUserMessages = (session?.messages.length ?? 0) > 1;

    return Scaffold(
      backgroundColor: const Color(0xFF0F0C15),
      body: Stack(
        children: [
          // ── 1. Ambient Gradient Mesh (Matches image.png) ───────────
          Positioned.fill(
            child: Container(
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topRight,
                  end: Alignment.bottomLeft,
                  colors: [
                    Color(0xFFE8557D), // Vibrant rose / coral top-right
                    Color(0xFFE05F86), // Warm pink
                    Color(0xFF8B3B66), // Mid dusky violet
                    Color(0xFF38203D), // Deep plum
                    Color(0xFF16121E), // Slate charcoal bottom
                  ],
                  stops: [0.0, 0.22, 0.50, 0.78, 1.0],
                ),
              ),
            ),
          ),

          // ── 2. Subtle Radial Glow Orbs ─────────────────────────────
          Positioned(
            top: -60,
            right: -60,
            child: Container(
              width: 300,
              height: 300,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  colors: [
                    const Color(0xFFFFA5B8).withValues(alpha: 0.55),
                    Colors.transparent,
                  ],
                ),
              ),
            ),
          ),

          // ── 3. Main Content Area ───────────────────────────────────
          SafeArea(
            child: Column(
              children: [
                // Top Navigation Bar
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
                  child: Row(
                    children: [
                      GestureDetector(
                        onTap: () => Navigator.pop(context),
                        child: Container(
                          width: 42,
                          height: 42,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: Colors.black.withValues(alpha: 0.28),
                            border: Border.all(
                              color: Colors.white.withValues(alpha: 0.12),
                            ),
                          ),
                          child: const Icon(
                            Icons.chevron_left_rounded,
                            color: Colors.white,
                            size: 28,
                          ),
                        ),
                      ),
                      const Spacer(),
                      if (session != null)
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                          decoration: BoxDecoration(
                            color: Colors.black.withValues(alpha: 0.25),
                            borderRadius: BorderRadius.circular(16),
                            border: Border.all(color: Colors.white.withValues(alpha: 0.1)),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(Icons.movie_outlined, size: 14, color: Colors.white70),
                              const SizedBox(width: 6),
                              Text(
                                '${session.specification.aspectRatio} · ${session.specification.durationTargetSeconds}s',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 12,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ],
                          ),
                        ),
                    ],
                  ),
                ),

                // Body: Initial Showcase (image.png) OR Conversational Stream
                Expanded(
                  child: hasUserMessages
                      ? _buildConversationView(session!, activeJob)
                      : _buildInitialShowcaseView(session),
                ),

                // Bottom Input Area (Illuminated Pill from image.png)
                _buildInputBar(directorState.isLoading || directorState.isGenerating),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── Initial State Layout (Strict reproduction of image.png) ─────────────────
  Widget _buildInitialShowcaseView(AISessionModel? session) {
    final suggestions = session?.suggestions ?? [
      'Time at home with friends',
      'Spring flowers to remember',
      'Joyful holidays, with epic music',
    ];

    return SingleChildScrollView(
      controller: _scrollController,
      physics: const BouncingScrollPhysics(),
      padding: const EdgeInsets.symmetric(horizontal: 24.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          const SizedBox(height: 48),

          // Hero Title (Matches image.png typography & hierarchy)
          const Text(
            'Create a',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.white,
              fontSize: 34,
              fontWeight: FontWeight.w800,
              letterSpacing: -0.5,
              height: 1.15,
            ),
          ),
          const Text(
            'Memory Movie',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.white,
              fontSize: 38,
              fontWeight: FontWeight.w800,
              letterSpacing: -0.5,
              height: 1.15,
            ),
          ),
          const SizedBox(height: 14),

          // Subtitle
          Text(
            'WITH JUST A DESCRIPTION',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.85),
              fontSize: 13,
              fontWeight: FontWeight.w600,
              letterSpacing: 2.2,
            ),
          ),

          const SizedBox(height: 64),

          // Contextual Suggestion Pills (Stacked vertically, left-aligned pills)
          Align(
            alignment: Alignment.centerLeft,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: suggestions.map((text) {
                return Padding(
                  padding: const EdgeInsets.only(bottom: 12.0),
                  child: _buildSuggestionPill(text),
                );
              }).toList(),
            ),
          ),

          const SizedBox(height: 24),
        ],
      ),
    );
  }

  Widget _buildSuggestionPill(String text) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () => _onSuggestionTap(text),
        borderRadius: BorderRadius.circular(24),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 13),
          decoration: BoxDecoration(
            color: const Color(0xFF1E1624).withValues(alpha: 0.72),
            borderRadius: BorderRadius.circular(24),
            border: Border.all(
              color: Colors.white.withValues(alpha: 0.12),
              width: 1,
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.18),
                blurRadius: 8,
                offset: const Offset(0, 3),
              ),
            ],
          ),
          child: Text(
            text,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 15,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
      ),
    );
  }

  // ── Conversational View Layout ──────────────────────────────────────────────
  Widget _buildConversationView(AISessionModel session, VideoJobModel? activeJob) {
    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 12.0),
      itemCount: session.messages.length + (activeJob != null ? 1 : 0),
      itemBuilder: (context, index) {
        if (activeJob != null && index == session.messages.length) {
          return _buildInlineJobCard(activeJob);
        }

        final msg = session.messages[index];
        final isUser = msg.role == 'user';
        final lastAssistantIndex = session.messages.lastIndexWhere((m) => m.role != 'user');
        final isLatestAssistant = index == lastAssistantIndex;

        return Column(
          crossAxisAlignment:
              isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 6.0),
              child: isUser
                  ? _buildUserBubble(msg.content)
                  : _buildAssistantBubble(msg, session.specification),
            ),
            if (!isUser && isLatestAssistant && msg.suggestions.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 6.0, bottom: 8.0),
                child: Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: msg.suggestions.map((s) {
                    final isGen = s.toLowerCase().contains('generate');
                    return ActionChip(
                      label: Text(s),
                      backgroundColor: isGen
                          ? const Color(0xFFE8557D)
                          : const Color(0xFF261D2E).withValues(alpha: 0.8),
                      labelStyle: TextStyle(
                        color: Colors.white,
                        fontSize: 13,
                        fontWeight: isGen ? FontWeight.w700 : FontWeight.w500,
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20),
                        side: BorderSide(
                          color: isGen
                              ? const Color(0xFFFFA5B8)
                              : Colors.white.withValues(alpha: 0.15),
                        ),
                      ),
                      onPressed: () => _onSuggestionTap(s),
                    );
                  }).toList(),
                ),
              ),
          ],
        );
      },
    );
  }

  Widget _buildUserBubble(String text) {
    return Container(
      constraints: BoxConstraints(
        maxWidth: MediaQuery.of(context).size.width * 0.78,
      ),
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFFE8557D).withValues(alpha: 0.85),
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(20),
          topRight: Radius.circular(4),
          bottomLeft: Radius.circular(20),
          bottomRight: Radius.circular(20),
        ),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFFE8557D).withValues(alpha: 0.3),
            blurRadius: 10,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Text(
        text,
        style: const TextStyle(color: Colors.white, fontSize: 15, height: 1.3),
      ),
    );
  }

  Widget _buildAssistantBubble(AIMessageModel msg, VideoSpecificationModel spec) {
    return Container(
      constraints: BoxConstraints(
        maxWidth: MediaQuery.of(context).size.width * 0.84,
      ),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1B1522).withValues(alpha: 0.85),
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(4),
          topRight: Radius.circular(20),
          bottomLeft: Radius.circular(20),
          bottomRight: Radius.circular(20),
        ),
        border: Border.all(color: Colors.white.withValues(alpha: 0.12)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.25),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _buildAiRosetteIcon(size: 18),
              const SizedBox(width: 8),
              const Text(
                'AI Memory Director',
                style: TextStyle(
                  color: Color(0xFFFFA5B8),
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.5,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            msg.content,
            style: const TextStyle(color: Colors.white, fontSize: 14.5, height: 1.4),
          ),
        ],
      ),
    );
  }

  // ── Inline Video Progress / Result Card (Section 13 & 14) ───────────────────
  Widget _buildInlineJobCard(VideoJobModel job) {
    final isDone = job.isCompleted;
    final isFailed = job.isFailed;

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 12),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1322).withValues(alpha: 0.92),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(
          color: isDone
              ? const Color(0xFF4CAF50).withValues(alpha: 0.6)
              : (isFailed
                  ? const Color(0xFFE53935).withValues(alpha: 0.6)
                  : const Color(0xFFE8557D).withValues(alpha: 0.6)),
        ),
        boxShadow: [
          BoxShadow(
            color: (isDone ? const Color(0xFF4CAF50) : const Color(0xFFE8557D))
                .withValues(alpha: 0.25),
            blurRadius: 14,
            spreadRadius: 1,
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                isDone
                    ? Icons.check_circle_outline
                    : (isFailed ? Icons.error_outline : Icons.auto_awesome),
                color: isDone
                    ? const Color(0xFF81C784)
                    : (isFailed ? const Color(0xFFE57373) : const Color(0xFFFFA5B8)),
                size: 20,
              ),
              const SizedBox(width: 8),
              Text(
                isDone
                    ? 'Your memory movie is ready'
                    : (isFailed ? 'Creation paused' : 'Creating your memory movie...'),
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                  fontSize: 15,
                ),
              ),
              const Spacer(),
              if (!isDone && !isFailed)
                Text(
                  '${job.overallProgress}%',
                  style: const TextStyle(
                    color: Color(0xFFFFA5B8),
                    fontWeight: FontWeight.w800,
                    fontSize: 14,
                  ),
                ),
            ],
          ),
          const SizedBox(height: 12),

          if (!isDone && !isFailed) ...[
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: LinearProgressIndicator(
                value: job.overallProgress / 100.0,
                minHeight: 6,
                backgroundColor: Colors.white.withValues(alpha: 0.12),
                valueColor: const AlwaysStoppedAnimation(Color(0xFFE8557D)),
              ),
            ),
            const SizedBox(height: 10),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Text(
                    job.currentTask,
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.75),
                      fontSize: 12,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                if (job.estimatedRemainingSeconds != null)
                  Text(
                    '~${job.estimatedRemainingSeconds}s remaining',
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.55),
                      fontSize: 11,
                    ),
                  ),
              ],
            ),
          ],

          if (isDone) ...[
            const SizedBox(height: 8),
            Container(
              height: 140,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.black,
                borderRadius: BorderRadius.circular(14),
                image: widget.memory.media.isNotEmpty && widget.memory.media.first.thumbnailUrl != null
                    ? DecorationImage(
                        image: NetworkImage(widget.memory.media.first.thumbnailUrl!),
                        fit: BoxFit.cover,
                        colorFilter: ColorFilter.mode(
                          Colors.black.withValues(alpha: 0.4),
                          BlendMode.darken,
                        ),
                      )
                    : null,
              ),
              child: Center(
                child: GestureDetector(
                  onTap: () => _playResultVideo(job),
                  child: Container(
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: const Color(0xFFE8557D),
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: const Color(0xFFE8557D).withValues(alpha: 0.6),
                          blurRadius: 16,
                        ),
                      ],
                    ),
                    child: const Icon(Icons.play_arrow_rounded, color: Colors.white, size: 36),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 14),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    icon: const Icon(Icons.play_arrow_rounded, size: 20),
                    label: const Text('Play Movie'),
                    style: FilledButton.styleFrom(
                      backgroundColor: const Color(0xFFE8557D),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                    ),
                    onPressed: () => _playResultVideo(job),
                  ),
                ),
                const SizedBox(width: 8),
                IconButton.filledTonal(
                  icon: _isDownloading
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                        )
                      : const Icon(Icons.download_rounded, size: 20),
                  tooltip: 'Download video offline',
                  onPressed: _isDownloading ? null : () => _downloadVideo(job),
                ),
                const SizedBox(width: 8),
                IconButton.filledTonal(
                  icon: const Icon(Icons.share_rounded, size: 20),
                  tooltip: 'Share video file with others',
                  onPressed: () async {
                    if (_downloadedVideoPath != null && File(_downloadedVideoPath!).existsSync()) {
                      await Share.shareXFiles(
                        [XFile(_downloadedVideoPath!)],
                        text: 'Watch my MemoryVerse movie: ${widget.memory.title}',
                      );
                    } else if (job.resultUrl != null) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text('Preparing video file to share...'),
                          duration: Duration(seconds: 2),
                        ),
                      );
                      final downloadedPath = await _downloadVideo(job);
                      if (downloadedPath != null && File(downloadedPath).existsSync()) {
                        await Share.shareXFiles(
                          [XFile(downloadedPath)],
                          text: 'Watch my MemoryVerse movie: ${widget.memory.title}',
                        );
                      } else {
                        await Share.share('Watch my MemoryVerse movie: ${job.resultUrl}');
                      }
                    }
                  },
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  // ── Bottom Input Pill (Matches image.png) ──────────────────────────────────
  Widget _buildInputBar(bool disabled) {
    return Container(
      padding: EdgeInsets.only(
        left: 16,
        right: 16,
        bottom: MediaQuery.of(context).viewInsets.bottom + 16,
        top: 8,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Attached media thumbnails
          if (_attachedMedia.isNotEmpty)
            Container(
              height: 64,
              margin: const EdgeInsets.only(bottom: 8),
              child: ListView.separated(
                scrollDirection: Axis.horizontal,
                itemCount: _attachedMedia.length,
                separatorBuilder: (_, __) => const SizedBox(width: 8),
                itemBuilder: (context, idx) {
                  final f = _attachedMedia[idx];
                  return Stack(
                    children: [
                      ClipRRect(
                        borderRadius: BorderRadius.circular(10),
                        child: Image.file(
                          File(f.path),
                          width: 64,
                          height: 64,
                          fit: BoxFit.cover,
                        ),
                      ),
                      Positioned(
                        top: 2,
                        right: 2,
                        child: GestureDetector(
                          onTap: () => setState(() => _attachedMedia.removeAt(idx)),
                          child: Container(
                            padding: const EdgeInsets.all(2),
                            decoration: const BoxDecoration(
                              color: Colors.black87,
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(Icons.close, size: 14, color: Colors.white),
                          ),
                        ),
                      ),
                    ],
                  );
                },
              ),
            ),
          Row(
            children: [
              // + Button beside chat to attach media
              GestureDetector(
                onTap: _pickMedia,
                child: Container(
                  width: 48,
                  height: 48,
                  margin: const EdgeInsets.only(right: 8),
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: const Color(0xFF1E1726).withValues(alpha: 0.95),
                    border: Border.all(
                      color: const Color(0xFFE8557D).withValues(alpha: 0.7),
                      width: 1.2,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: const Color(0xFFE8557D).withValues(alpha: 0.35),
                        blurRadius: 10,
                      ),
                    ],
                  ),
                  child: const Icon(
                    Icons.add_rounded,
                    color: Color(0xFFFFA5B8),
                    size: 26,
                  ),
                ),
              ),
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                  decoration: BoxDecoration(
                    color: const Color(0xFF1E1726).withValues(alpha: 0.92),
                    borderRadius: BorderRadius.circular(32),
                    border: Border.all(
                      color: const Color(0xFFE8557D).withValues(alpha: 0.7),
                      width: 1.2,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: const Color(0xFFE8557D).withValues(alpha: 0.55),
                        blurRadius: 18,
                        spreadRadius: 1,
                      ),
                    ],
                  ),
                  child: Row(
                    children: [
                      _buildAiRosetteIcon(size: 26),
                      const SizedBox(width: 12),
                      Expanded(
                        child: TextField(
                          controller: _textController,
                          focusNode: _inputFocusNode,
                          style: const TextStyle(color: Colors.white, fontSize: 16),
                          cursorColor: const Color(0xFFFFA5B8),
                          decoration: InputDecoration(
                            hintText: 'Describe a memory...',
                            hintStyle: TextStyle(
                              color: Colors.white.withValues(alpha: 0.42),
                              fontSize: 16,
                            ),
                            border: InputBorder.none,
                          ),
                          onSubmitted: _submitMessage,
                        ),
                      ),
                      ValueListenableBuilder<TextEditingValue>(
                        valueListenable: _textController,
                        builder: (context, value, _) {
                          final hasText = value.text.trim().isNotEmpty || _attachedMedia.isNotEmpty;
                          return GestureDetector(
                            onTap: hasText ? () => _submitMessage(value.text) : null,
                            child: Container(
                              width: 34,
                              height: 34,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: hasText
                                    ? const Color(0xFFE8557D)
                                    : Colors.white.withValues(alpha: 0.12),
                              ),
                              child: Icon(
                                Icons.arrow_upward_rounded,
                                color: hasText ? Colors.white : Colors.white38,
                                size: 20,
                              ),
                            ),
                          );
                        },
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  // ── Decorative Geometric AI Rosette Flower (Exact reproduction from image.png) ─
  Widget _buildAiRosetteIcon({double size = 24}) {
    return SizedBox(
      width: size,
      height: size,
      child: CustomPaint(
        painter: _AiRosettePainter(),
      ),
    );
  }
}

class _AiRosettePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width / 2.3;

    final paint = Paint()
      ..color = const Color(0xFFFFA5B8)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.4;

    // Draw 8 intersecting petals forming the flower rosette
    const petalCount = 8;
    for (int i = 0; i < petalCount; i++) {
      final angle = (i * 2 * math.pi) / petalCount;
      final petalCenter = Offset(
        center.dx + (radius * 0.45) * math.cos(angle),
        center.dy + (radius * 0.45) * math.sin(angle),
      );
      canvas.drawCircle(petalCenter, radius * 0.48, paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
