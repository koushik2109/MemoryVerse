import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/features/memories/presentation/video_player_screen.dart';
import 'package:share_plus/share_plus.dart';

class MemoryMediaViewer extends ConsumerStatefulWidget {
  final MemoryModel memory;
  final int initialIndex;

  const MemoryMediaViewer({
    super.key,
    required this.memory,
    this.initialIndex = 0,
  });

  static void open(BuildContext context, MemoryModel memory, int initialIndex) {
    Navigator.push(
      context,
      PageRouteBuilder(
        pageBuilder: (context, animation, secondaryAnimation) => MemoryMediaViewer(memory: memory, initialIndex: initialIndex),
        transitionsBuilder: (context, animation, secondaryAnimation, child) {
          return FadeTransition(opacity: animation, child: child);
        },
      ),
    );
  }

  @override
  ConsumerState<MemoryMediaViewer> createState() => _MemoryMediaViewerState();
}

class _MemoryMediaViewerState extends ConsumerState<MemoryMediaViewer> {
  late PageController _pageController;
  late int _currentIndex;
  bool _showControls = true;

  @override
  void initState() {
    super.initState();
    _currentIndex = widget.initialIndex;
    _pageController = PageController(initialPage: _currentIndex);
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  void _toggleControls() {
    setState(() => _showControls = !_showControls);
  }

  Future<void> _deleteCurrentMedia() async {
    final media = widget.memory.media[_currentIndex];
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: context.colors.surface,
        title: Text('Delete Media?', style: TextStyle(color: context.colors.text)),
        content: Text('Are you sure you want to delete this item?', style: TextStyle(color: context.colors.textMuted)),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: Text('Cancel', style: TextStyle(color: context.colors.textMuted))),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text('Delete', style: TextStyle(color: context.colors.error)),
          ),
        ],
      ),
    );

    if (confirm == true && mounted) {
      try {
        await ref.read(mediaRepositoryProvider).deleteMedia(media.id);
        ref.invalidate(memoryDetailProvider(widget.memory.id));
        if (mounted) {
          if (widget.memory.media.length == 1) {
            Navigator.pop(context); // Close viewer if it was the last item
          } else {
            // UI will rebuild naturally once provider updates, but we'll stay open.
            ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Deleted successfully')));
          }
        }
      } catch (e) {
        if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to delete: $e')));
      }
    }
  }

  Future<void> _setAsCover() async {
    final media = widget.memory.media[_currentIndex];
    if (media.mediaType != 'image') {
       ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Cover must be an image.')));
       return;
    }
    
    try {
      await ref.read(memoryRepositoryProvider).updateMemory(widget.memory.id, coverMediaId: media.id);
      ref.invalidate(memoryDetailProvider(widget.memory.id));
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Cover updated')));
    } catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to update cover: $e')));
    }
  }

  void _shareCurrentMedia() {
    final media = widget.memory.media[_currentIndex];
    Share.share('Check out this moment: ${media.url}');
  }

  @override
  Widget build(BuildContext context) {
    // If the memory media was modified and the index is now out of bounds
    if (_currentIndex >= widget.memory.media.length && widget.memory.media.isNotEmpty) {
      _currentIndex = widget.memory.media.length - 1;
    } else if (widget.memory.media.isEmpty) {
      // In case we deleted the last one and didn't pop yet
      return Scaffold(
        backgroundColor: Colors.black,
        appBar: AppBar(backgroundColor: Colors.transparent, iconTheme: const IconThemeData(color: Colors.white)),
        body: const Center(child: Text('No media', style: TextStyle(color: Colors.white))),
      );
    }

    final mediaList = widget.memory.media;
    final currentMedia = mediaList[_currentIndex];

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          GestureDetector(
            onTap: _toggleControls,
            child: PageView.builder(
              controller: _pageController,
              itemCount: mediaList.length,
              onPageChanged: (idx) {
                setState(() => _currentIndex = idx);
              },
              itemBuilder: (context, index) {
                final media = mediaList[index];
                if (media.mediaType == 'video') {
                  // For videos, we could embed a player, but for a simple viewer,
                  // we can show the thumbnail and a big play button that opens full video player,
                  // OR embed the video player directly.
                  // For simplicity in a PageView, showing a thumbnail + play button is safer for performance.
                  return Stack(
                    fit: StackFit.expand,
                    children: [
                      Image.network(
                        media.thumbnailUrl ?? media.url,
                        fit: BoxFit.contain,
                        errorBuilder: (_, __, ___) => const Center(child: Icon(Icons.error, color: Colors.white)),
                      ),
                      Center(
                        child: IconButton(
                          iconSize: 72,
                          icon: const Icon(Icons.play_circle_fill, color: Colors.white70),
                          onPressed: () => VideoPlayerScreen.open(context, media),
                        ),
                      ),
                    ],
                  );
                } else {
                  return InteractiveViewer(
                    minScale: 1.0,
                    maxScale: 4.0,
                    child: Image.network(
                      media.url,
                      fit: BoxFit.contain,
                      loadingBuilder: (context, child, progress) {
                        if (progress == null) return child;
                        return const Center(child: CircularProgressIndicator(color: Colors.white24));
                      },
                      errorBuilder: (_, __, ___) => const Center(child: Icon(Icons.error, color: Colors.white)),
                    ),
                  );
                }
              },
            ),
          ),

          // Top App Bar
          AnimatedOpacity(
            opacity: _showControls ? 1.0 : 0.0,
            duration: const Duration(milliseconds: 200),
            child: Align(
              alignment: Alignment.topCenter,
              child: Container(
                padding: EdgeInsets.only(top: MediaQuery.of(context).padding.top),
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Colors.black87, Colors.transparent],
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                  ),
                ),
                child: Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.arrow_back_rounded, color: Colors.white),
                      onPressed: () => Navigator.pop(context),
                    ),
                    Expanded(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            DateFormat('MMMM d, yyyy h:mm a').format(currentMedia.createdAt),
                            style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w600),
                          ),
                          Text(
                            '${_currentIndex + 1} of ${mediaList.length}',
                            style: const TextStyle(color: Colors.white70, fontSize: 11),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // Bottom Actions
          AnimatedOpacity(
            opacity: _showControls ? 1.0 : 0.0,
            duration: const Duration(milliseconds: 200),
            child: Align(
              alignment: Alignment.bottomCenter,
              child: Container(
                padding: EdgeInsets.only(bottom: MediaQuery.of(context).padding.bottom, top: 16),
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Colors.transparent, Colors.black87],
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    IconButton(
                      icon: const Icon(Icons.ios_share, color: Colors.white),
                      onPressed: _shareCurrentMedia,
                    ),
                    if (currentMedia.mediaType == 'image')
                      IconButton(
                        icon: const Icon(Icons.wallpaper, color: Colors.white),
                        tooltip: 'Set as Cover',
                        onPressed: _setAsCover,
                      ),
                    IconButton(
                      icon: const Icon(Icons.delete_outline, color: Colors.redAccent),
                      onPressed: _deleteCurrentMedia,
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
