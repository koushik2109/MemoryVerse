import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:dio/dio.dart';
import 'package:path_provider/path_provider.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/api/api_client.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/core/services/media/media_gallery_export_service.dart';
import 'package:memory_verse/core/widgets/media.dart';
import 'package:memory_verse/features/memories/presentation/video_player_screen.dart';
import 'package:share_plus/share_plus.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;

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
        pageBuilder: (context, animation, secondaryAnimation) =>
            MemoryMediaViewer(memory: memory, initialIndex: initialIndex),
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
        title: Text(
          'Delete Media?',
          style: TextStyle(color: context.colors.text),
        ),
        content: Text(
          'Are you sure you want to delete this item?',
          style: TextStyle(color: context.colors.textMuted),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text(
              'Cancel',
              style: TextStyle(color: context.colors.textMuted),
            ),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text(
              'Delete',
              style: TextStyle(color: context.colors.error),
            ),
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
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('Deleted successfully')),
            );
          }
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context)
              .showSnackBar(SnackBar(content: Text('Failed to delete: $e')));
        }
      }
    }
  }

  Future<void> _setAsCover() async {
    final media = widget.memory.media[_currentIndex];
    if (media.mediaType != 'image') {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Cover must be an image.')));
      return;
    }

    try {
      await ref
          .read(memoryRepositoryProvider)
          .updateMemory(widget.memory.id, coverMediaId: media.id);
      ref.invalidate(memoryDetailProvider(widget.memory.id));
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(const SnackBar(content: Text('Cover updated')));
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Failed to update cover: $e')));
      }
    }
  }

  bool _isDownloading = false;

  Future<String?> _downloadCurrentMedia() async {
    final media = widget.memory.media[_currentIndex];
    if (_isDownloading) return null;

    setState(() => _isDownloading = true);
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Downloading media to offline storage...'),
        duration: Duration(seconds: 2),
      ),
    );

    try {
      final dir = await getApplicationDocumentsDirectory();
      final isVideo = media.mediaType == 'video';
      final safeName =
          media.filename.replaceAll(RegExp(r'[^a-zA-Z0-9._-]'), '_');
      final savePath = '${dir.path}/$safeName';

      final downloadUrl = media.url.isNotEmpty
          ? media.url
          : '${ApiClient.baseUrl}/media/stream/${media.id}';

      final dio = Dio();
      await dio.download(downloadUrl, savePath);

      String? publicFolder;
      try {
        if (MediaGalleryExportService.isSupportedPlatform) {
          publicFolder = await MediaGalleryExportService.saveDownloadedFile(
            filePath: savePath,
            fileName: safeName,
            isVideo: isVideo,
          );
        } else {
          final dlDir = await getDownloadsDirectory();
          if (dlDir != null) {
            final publicFile = File('${dlDir.path}/$safeName');
            await File(savePath).copy(publicFile.path);
            publicFolder = 'Downloads';
          }
        }
      } catch (copyErr) {
        debugPrint('Could not copy to public folder: $copyErr');
      }

      if (mounted) {
        setState(() => _isDownloading = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(publicFolder != null
                ? '${isVideo ? "Video" : "Photo"} saved to device $publicFolder folder: $safeName'
                : 'Saved locally: $safeName'),
            backgroundColor: const Color(0xFF4CAF50),
            duration: const Duration(seconds: 4),
            action: SnackBarAction(
              label: 'Share',
              textColor: Colors.white,
              onPressed: _shareCurrentMedia,
            ),
          ),
        );
      }
      return savePath;
    } catch (e) {
      debugPrint("Download media error: $e");
      if (mounted) {
        setState(() => _isDownloading = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Download failed: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
      return null;
    }
  }

  Future<void> _shareCurrentMedia() async {
    final media = widget.memory.media[_currentIndex];
    try {
      final dir = await getApplicationDocumentsDirectory();
      final safeName =
          media.filename.replaceAll(RegExp(r'[^a-zA-Z0-9._-]'), '_');
      final localPath = '${dir.path}/$safeName';

      if (!File(localPath).existsSync()) {
        final downloaded = await _downloadCurrentMedia();
        if (downloaded != null && File(downloaded).existsSync()) {
          await Share.shareXFiles(
            [XFile(downloaded)],
            text: 'Check out this moment from "${widget.memory.title}"',
          );
          return;
        }
      } else {
        await Share.shareXFiles(
          [XFile(localPath)],
          text: 'Check out this moment from "${widget.memory.title}"',
        );
        return;
      }
    } catch (shareErr) {
      debugPrint("Error sharing local file: $shareErr");
    }

    if (media.url.isNotEmpty) {
      await Share.share('Check out this moment: ${media.url}');
    }
  }

  @override
  Widget build(BuildContext context) {
    // If the memory media was modified and the index is now out of bounds
    if (_currentIndex >= widget.memory.media.length &&
        widget.memory.media.isNotEmpty) {
      _currentIndex = widget.memory.media.length - 1;
    } else if (widget.memory.media.isEmpty) {
      // In case we deleted the last one and didn't pop yet
      return Scaffold(
        backgroundColor: adt.AppColors.plum900,
        appBar: AppBar(
          backgroundColor: Colors.transparent,
          iconTheme: const IconThemeData(color: adt.AppColors.onDarkPrimary),
        ),
        body: const Center(
          child: Text('No media',
              style: TextStyle(color: adt.AppColors.onDarkPrimary)),
        ),
      );
    }

    final mediaList = widget.memory.media;
    final currentMedia = mediaList[_currentIndex];

    return Scaffold(
      backgroundColor: adt.AppColors.plum900,
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
                      AppNetworkImage(
                        imageUrl: media.thumbnailUrl ?? media.url,
                        fit: BoxFit.contain,
                        memCacheWidth: 800,
                      ),
                      Center(
                        child: IconButton(
                          iconSize: 72,
                          icon: const Icon(
                            Icons.play_circle_fill,
                            color: adt.AppColors.onDarkSecondary,
                          ),
                          onPressed: () =>
                              VideoPlayerScreen.open(context, media),
                        ),
                      ),
                    ],
                  );
                } else {
                  return InteractiveViewer(
                    minScale: 1.0,
                    maxScale: 4.0,
                    child: AppNetworkImage(
                      imageUrl: media.url,
                      fit: BoxFit.contain,
                      memCacheWidth: 1200,
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
                padding: EdgeInsets.only(
                  top: MediaQuery.of(context).padding.top,
                ),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      adt.AppColors.plum900.withValues(alpha: 0.87),
                      Colors.transparent
                    ],
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                  ),
                ),
                child: Row(
                  children: [
                    IconButton(
                      icon: const Icon(
                        Icons.arrow_back_rounded,
                        color: adt.AppColors.onDarkPrimary,
                      ),
                      onPressed: () => Navigator.pop(context),
                    ),
                    Expanded(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            DateFormat('MMMM d, yyyy h:mm a')
                                .format(currentMedia.createdAt),
                            style: const TextStyle(
                              color: adt.AppColors.onDarkPrimary,
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          Text(
                            '${_currentIndex + 1} of ${mediaList.length}',
                            style: const TextStyle(
                              color: adt.AppColors.onDarkSecondary,
                              fontSize: 11,
                            ),
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
                padding: EdgeInsets.only(
                  bottom: MediaQuery.of(context).padding.bottom,
                  top: 16,
                ),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      Colors.transparent,
                      adt.AppColors.plum900.withValues(alpha: 0.87)
                    ],
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    IconButton(
                      icon: _isDownloading
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: adt.AppColors.onDarkPrimary,
                              ),
                            )
                          : const Icon(Icons.download_rounded,
                              color: adt.AppColors.onDarkPrimary),
                      tooltip: 'Download offline',
                      onPressed: _isDownloading ? null : _downloadCurrentMedia,
                    ),
                    IconButton(
                      icon: const Icon(Icons.ios_share,
                          color: adt.AppColors.onDarkPrimary),
                      tooltip: 'Share',
                      onPressed: _shareCurrentMedia,
                    ),
                    if (currentMedia.mediaType == 'image')
                      IconButton(
                        icon: const Icon(Icons.wallpaper,
                            color: adt.AppColors.onDarkPrimary),
                        tooltip: 'Set as Cover',
                        onPressed: _setAsCover,
                      ),
                    IconButton(
                      icon: const Icon(
                        Icons.delete_outline,
                        color: Colors.redAccent,
                      ),
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
