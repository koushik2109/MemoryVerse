import 'dart:io';
import 'package:flutter/material.dart';
import 'package:chewie/chewie.dart';
import 'package:dio/dio.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:video_player/video_player.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/api/api_client.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;

class VideoPlayerScreen extends StatefulWidget {
  const VideoPlayerScreen({super.key, required this.media});
  final MediaModel media;

  static void open(BuildContext context, MediaModel media) {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => VideoPlayerScreen(media: media)),
    );
  }

  @override
  State<VideoPlayerScreen> createState() => _VideoPlayerScreenState();
}

class _VideoPlayerScreenState extends State<VideoPlayerScreen> {
  late VideoPlayerController _videoPlayerController;
  ChewieController? _chewieController;
  bool _isLoading = true;
  bool _hasError = false;
  bool _isDownloading = false;
  String? _localSavedPath;

  @override
  void initState() {
    super.initState();
    _initializePlayer();
  }

  Future<void> _initializePlayer() async {
    try {
      final streamUrl = '${ApiClient.baseUrl}/media/stream/${widget.media.id}';
      debugPrint('Initializing VideoPlayer with streamUrl: $streamUrl');

      try {
        _videoPlayerController = VideoPlayerController.networkUrl(
          Uri.parse(streamUrl),
        );
        await _videoPlayerController.initialize();
      } catch (streamErr) {
        debugPrint('Stream URL failed ($streamErr), attempting direct media.url: ${widget.media.url}');
        if (widget.media.url.isNotEmpty) {
          _videoPlayerController = VideoPlayerController.networkUrl(
            Uri.parse(widget.media.url),
          );
          await _videoPlayerController.initialize();
        } else {
          rethrow;
        }
      }

      _chewieController = ChewieController(
        videoPlayerController: _videoPlayerController,
        autoPlay: true,
        looping: false,
        aspectRatio: _videoPlayerController.value.aspectRatio,
        allowFullScreen: true,
        materialProgressColors: ChewieProgressColors(
          playedColor: AppColors.light.primary,
          handleColor: AppColors.light.primary,
          backgroundColor: AppColors.light.surfaceElevated,
          bufferedColor: AppColors.light.border,
        ),
      );

      if (mounted) {
        setState(() {
          _isLoading = false;
          _hasError = false;
        });
      }
    } catch (e) {
      debugPrint("Video Player Error: $e");
      if (mounted) {
        setState(() {
          _isLoading = false;
          _hasError = true;
        });
      }
    }
  }

  Future<String?> _downloadVideo() async {
    if (_localSavedPath != null && File(_localSavedPath!).existsSync()) {
      return _localSavedPath;
    }
    if (_isDownloading) return null;

    setState(() => _isDownloading = true);
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Downloading video to offline storage...'),
        duration: Duration(seconds: 2),
      ),
    );

    try {
      final dir = await getApplicationDocumentsDirectory();
      final safeName = widget.media.filename.replaceAll(RegExp(r'[^a-zA-Z0-9._-]'), '_');
      final savePath = '${dir.path}/$safeName';

      final downloadUrl = widget.media.url.isNotEmpty
          ? widget.media.url
          : '${ApiClient.baseUrl}/media/stream/${widget.media.id}';

      final dio = Dio();
      await dio.download(downloadUrl, savePath);
      _localSavedPath = savePath;

      String? publicFolder;
      try {
        if (Platform.isAndroid) {
          final publicMoviesDir = Directory('/storage/emulated/0/Movies');
          if (publicMoviesDir.existsSync()) {
            final publicFile = File('${publicMoviesDir.path}/$safeName');
            await File(savePath).copy(publicFile.path);
            publicFolder = 'Movies';
          } else {
            final extMoviesDirs = await getExternalStorageDirectories(type: StorageDirectory.movies);
            if (extMoviesDirs != null && extMoviesDirs.isNotEmpty) {
              final publicFile = File('${extMoviesDirs.first.path}/$safeName');
              await File(savePath).copy(publicFile.path);
              publicFolder = 'Movies';
            }
          }
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
        // Dismiss any stuck snackbar before showing the new one
        ScaffoldMessenger.of(context).hideCurrentSnackBar();
        final displayName = widget.media.filename
            .replaceAll(RegExp(r'\.mp4$'), '')
            .replaceAll('_', ' ');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            behavior: SnackBarBehavior.floating,
            margin: const EdgeInsets.fromLTRB(16, 0, 16, 12),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            backgroundColor: const Color(0xFF2E7D32),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            duration: const Duration(seconds: 4),
            content: Row(
              children: [
                const Icon(Icons.check_circle_rounded, color: Colors.white, size: 20),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Saved to device',
                        style: TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w600,
                          fontSize: 13,
                        ),
                      ),
                      Text(
                        publicFolder != null ? '$displayName · $publicFolder' : displayName,
                        style: const TextStyle(
                          color: Colors.white70,
                          fontSize: 11,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ),
              ],
            ),
            action: SnackBarAction(
              label: 'Share',
              textColor: Colors.white,
              onPressed: _shareVideo,
            ),
          ),
        );
      }
      return savePath;
    } catch (e) {
      debugPrint("Download video error: $e");
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

  Future<void> _shareVideo() async {
    String? localPath = _localSavedPath;
    if (localPath == null || !File(localPath).existsSync()) {
      localPath = await _downloadVideo();
    }
    final displayName = widget.media.filename
        .replaceAll(RegExp(r'\.mp4$'), '')
        .replaceAll('_', ' ');
    if (localPath != null && File(localPath).existsSync()) {
      await Share.shareXFiles(
        [XFile(localPath)],
        text: 'Watch my memory: $displayName',
      );
    } else if (widget.media.url.isNotEmpty) {
      await Share.share('Watch my memory: ${widget.media.url}');
    }
  }

  @override
  void dispose() {
    _videoPlayerController.dispose();
    _chewieController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;

    return Scaffold(
      backgroundColor: adt.AppColors.plum900,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: const IconThemeData(color: adt.AppColors.onDarkPrimary),
        title: Text(
          // Show clean memory name — strip .mp4 extension for display
          widget.media.filename.replaceAll(RegExp(r'\.mp4$'), '').replaceAll('_', ' '),
          style: const TextStyle(color: adt.AppColors.onDarkPrimary, fontSize: 16),
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
        actions: [
          IconButton(
            tooltip: 'Download video offline',
            icon: _isDownloading
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: adt.AppColors.onDarkPrimary,
                    ),
                  )
                : const Icon(Icons.download_rounded, color: adt.AppColors.onDarkPrimary),
            onPressed: _isDownloading ? null : _downloadVideo,
          ),
          IconButton(
            tooltip: 'Share video',
            icon: const Icon(Icons.ios_share, color: adt.AppColors.onDarkPrimary),
            onPressed: _shareVideo,
          ),
        ],
      ),
      body: Center(
        child: _isLoading
            ? CircularProgressIndicator(color: c.primary)
            : _hasError
            ? Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(
                    Icons.error_outline,
                    color: adt.AppColors.onDarkMuted,
                    size: 48,
                  ),
                  const SizedBox(height: AppSpacing.s16),
                  const Text(
                    'Failed to load video',
                    style: TextStyle(color: adt.AppColors.onDarkPrimary),
                  ),
                  const SizedBox(height: AppSpacing.s16),
                  TextButton(
                    onPressed: () {
                      setState(() {
                        _isLoading = true;
                        _hasError = false;
                      });
                      _initializePlayer();
                    },
                    child: Text('Retry', style: TextStyle(color: c.primary)),
                  ),
                ],
              )
            : _chewieController != null
            ? Chewie(controller: _chewieController!)
            : const SizedBox.shrink(),
      ),
    );
  }
}
