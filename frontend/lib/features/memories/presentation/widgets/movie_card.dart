import 'dart:io';
import 'dart:math' as math;
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/widgets/media.dart';
import 'package:memory_verse/features/memories/presentation/video_player_screen.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart' as adt;

class MovieCard extends StatefulWidget {
  final MediaModel movie;
  final String memoryTitle;
  final VoidCallback? onPlay;

  const MovieCard({
    super.key,
    required this.movie,
    required this.memoryTitle,
    this.onPlay,
  });

  @override
  State<MovieCard> createState() => _MovieCardState();
}

class _MovieCardState extends State<MovieCard> {
  bool _isDownloading = false;
  double _downloadProgress = 0.0;
  String? _localSavedPath;

  Future<String?> _downloadVideo({bool forShare = false}) async {
    final videoUrl = widget.movie.url;
    if (videoUrl.isEmpty) return null;

    if (_localSavedPath != null && File(_localSavedPath!).existsSync()) {
      return _localSavedPath;
    }

    if (_isDownloading) return null;

    setState(() {
      _isDownloading = true;
      _downloadProgress = 0.0;
    });

    try {
      final dir = await getApplicationDocumentsDirectory();
      final sanitizedTitle = widget.memoryTitle
          .replaceAll(RegExp(r'[^a-zA-Z0-9]'), '_')
          .trim();
      final safeId = widget.movie.id.substring(0, math.min(6, widget.movie.id.length));
      final filename = 'Movie_${sanitizedTitle}_$safeId.mp4';
      final savePath = '${dir.path}/$filename';

      final dio = Dio();
      await dio.download(
        videoUrl,
        savePath,
        onReceiveProgress: (received, total) {
          if (total > 0 && mounted) {
            setState(() {
              _downloadProgress = received / total;
            });
          }
        },
      );

      _localSavedPath = savePath;

      String? publicFolder;
      try {
        if (Platform.isAndroid) {
          final publicMoviesDir = Directory('/storage/emulated/0/Movies');
          if (publicMoviesDir.existsSync()) {
            final publicFile = File('${publicMoviesDir.path}/$filename');
            await File(savePath).copy(publicFile.path);
            publicFolder = 'Movies';
          } else {
            final extMoviesDirs = await getExternalStorageDirectories(type: StorageDirectory.movies);
            if (extMoviesDirs != null && extMoviesDirs.isNotEmpty) {
              final publicFile = File('${extMoviesDirs.first.path}/$filename');
              await File(savePath).copy(publicFile.path);
              publicFolder = 'Movies';
            }
          }
        } else {
          final dlDir = await getDownloadsDirectory();
          if (dlDir != null) {
            final publicFile = File('${dlDir.path}/$filename');
            await File(savePath).copy(publicFile.path);
            publicFolder = 'Downloads';
          }
        }
      } catch (copyErr) {
        debugPrint('Could not copy to public movies directory: $copyErr');
      }

      if (mounted) {
        setState(() {
          _isDownloading = false;
        });

        if (!forShare) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                publicFolder != null
                    ? 'Movie saved to device $publicFolder folder: $filename'
                    : 'Movie saved locally: $filename',
              ),
              backgroundColor: const Color(0xFF2E7D32),
              duration: const Duration(seconds: 4),
              action: SnackBarAction(
                label: 'Share',
                textColor: Colors.white,
                onPressed: _shareVideo,
              ),
            ),
          );
        }
      }
      return savePath;
    } catch (e) {
      if (mounted) {
        setState(() {
          _isDownloading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Download failed: $e'),
            backgroundColor: Colors.redAccent,
          ),
        );
      }
      return null;
    }
  }

  Future<void> _shareVideo() async {
    String? filePath = _localSavedPath;
    if (filePath == null || !File(filePath).existsSync()) {
      filePath = await _downloadVideo(forShare: true);
    }

    if (filePath != null && File(filePath).existsSync()) {
      await Share.shareXFiles(
        [XFile(filePath)],
        text: 'Watch my memory movie: ${widget.memoryTitle}',
      );
    } else if (widget.movie.url.isNotEmpty) {
      await Share.share(
        'Watch my memory movie: ${widget.memoryTitle}\n${widget.movie.url}',
      );
    }
  }

  void _openPlayer() {
    if (widget.onPlay != null) {
      widget.onPlay!();
    } else {
      VideoPlayerScreen.open(context, widget.movie);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final thumbUrl = widget.movie.thumbnailUrl ?? widget.movie.url;
    final title = widget.movie.displayTitle.isNotEmpty
        ? widget.movie.displayTitle
        : widget.memoryTitle;

    return Container(
      margin: const EdgeInsets.symmetric(vertical: AppSpacing.s8),
      decoration: BoxDecoration(
        color: c.surfaceElevated,
        borderRadius: BorderRadius.circular(AppRadii.xl),
        border: Border.all(
          color: c.primary.withValues(alpha: 0.25),
          width: 1.2,
        ),
        boxShadow: [
          BoxShadow(
            color: adt.AppColors.plum900.withValues(alpha: 0.35),
            blurRadius: 16,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Movie Coverpage & Video Player Preview
          GestureDetector(
            onTap: _openPlayer,
            child: AspectRatio(
              aspectRatio: 16 / 9,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  // Cover Image (Intro coverpage or user-chosen photo)
                  if (thumbUrl.isNotEmpty)
                    AppNetworkImage(
                      imageUrl: thumbUrl,
                      fit: BoxFit.cover,
                      memCacheWidth: 800,
                    )
                  else
                    Container(
                      decoration: const BoxDecoration(
                        gradient: LinearGradient(
                          colors: [
                            Color(0xFF1E0E2C),
                            Color(0xFF2C133C),
                            Color(0xFF150A20),
                          ],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                      ),
                      child: Center(
                        child: Icon(
                          Icons.movie_creation_rounded,
                          size: 48,
                          color: c.primary.withValues(alpha: 0.6),
                        ),
                      ),
                    ),

                  // Dark gradient vignette
                  Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          Colors.black.withValues(alpha: 0.3),
                          Colors.transparent,
                          adt.AppColors.plum900.withValues(alpha: 0.85),
                        ],
                        stops: const [0.0, 0.4, 1.0],
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                      ),
                    ),
                  ),

                  // Luminous Play Button in Center
                  Center(
                    child: Container(
                      width: 58,
                      height: 58,
                      decoration: BoxDecoration(
                        color: adt.AppColors.plum900.withValues(alpha: 0.75),
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.8),
                          width: 2,
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: c.primary.withValues(alpha: 0.5),
                            blurRadius: 18,
                            spreadRadius: 2,
                          ),
                        ],
                      ),
                      child: const Icon(
                        Icons.play_arrow_rounded,
                        color: Colors.white,
                        size: 34,
                      ),
                    ),
                  ),

                  // Top Pill: "AI Movie" Badge
                  Positioned(
                    top: 12,
                    left: 12,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: adt.AppColors.plum900.withValues(alpha: 0.75),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: c.primary.withValues(alpha: 0.4),
                          width: 0.8,
                        ),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.auto_awesome,
                            size: 13,
                            color: c.primary,
                          ),
                          const SizedBox(width: 4),
                          Text(
                            'Movie',
                            style: TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                              color: c.primary,
                              letterSpacing: 0.3,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                  // Top Right: Duration / Quality badge
                  Positioned(
                    top: 12,
                    right: 12,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.65),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        '1080p · MP4',
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.9),
                        ),
                      ),
                    ),
                  ),

                  // Bottom Title on Thumbnail
                  Positioned(
                    bottom: 12,
                    left: 16,
                    right: 16,
                    child: Text(
                      title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                        shadows: [
                          Shadow(
                            color: Colors.black,
                            blurRadius: 6,
                            offset: Offset(0, 2),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Download progress indicator
          if (_isDownloading)
            LinearProgressIndicator(
              value: _downloadProgress > 0 ? _downloadProgress : null,
              backgroundColor: c.primary.withValues(alpha: 0.1),
              valueColor: AlwaysStoppedAnimation<Color>(c.primary),
              minHeight: 3,
            ),

          // Action Toolbar: Download, Share, Play
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 14),
            child: Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    icon: _isDownloading
                        ? SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: c.primary,
                            ),
                          )
                        : const Icon(Icons.download_rounded, size: 18),
                    label: Text(
                      _isDownloading
                          ? '${(_downloadProgress * 100).toInt()}%'
                          : 'Download',
                      style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
                    ),
                    onPressed: _isDownloading ? null : () => _downloadVideo(),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: adt.AppColors.onDarkPrimary,
                      side: BorderSide(
                        color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.2),
                      ),
                      padding: const EdgeInsets.symmetric(vertical: 10),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: OutlinedButton.icon(
                    icon: const Icon(Icons.share_rounded, size: 18),
                    label: const Text(
                      'Share',
                      style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
                    ),
                    onPressed: _shareVideo,
                    style: OutlinedButton.styleFrom(
                      foregroundColor: adt.AppColors.onDarkPrimary,
                      side: BorderSide(
                        color: adt.AppColors.onDarkPrimary.withValues(alpha: 0.2),
                      ),
                      padding: const EdgeInsets.symmetric(vertical: 10),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                FilledButton.icon(
                  icon: const Icon(Icons.play_arrow_rounded, size: 20),
                  label: const Text('Play'),
                  onPressed: _openPlayer,
                  style: FilledButton.styleFrom(
                    backgroundColor: c.primary,
                    foregroundColor: c.primaryInverse,
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
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
