import 'package:flutter/material.dart';
import 'package:chewie/chewie.dart';
import 'package:video_player/video_player.dart';
import 'package:memory_verse/contracts/models.dart';
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

  @override
  void initState() {
    super.initState();
    _initializePlayer();
  }

  Future<void> _initializePlayer() async {
    try {
      _videoPlayerController = VideoPlayerController.networkUrl(
        Uri.parse(widget.media.url),
      );
      await _videoPlayerController.initialize();

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

      setState(() {
        _isLoading = false;
      });
    } catch (e) {
      debugPrint("Video Player Error: $e");
      setState(() {
        _isLoading = false;
        _hasError = true;
      });
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
          widget.media.filename,
          style: const TextStyle(color: adt.AppColors.onDarkPrimary, fontSize: 16),
        ),
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
