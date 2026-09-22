import 'dart:async';
import 'dart:convert';
import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/api/api_client.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

/// State representation of the active background video generation job in the UI
class VideoJobUIState {
  final VideoJobModel? job;
  final bool isTracking;
  final bool isStreamConnected;
  final String? clientError;
  final bool isSheetOpen;
  final bool isDirectorOpen;

  const VideoJobUIState({
    this.job,
    this.isTracking = false,
    this.isStreamConnected = false,
    this.clientError,
    this.isSheetOpen = false,
    this.isDirectorOpen = false,
  });

  bool get hasActiveJob => job != null && (job?.isProcessing == true);
  bool get hasTerminalJob => job != null && !(job?.isProcessing == true);

  VideoJobUIState copyWith({
    VideoJobModel? job,
    bool? isTracking,
    bool? isStreamConnected,
    String? clientError,
    bool? isSheetOpen,
    bool? isDirectorOpen,
    bool clearJob = false,
  }) {
    return VideoJobUIState(
      job: clearJob ? null : (job ?? this.job),
      isTracking: isTracking ?? this.isTracking,
      isStreamConnected: isStreamConnected ?? this.isStreamConnected,
      clientError: clientError,
      isSheetOpen: isSheetOpen ?? this.isSheetOpen,
      isDirectorOpen: isDirectorOpen ?? this.isDirectorOpen,
    );
  }
}

/// Standalone provider tracking whether the user is actively inside the AI Director screen
final aiDirectorActiveProvider = StateProvider<bool>((ref) => false);

/// Controller managing video job creation, SSE live updates, adaptive polling fallback,
/// cancellation, retry, and non-blocking background lifecycle.
class VideoJobController extends StateNotifier<VideoJobUIState> {
  final MediaRepository _mediaRepo;
  final ApiClient _apiClient;
  StreamSubscription? _streamSub;
  Timer? _pollingTimer;
  CancelToken? _cancelToken;

  VideoJobController(this._mediaRepo, this._apiClient)
      : super(const VideoJobUIState());

  /// Sets whether the modal video creator sheet is currently mounted
  void setSheetOpen(bool open) {
    state = state.copyWith(isSheetOpen: open);
  }

  /// Sets whether the AI Director screen is currently open
  void setDirectorOpen(bool open) {
    state = state.copyWith(isDirectorOpen: open);
  }

  /// Enqueue a new video job and commence background tracking.
  Future<String?> startJob({
    required String memoryId,
    String? dimension,
    String? mood,
    List<String>? mediaIds,
  }) async {
    try {
      state = state.copyWith(isTracking: true, clientError: null);
      final jobId = await _mediaRepo.generateVideo(
        memoryId,
        dimension: dimension,
        mood: mood,
        mediaIds: mediaIds,
      );

      final initialJob = VideoJobModel(
        id: jobId,
        status: 'queued',
        stage: 'queued',
        memoryId: memoryId,
        currentTask: 'Waiting for available background worker...',
      );
      state = state.copyWith(job: initialJob, isTracking: true);

      _startSSEStream(jobId);
      return jobId;
    } catch (e) {
      state = state.copyWith(
        isTracking: false,
        clientError: e.toString(),
      );
      return null;
    }
  }

  /// Attach tracking to an existing job (e.g. from notification or history)
  void trackExistingJob(String jobId) {
    _stopTracking(keepJob: true);
    state = state.copyWith(isTracking: true, clientError: null);
    _startSSEStream(jobId);
  }

  /// Establishes Server-Sent Events (SSE) stream for sub-second updates.
  Future<void> _startSSEStream(String jobId) async {
    _streamSub?.cancel();
    _cancelToken?.cancel();
    _cancelToken = CancelToken();

    final session = Supabase.instance.client.auth.currentSession;
    final token = session?.accessToken;

    try {
      final response = await _apiClient.dio.get<ResponseBody>(
        '/media/jobs/$jobId/stream',
        queryParameters: token != null ? {'token': token} : null,
        options: Options(
          responseType: ResponseType.stream,
          headers: {
            'Accept': 'text/event-stream',
            'Cache-Control': 'no-cache',
          },
        ),
        cancelToken: _cancelToken,
      );

      state = state.copyWith(isStreamConnected: true, clientError: null);

      final stream = response.data?.stream;
      if (stream == null) {
        _startAdaptivePolling(jobId);
        return;
      }

      StringBuffer buffer = StringBuffer();

      _streamSub = stream
          .map((uint8List) => utf8.decode(uint8List, allowMalformed: true))
          .listen(
        (chunk) {
          buffer.write(chunk);
          final text = buffer.toString();
          final lines = text.split('\n');

          if (!text.endsWith('\n')) {
            buffer = StringBuffer(lines.removeLast());
          } else {
            buffer.clear();
          }

          for (final line in lines) {
            final trimmed = line.trim();
            if (trimmed.startsWith('data:')) {
              final jsonStr = trimmed.substring(5).trim();
              if (jsonStr.isNotEmpty) {
                try {
                  final data = json.decode(jsonStr) as Map<String, dynamic>;
                  final updatedJob = VideoJobModel.fromJson(data);
                  state = state.copyWith(job: updatedJob);

                  if (updatedJob.isCompleted ||
                      updatedJob.isFailed ||
                      updatedJob.isCancelled) {
                    _stopTracking(keepJob: true);
                  }
                } catch (_) {}
              }
            }
          }
        },
        onError: (err) {
          // SSE stream encountered an error or network drop; fallback to polling
          state = state.copyWith(isStreamConnected: false);
          _startAdaptivePolling(jobId);
        },
        onDone: () {
          if (state.job != null && state.job!.isProcessing) {
            _startAdaptivePolling(jobId);
          }
        },
        cancelOnError: true,
      );
    } catch (e) {
      // Fall back directly to polling if streaming endpoint is unreachable
      state = state.copyWith(isStreamConnected: false);
      _startAdaptivePolling(jobId);
    }
  }

  /// Adaptive polling fallback with interval tuned to current stage (1s - 3s)
  void _startAdaptivePolling(String jobId) {
    _pollingTimer?.cancel();
    int intervalSeconds = 2;

    void poll() async {
      if (!mounted) return;
      try {
        final job = await _mediaRepo.getJobStatus(jobId);
        state = state.copyWith(job: job);

        if (job.isCompleted || job.isFailed || job.isCancelled) {
          _stopTracking(keepJob: true);
          return;
        }

        if (job.stage == 'uploading') {
          intervalSeconds = 1;
        } else if (job.stage == 'composing_video' || job.stage == 'encoding_video') {
          intervalSeconds = 3;
        } else {
          intervalSeconds = 2;
        }
      } catch (e) {
        intervalSeconds = 4; // Backoff upon transient network failure
      }

      if (mounted && state.isTracking) {
        _pollingTimer = Timer(Duration(seconds: intervalSeconds), poll);
      }
    }

    poll();
  }

  /// Retry the current failed job (resumes upload if staged, or recovers)
  Future<void> retry() async {
    final currentJob = state.job;
    if (currentJob == null) return;
    try {
      state = state.copyWith(
        isTracking: true,
        clientError: null,
        job: currentJob.copyWith(
          status: 'retrying',
          stage: 'retrying',
          currentTask: 'Recovering and resuming generation...',
        ),
      );
      await _mediaRepo.retryJob(currentJob.id);
      _startSSEStream(currentJob.id);
    } catch (e) {
      state = state.copyWith(clientError: e.toString());
    }
  }

  /// Cancel the active video generation job
  Future<void> cancel() async {
    final currentJob = state.job;
    if (currentJob == null) return;
    try {
      await _mediaRepo.cancelJob(currentJob.id);
      state = state.copyWith(
        job: currentJob.copyWith(
          status: 'cancelled',
          currentTask: 'Memory video creation cancelled by user.',
        ),
      );
      _stopTracking(keepJob: true);
    } catch (e) {
      state = state.copyWith(clientError: e.toString());
    }
  }

  /// Dismiss the banner/modal without stopping background work
  void dismiss() {
    _stopTracking(keepJob: false);
    state = const VideoJobUIState();
  }

  void _stopTracking({required bool keepJob}) {
    _streamSub?.cancel();
    _streamSub = null;
    _cancelToken?.cancel();
    _cancelToken = null;
    _pollingTimer?.cancel();
    _pollingTimer = null;
    state = state.copyWith(
      isTracking: false,
      isStreamConnected: false,
      clearJob: !keepJob,
    );
  }

  @override
  void dispose() {
    _stopTracking(keepJob: false);
    super.dispose();
  }
}

/// Global provider for the active video job controller
final videoJobControllerProvider =
    StateNotifierProvider<VideoJobController, VideoJobUIState>((ref) {
  final mediaRepo = ref.watch(mediaRepositoryProvider);
  final apiClient = ref.watch(apiClientProvider);
  return VideoJobController(mediaRepo, apiClient);
});
