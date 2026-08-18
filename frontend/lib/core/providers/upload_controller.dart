import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:uuid/uuid.dart';

enum UploadStatus { waiting, uploading, completed, error, canceled }

class UploadTask {
  final String id;
  final File file;
  final String filename;
  final String mediaType; // 'image' or 'video'
  final String memoryId;
  final String? vaultId;
  final bool isCover;

  final double progress;
  final UploadStatus status;
  final CancelToken? cancelToken;
  final String? error;

  UploadTask({
    required this.id,
    required this.file,
    required this.filename,
    required this.mediaType,
    required this.memoryId,
    this.vaultId,
    this.isCover = false,
    this.progress = 0.0,
    this.status = UploadStatus.waiting,
    this.cancelToken,
    this.error,
  });

  UploadTask copyWith({
    double? progress,
    UploadStatus? status,
    CancelToken? cancelToken,
    String? error,
  }) {
    return UploadTask(
      id: id,
      file: file,
      filename: filename,
      mediaType: mediaType,
      memoryId: memoryId,
      vaultId: vaultId,
      isCover: isCover,
      progress: progress ?? this.progress,
      status: status ?? this.status,
      cancelToken: cancelToken ?? this.cancelToken,
      error: error,
    );
  }
}

class UploadController extends Notifier<List<UploadTask>> {
  @override
  List<UploadTask> build() {
    return [];
  }

  void addUploads({
    required List<File> files,
    required List<String> mediaTypes,
    required String memoryId,
    String? vaultId,
    int? coverIndex,
  }) {
    final newTasks = <UploadTask>[];
    for (int i = 0; i < files.length; i++) {
      newTasks.add(
        UploadTask(
          id: const Uuid().v4(),
          file: files[i],
          filename: files[i].path.split('/').last,
          mediaType: mediaTypes[i],
          memoryId: memoryId,
          vaultId: vaultId,
          isCover: i == coverIndex,
        ),
      );
    }

    state = [...state, ...newTasks];
    _processQueue();
  }

  Future<void> _processQueue() async {
    // Find all waiting tasks
    final waitingTasks = state
        .where((t) => t.status == UploadStatus.waiting)
        .toList();
    if (waitingTasks.isEmpty) return;

    // Process sequentially (or concurrently, but sequential is safer for mobile bandwidth)
    for (final task in waitingTasks) {
      // Check if it's still in the queue (might have been canceled)
      if (!state.any(
        (t) => t.id == task.id && t.status == UploadStatus.waiting,
      ))
        continue;

      await _uploadSingle(task.id);
    }
  }

  Future<void> _uploadSingle(String taskId) async {
    final taskIndex = state.indexWhere((t) => t.id == taskId);
    if (taskIndex == -1) return;

    final task = state[taskIndex];
    final cancelToken = CancelToken();

    _updateTask(
      taskId,
      task.copyWith(
        status: UploadStatus.uploading,
        cancelToken: cancelToken,
        progress: 0.0,
        error: null,
      ),
    );

    try {
      final repo = ref.read(mediaRepositoryProvider);
      final mediaResult = await repo.uploadMedia(
        file: task.file,
        filename: task.filename,
        mediaType: task.mediaType,
        memoryId: task.memoryId,
        vaultId: task.vaultId,
        cancelToken: cancelToken,
        onSendProgress: (count, total) {
          if (total > 0) {
            final progress = count / total;
            // Only update state if progress significantly changed to prevent UI rebuild spam
            final currentTask = state.firstWhere((t) => t.id == taskId);
            if ((progress - currentTask.progress).abs() > 0.05 ||
                progress == 1.0) {
              _updateTask(taskId, currentTask.copyWith(progress: progress));
            }
          }
        },
      );

      // If it's a cover, update the memory
      if (task.isCover) {
        try {
          await ref
              .read(memoryRepositoryProvider)
              .updateMemory(task.memoryId, coverMediaId: mediaResult.id);
        } catch (_) {
          // Non-critical if setting cover fails
        }
      }

      _updateTask(
        taskId,
        state
            .firstWhere((t) => t.id == taskId)
            .copyWith(status: UploadStatus.completed, progress: 1.0),
      );

      // Invalidate relevant providers to refresh UI
      ref.invalidate(memoriesListProvider);
      ref.invalidate(memoryDetailProvider(task.memoryId));
    } catch (e) {
      if (e is DioException && CancelToken.isCancel(e)) {
        _updateTask(
          taskId,
          state
              .firstWhere((t) => t.id == taskId)
              .copyWith(status: UploadStatus.canceled),
        );
      } else {
        _updateTask(
          taskId,
          state
              .firstWhere((t) => t.id == taskId)
              .copyWith(status: UploadStatus.error, error: e.toString()),
        );
      }
    }
  }

  void _updateTask(String id, UploadTask newTask) {
    state = [
      for (final t in state)
        if (t.id == id) newTask else t,
    ];
  }

  void cancelUpload(String taskId) {
    final task = state.firstWhere((t) => t.id == taskId);
    if (task.status == UploadStatus.uploading) {
      task.cancelToken?.cancel();
    }
    _updateTask(taskId, task.copyWith(status: UploadStatus.canceled));
  }

  void retryUpload(String taskId) {
    final task = state.firstWhere((t) => t.id == taskId);
    _updateTask(
      taskId,
      task.copyWith(status: UploadStatus.waiting, progress: 0.0, error: null),
    );
    _processQueue();
  }

  void removeCompleted() {
    state = state.where((t) => t.status != UploadStatus.completed).toList();
  }

  void clearAll() {
    for (final task in state) {
      if (task.status == UploadStatus.uploading) {
        task.cancelToken?.cancel();
      }
    }
    state = [];
  }
}

final uploadControllerProvider =
    NotifierProvider<UploadController, List<UploadTask>>(() {
      return UploadController();
    });
