import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/providers/video_job_controller.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';

class AiDirectorState {
  final AISessionModel? session;
  final bool isLoading;
  final String? error;
  final String? activeJobId;
  final bool isGenerating;

  const AiDirectorState({
    this.session,
    this.isLoading = false,
    this.error,
    this.activeJobId,
    this.isGenerating = false,
  });

  AiDirectorState copyWith({
    AISessionModel? session,
    bool? isLoading,
    String? error,
    String? activeJobId,
    bool? isGenerating,
    bool clearError = false,
  }) {
    return AiDirectorState(
      session: session ?? this.session,
      isLoading: isLoading ?? this.isLoading,
      error: clearError ? null : (error ?? this.error),
      activeJobId: activeJobId ?? this.activeJobId,
      isGenerating: isGenerating ?? this.isGenerating,
    );
  }
}

class AiDirectorController extends StateNotifier<AiDirectorState> {
  final AiDirectorRepository _repo;
  final Ref _ref;
  int _sessionGeneration = 0;

  AiDirectorController(this._repo, this._ref) : super(const AiDirectorState());

  Future<void> initSession(String memoryId, {String? initialPrompt}) async {
    final generation = ++_sessionGeneration;
    // A provider can be reused when the user opens the Director for another
    // memory. Clear the previous conversation before starting the new request.
    state = const AiDirectorState(isLoading: true);
    try {
      final session = await _repo.createOrGetSession(
        memoryId,
        initialPrompt: initialPrompt,
      );
      if (generation != _sessionGeneration) return;
      state = state.copyWith(
        session: session,
        isLoading: false,
        activeJobId: session.activeJobId,
      );
    } catch (e) {
      if (generation != _sessionGeneration) return;
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to initialize AI Memory Director: ${e.toString()}',
      );
    }
  }

  Future<void> sendMessage(String text) async {
    final session = state.session;
    if (session == null || text.trim().isEmpty) return;
    final generation = _sessionGeneration;

    // Optimistically add user message
    final optimisticMsg = AIMessageModel(
      id: 'opt_${DateTime.now().millisecondsSinceEpoch}',
      role: 'user',
      content: text,
      createdAt: DateTime.now(),
    );
    final updatedMessages = List<AIMessageModel>.from(session.messages)
      ..add(optimisticMsg);
    state = state.copyWith(
      session: session.copyWith(messages: updatedMessages),
      isLoading: true,
      clearError: true,
    );

    try {
      final updatedSession = await _repo.sendMessage(session.sessionId, text);
      if (!_isCurrentSession(generation, session.sessionId)) return;
      state = state.copyWith(
        session: updatedSession,
        isLoading: false,
      );
    } catch (e) {
      if (!_isCurrentSession(generation, session.sessionId)) return;
      state = state.copyWith(
        session: session,
        isLoading: false,
        error: 'Failed to send message: ${e.toString()}',
      );
    }
  }

  Future<String?> approveAndGenerate() async {
    final session = state.session;
    if (session == null) return null;
    final generation = _sessionGeneration;

    state = state.copyWith(isGenerating: true, clearError: true);
    try {
      final res = await _repo.approveAndGenerate(
        session.sessionId,
        spec: session.specification,
      );
      if (!_isCurrentSession(generation, session.sessionId)) return null;
      final jobId = res['job_id']?.toString();

      if (jobId != null) {
        state = state.copyWith(
          activeJobId: jobId,
          isGenerating: false,
        );
        // Bind background tracking into VideoJobController
        _ref.read(videoJobControllerProvider.notifier).trackExistingJob(jobId);
        return jobId;
      }
      state = state.copyWith(isGenerating: false);
      return null;
    } catch (e) {
      if (!_isCurrentSession(generation, session.sessionId)) return null;
      state = state.copyWith(
        isGenerating: false,
        error: 'Failed to generate video: ${e.toString()}',
      );
      return null;
    }
  }

  Future<void> requestRevision(String prompt) async {
    final session = state.session;
    if (session == null || prompt.trim().isEmpty) return;
    final generation = _sessionGeneration;

    state = state.copyWith(isLoading: true, clearError: true);
    try {
      final res = await _repo.requestRevision(session.sessionId, prompt);
      if (!_isCurrentSession(generation, session.sessionId)) return;
      if (res['session'] != null) {
        final updatedSession =
            AISessionModel.fromJson(res['session'] as Map<String, dynamic>);
        state = state.copyWith(session: updatedSession, isLoading: false);
      } else {
        state = state.copyWith(isLoading: false);
      }
    } catch (e) {
      if (!_isCurrentSession(generation, session.sessionId)) return;
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to apply revision: ${e.toString()}',
      );
    }
  }

  bool _isCurrentSession(int generation, String sessionId) {
    return generation == _sessionGeneration &&
        state.session?.sessionId == sessionId;
  }
}

final aiDirectorControllerProvider =
    StateNotifierProvider.autoDispose<AiDirectorController, AiDirectorState>(
        (ref) {
  return AiDirectorController(
    ref.watch(aiDirectorRepositoryProvider),
    ref,
  );
});
