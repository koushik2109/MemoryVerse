import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/providers/video_job_controller.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/features/memories/presentation/ai_director_screen.dart';
import 'package:memory_verse/features/memories/presentation/create_memory_sheet.dart';
import 'package:memory_verse/features/memories/presentation/memory_detail_screen.dart';
import 'package:memory_verse/features/memories/presentation/video_player_screen.dart';
import 'package:memory_verse/features/vaults/presentation/create_vault_dialog.dart';
import 'package:memory_verse/features/vaults/presentation/vault_detail_screen.dart';

class HomeAssistantState {
  final bool isLoading;
  final String? query;
  final String? responseMessage;
  final String? intent;
  final String? action;
  final Map<String, dynamic> actionParams;
  final List<String> suggestedChips;
  final String? error;
  final bool isExpanded;

  const HomeAssistantState({
    this.isLoading = false,
    this.query,
    this.responseMessage,
    this.intent,
    this.action,
    this.actionParams = const {},
    this.suggestedChips = const [
      'Create Memory',
      'Show my memories',
      'Check readiness',
      'Generate Video',
      'Show Rooms',
    ],
    this.error,
    this.isExpanded = false,
  });

  HomeAssistantState copyWith({
    bool? isLoading,
    String? query,
    String? responseMessage,
    String? intent,
    String? action,
    Map<String, dynamic>? actionParams,
    List<String>? suggestedChips,
    String? error,
    bool? isExpanded,
    bool clearError = false,
    bool clearAction = false,
  }) {
    return HomeAssistantState(
      isLoading: isLoading ?? this.isLoading,
      query: query ?? this.query,
      responseMessage: responseMessage ?? this.responseMessage,
      intent: intent ?? this.intent,
      action: clearAction ? null : (action ?? this.action),
      actionParams:
          clearAction ? const {} : (actionParams ?? this.actionParams),
      suggestedChips: suggestedChips ?? this.suggestedChips,
      error: clearError ? null : (error ?? this.error),
      isExpanded: isExpanded ?? this.isExpanded,
    );
  }
}

class HomeAssistantController extends StateNotifier<HomeAssistantState> {
  final AiRepository _aiRepo;
  final Ref _ref;

  HomeAssistantController(this._aiRepo, this._ref)
      : super(const HomeAssistantState());

  void setExpanded(bool expanded) {
    state = state.copyWith(isExpanded: expanded);
  }

  void clear() {
    state = const HomeAssistantState();
  }

  Future<void> submitQuery(String message) async {
    final trimmed = message.trim();
    if (trimmed.isEmpty) return;

    state = state.copyWith(
      isLoading: true,
      query: trimmed,
      clearError: true,
      isExpanded: true,
    );

    try {
      final res = await _aiRepo.askHomeAssistant(message: trimmed);
      final reply = res['message'] as String? ?? 'Done.';
      final intent = res['intent'] as String?;
      final rawAction = res['structured_action'] as Map<String, dynamic>?;
      final actionType = rawAction?['action'] as String?;
      final actionParams =
          (rawAction?['params'] as Map<String, dynamic>?) ?? {};
      final chips = (res['suggested_chips'] as List<dynamic>?)
              ?.map((e) => e.toString())
              .toList() ??
          state.suggestedChips;

      state = state.copyWith(
        isLoading: false,
        responseMessage: reply,
        intent: intent,
        action: actionType != 'NONE' ? actionType : null,
        actionParams: actionParams,
        suggestedChips: chips,
      );

      // If a video generation job was spawned, immediately track it in videoJobController
      if (actionType == 'WATCH_VIDEO_JOB' && actionParams['job_id'] != null) {
        final jobId = actionParams['job_id'].toString();
        _ref.read(videoJobControllerProvider.notifier).trackExistingJob(jobId);
      }
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Assistant error: ${e.toString()}',
      );
    }
  }

  void executeAction(BuildContext context) {
    final action = state.action;
    final params = state.actionParams;
    if (action == null) return;

    switch (action) {
      case 'OPEN_MEMORY':
      case 'SHOW_READINESS':
        final memoryId = params['memory_id']?.toString();
        if (memoryId != null && memoryId.isNotEmpty) {
          final memories = _ref.read(memoriesListProvider).valueOrNull ?? [];
          final match = memories.where((m) => m.id == memoryId).firstOrNull;
          if (match != null) {
            MemoryDetailScreen.open(context, match);
          } else {
            _openMemoryById(context, memoryId);
          }
        }
        break;

      case 'CREATE_MEMORY':
        CreateMemorySheet.show(
          context,
          initialPrompt: params['initial_prompt']?.toString(),
          autoStartDiscovery: params['auto_start_discovery'] == true,
        );
        break;

      case 'OPEN_VAULT':
        final vaultId = params['vault_id']?.toString();
        if (vaultId != null && vaultId.isNotEmpty) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => VaultDetailScreen(vaultId: vaultId),
            ),
          );
        }
        break;

      case 'CREATE_VAULT':
        CreateVaultDialog.show(context);
        break;

      case 'WATCH_VIDEO_JOB':
      case 'OPEN_AI_DIRECTOR':
        final jobId = params['job_id']?.toString();
        final memoryId = params['memory_id']?.toString();
        if (jobId != null) {
          _ref
              .read(videoJobControllerProvider.notifier)
              .trackExistingJob(jobId);
        }
        if (memoryId != null && memoryId.isNotEmpty) {
          final memories = _ref.read(memoriesListProvider).valueOrNull ?? [];
          final match = memories.where((m) => m.id == memoryId).firstOrNull;
          if (match != null) {
            AiDirectorScreen.open(context, match);
          } else {
            _openAiDirectorById(context, memoryId);
          }
        }
        break;

      case 'OPEN_VIDEO':
        final videoUrl = params['video_url']?.toString();
        final mediaId = params['media_id']?.toString() ?? 'result_video';
        if (videoUrl != null && videoUrl.isNotEmpty) {
          final dummyMedia = MediaModel(
            id: mediaId,
            memoryId: params['memory_id']?.toString() ?? '',
            ownerId: '',
            filename: 'memory_reel.mp4',
            storagePath: 'reels/memory_reel.mp4',
            url: videoUrl,
            thumbnailUrl: null,
            mediaType: 'video',
            fileSize: 0,
            createdAt: DateTime.now(),
          );
          VideoPlayerScreen.open(context, dummyMedia);
        }
        break;

      default:
        break;
    }
  }

  Future<void> _openMemoryById(BuildContext context, String memoryId) async {
    try {
      final memory = await _ref
          .read(memoryRepositoryProvider)
          .fetchMemoryDetails(memoryId);
      if (context.mounted) {
        MemoryDetailScreen.open(context, memory);
      }
    } catch (_) {}
  }

  Future<void> _openAiDirectorById(
      BuildContext context, String memoryId) async {
    try {
      final memory = await _ref
          .read(memoryRepositoryProvider)
          .fetchMemoryDetails(memoryId);
      if (context.mounted) {
        AiDirectorScreen.open(context, memory);
      }
    } catch (_) {}
  }
}

final homeAssistantProvider =
    StateNotifierProvider<HomeAssistantController, HomeAssistantState>((ref) {
  final aiRepo = ref.watch(aiRepositoryProvider);
  return HomeAssistantController(aiRepo, ref);
});
