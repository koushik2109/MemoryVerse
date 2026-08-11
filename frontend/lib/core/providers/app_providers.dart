import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';

final vaultsListProvider = FutureProvider<List<VaultModel>>((ref) async {
  final repo = ref.watch(vaultRepositoryProvider);
  return repo.fetchVaults();
});

final userMediaProvider = FutureProvider<List<MediaModel>>((ref) async {
  final repo = ref.watch(mediaRepositoryProvider);
  return repo.fetchMedia(limit: 50);
});

final memoriesListProvider = FutureProvider<List<MemoryModel>>((ref) async {
  final repo = ref.watch(memoryRepositoryProvider);
  return repo.fetchMemories(limit: 50);
});

final memoryDetailProvider = FutureProvider.family<MemoryModel, String>((ref, id) async {
  final repo = ref.watch(memoryRepositoryProvider);
  return repo.fetchMemoryDetails(id);
});

final userProfileProvider = FutureProvider<UserModel>((ref) async {
  final repo = ref.watch(profileRepositoryProvider);
  return repo.fetchProfile();
});

final notificationsListProvider = FutureProvider<List<NotificationModel>>((ref) async {
  final repo = ref.watch(notificationRepositoryProvider);
  return repo.fetchNotifications();
});

final searchResultsProvider = FutureProvider.family<Map<String, dynamic>, String>((ref, query) async {
  if (query.trim().isEmpty) return {'vaults': [], 'media': [], 'collaborators': []};
  final repo = ref.watch(searchRepositoryProvider);
  return repo.search(query);
});

// Timeline provider
final timelineProvider = FutureProvider<TimelineResponse>((ref) async {
  final repo = ref.watch(timelineRepositoryProvider);
  return repo.fetchTimeline();
});

// AI providers
final aiConversationsProvider = FutureProvider<List<AiConversationModel>>((ref) async {
  final repo = ref.watch(aiRepositoryProvider);
  return repo.listConversations();
});

