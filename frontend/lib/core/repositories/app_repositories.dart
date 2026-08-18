import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/api/api_client.dart';

import 'dart:io';

// ── VAULT REPOSITORY ─────────────────────────────────────
class VaultRepository {
  final ApiClient _api;
  VaultRepository(this._api);

  Future<List<VaultModel>> fetchVaults() async {
    final res = await _api.get('/vaults');
    return (res.data as List).map((x) => VaultModel.fromJson(x)).toList();
  }

  Future<VaultModel> fetchVaultDetails(String id) async {
    final res = await _api.get('/vaults/$id');
    return VaultModel.fromJson(res.data);
  }

  Future<VaultModel> createVault({
    required String name,
    String? description,
    String? coverImageUrl,
  }) async {
    final res = await _api.post(
      '/vaults',
      data: {
        'name': name,
        'description': description,
        'cover_image_url': coverImageUrl,
      },
    );
    return VaultModel.fromJson(res.data);
  }

  Future<VaultModel> updateVault(
    String id, {
    String? name,
    String? description,
    String? coverImageUrl,
    bool? isArchived,
  }) async {
    final res = await _api.put(
      '/vaults/$id',
      data: {
        'name': name,
        'description': description,
        'cover_image_url': coverImageUrl,
        'is_archived': isArchived,
      },
    );
    return VaultModel.fromJson(res.data);
  }

  Future<void> deleteVault(String id) async {
    await _api.delete('/vaults/$id');
  }

  Future<void> leaveVault(String id) async {
    await _api.post('/vaults/$id/leave');
  }

  Future<InviteModel> generateInvite(String vaultId) async {
    final res = await _api.post('/invites/vaults/$vaultId/invite-link');
    return InviteModel.fromJson(res.data);
  }

  Future<InviteModel> getInviteInfo(String code) async {
    final res = await _api.get('/invites/$code');
    return InviteModel.fromJson(res.data);
  }

  Future<VaultModel> acceptInvite(String code) async {
    final res = await _api.post('/invites/$code/accept');
    return VaultModel.fromJson(res.data);
  }

  Future<VaultModel> joinVault(String inviteCode) async {
    final res = await _api.post(
      '/vaults/join',
      data: {'invite_code': inviteCode},
    );
    return VaultModel.fromJson(res.data);
  }

  Future<void> removeMember(String vaultId, String targetUserId) async {
    await _api.delete('/vaults/$vaultId/members/$targetUserId');
  }
}

final vaultRepositoryProvider = Provider<VaultRepository>((ref) {
  return VaultRepository(ref.watch(apiClientProvider));
});

// ── MEMORY REPOSITORY ─────────────────────────────────────
class MemoryRepository {
  final ApiClient _api;
  MemoryRepository(this._api);

  Future<List<MemoryModel>> fetchMemories({
    String? vaultId,
    int limit = 50,
  }) async {
    final query = <String, dynamic>{'limit': limit};
    if (vaultId != null) query['vault_id'] = vaultId;
    final res = await _api.get('/memories', queryParameters: query);
    return (res.data as List).map((x) => MemoryModel.fromJson(x)).toList();
  }

  Future<MemoryModel> fetchMemoryDetails(String id) async {
    final res = await _api.get('/memories/$id');
    return MemoryModel.fromJson(res.data);
  }

  Future<MemoryModel> createMemory({
    required String title,
    String? description,
    String? vaultId,
    String? locationName,
  }) async {
    final res = await _api.post(
      '/memories',
      data: {
        'title': title,
        'description': description,
        if (vaultId != null) 'vault_id': vaultId,
        if (locationName != null) 'location_name': locationName,
      },
    );
    return MemoryModel.fromJson(res.data);
  }

  Future<MemoryModel> updateMemory(
    String id, {
    String? title,
    String? description,
    String? coverMediaId,
    String? locationName,
    DateTime? memoryDate,
  }) async {
    final res = await _api.put(
      '/memories/$id',
      data: {
        if (title != null) 'title': title,
        if (description != null) 'description': description,
        if (coverMediaId != null) 'cover_media_id': coverMediaId,
        if (locationName != null) 'location_name': locationName,
        if (memoryDate != null) 'memory_date': memoryDate.toIso8601String(),
      },
    );
    return MemoryModel.fromJson(res.data);
  }

  Future<void> deleteMemory(String id) async {
    await _api.delete('/memories/$id');
  }
}

final memoryRepositoryProvider = Provider<MemoryRepository>((ref) {
  return MemoryRepository(ref.watch(apiClientProvider));
});

// ── MEDIA REPOSITORY ─────────────────────────────────────
class MediaRepository {
  final ApiClient _api;
  MediaRepository(this._api);

  Future<List<MediaModel>> fetchMedia({String? vaultId, int limit = 50}) async {
    final query = <String, dynamic>{'limit': limit};
    if (vaultId != null) query['vault_id'] = vaultId;
    final res = await _api.get('/media', queryParameters: query);
    return (res.data as List).map((x) => MediaModel.fromJson(x)).toList();
  }

  /// Server-side upload: sends the file to the backend, which runs moviepy
  /// thumbnail extraction and uploads to Supabase Storage.
  Future<MediaModel> uploadMedia({
    required File file,
    required String filename,
    required String mediaType, // image or video
    String? vaultId,
    String? memoryId,
    CancelToken? cancelToken,
    ProgressCallback? onSendProgress,
  }) async {
    final formData = {
      'file': await MultipartFile.fromFile(file.path, filename: filename),
      if (vaultId != null) 'vault_id': vaultId,
      if (memoryId != null) 'memory_id': memoryId,
    };

    final res = await _api.postForm(
      '/media/upload',
      formData: formData,
      cancelToken: cancelToken,
      onSendProgress: onSendProgress,
    );
    return MediaModel.fromJson(res.data);
  }

  Future<void> deleteMedia(String id) async {
    await _api.delete('/media/$id');
  }

  Future<String> generateVideo(String memoryId, {String? dimension}) async {
    final query = dimension != null ? {'dimension': dimension} : null;
    final res = await _api.post(
      '/media/memory/$memoryId/generate-video',
      queryParameters: query,
    );
    return res.data['job_id'] as String;
  }

  Future<VideoJobModel> getJobStatus(String jobId) async {
    final res = await _api.get('/media/jobs/$jobId');
    return VideoJobModel.fromJson(res.data);
  }

  Future<void> reorderMedia(List<String> mediaIds) async {
    await _api.put('/media/reorder', data: {'media_ids': mediaIds});
  }
}

final mediaRepositoryProvider = Provider<MediaRepository>((ref) {
  return MediaRepository(ref.watch(apiClientProvider));
});

// ── PROFILE REPOSITORY ───────────────────────────────────

class ProfileRepository {
  final ApiClient _api;
  ProfileRepository(this._api);

  Future<UserModel> fetchProfile() async {
    final res = await _api.get('/profile');
    return UserModel.fromJson(res.data);
  }

  Future<UserModel> updateProfile({
    String? fullName,
    String? username,
    String? avatarUrl,
    String? bio,
  }) async {
    final res = await _api.put(
      '/profile',
      data: {
        'full_name': fullName,
        'username': username,
        'avatar_url': avatarUrl,
        'bio': bio,
      },
    );
    return UserModel.fromJson(res.data);
  }

  Future<void> deleteAccount() async {
    await _api.delete('/profile/account');
  }
}

final profileRepositoryProvider = Provider<ProfileRepository>((ref) {
  return ProfileRepository(ref.watch(apiClientProvider));
});

// ── NOTIFICATION REPOSITORY ───────────────────────────────
class NotificationRepository {
  final ApiClient _api;
  NotificationRepository(this._api);

  Future<List<NotificationModel>> fetchNotifications() async {
    final res = await _api.get('/notifications');
    return (res.data as List)
        .map((x) => NotificationModel.fromJson(x))
        .toList();
  }

  Future<void> markRead(String id) async {
    await _api.post('/notifications/$id/read');
  }

  Future<void> markAllRead() async {
    await _api.post('/notifications/read-all');
  }
}

final notificationRepositoryProvider = Provider<NotificationRepository>((ref) {
  return NotificationRepository(ref.watch(apiClientProvider));
});

// ── SEARCH REPOSITORY ────────────────────────────────────
class SearchRepository {
  final ApiClient _api;
  SearchRepository(this._api);

  Future<Map<String, dynamic>> search(String query) async {
    final res = await _api.get('/search', queryParameters: {'q': query});
    return res.data as Map<String, dynamic>;
  }
}

final searchRepositoryProvider = Provider<SearchRepository>((ref) {
  return SearchRepository(ref.watch(apiClientProvider));
});

// ── TIMELINE REPOSITORY ──────────────────────────────────────
class TimelineRepository {
  final ApiClient _api;
  TimelineRepository(this._api);

  Future<TimelineResponse> fetchTimeline({String? vaultId}) async {
    final query = <String, dynamic>{};
    if (vaultId != null) query['vault_id'] = vaultId;
    final res = await _api.get(
      '/timeline',
      queryParameters: query.isEmpty ? null : query,
    );
    return TimelineResponse.fromJson(res.data as Map<String, dynamic>);
  }
}

final timelineRepositoryProvider = Provider<TimelineRepository>((ref) {
  return TimelineRepository(ref.watch(apiClientProvider));
});

// ── AI REPOSITORY ────────────────────────────────────────────
class AiRepository {
  final ApiClient _api;
  AiRepository(this._api);

  Future<Map<String, dynamic>> chat({
    required String message,
    String? conversationId,
  }) async {
    final res = await _api.post(
      '/ai/chat',
      data: {
        'message': message,
        if (conversationId != null) 'conversation_id': conversationId,
      },
    );
    return res.data as Map<String, dynamic>;
  }

  Future<List<AiConversationModel>> listConversations() async {
    final res = await _api.get('/ai/conversations');
    return (res.data as List)
        .map((x) => AiConversationModel.fromJson(x))
        .toList();
  }

  Future<List<AiMessageModel>> getMessages(String conversationId) async {
    final res = await _api.get('/ai/conversations/$conversationId/messages');
    return (res.data as List).map((x) => AiMessageModel.fromJson(x)).toList();
  }
}

final aiRepositoryProvider = Provider<AiRepository>((ref) {
  return AiRepository(ref.watch(apiClientProvider));
});
