class UserModel {
  final String id;
  final String email;
  final String? fullName;
  final String? username;
  final String? avatarUrl;
  final String? bio;
  final int vaultCount;
  final int mediaCount;
  final DateTime? createdAt;
  final DateTime? updatedAt;

  UserModel({
    required this.id,
    required this.email,
    this.fullName,
    this.username,
    this.avatarUrl,
    this.bio,
    this.vaultCount = 0,
    this.mediaCount = 0,
    this.createdAt,
    this.updatedAt,
  });

  factory UserModel.fromJson(Map<String, dynamic> json) {
    return UserModel(
      id: json['id'] ?? '',
      email: json['email'] ?? '',
      fullName: json['full_name'],
      username: json['username'],
      avatarUrl: json['avatar_url'],
      bio: json['bio'],
      vaultCount: json['vault_count'] ?? 0,
      mediaCount: json['media_count'] ?? 0,
      createdAt: json['created_at'] != null ? DateTime.tryParse(json['created_at']) : null,
      updatedAt: json['updated_at'] != null ? DateTime.tryParse(json['updated_at']) : null,
    );
  }
}

class VaultMemberModel {
  final String id;
  final String userId;
  final String? fullName;
  final String? email;
  final String? avatarUrl;
  final String role; // owner, editor, viewer
  final DateTime? joinedAt;

  VaultMemberModel({
    required this.id,
    required this.userId,
    this.fullName,
    this.email,
    this.avatarUrl,
    this.role = 'editor',
    this.joinedAt,
  });

  factory VaultMemberModel.fromJson(Map<String, dynamic> json) {
    return VaultMemberModel(
      id: json['id'] ?? '',
      userId: json['user_id'] ?? '',
      fullName: json['full_name'],
      email: json['email'],
      avatarUrl: json['avatar_url'],
      role: json['role'] ?? 'editor',
      joinedAt: json['joined_at'] != null ? DateTime.tryParse(json['joined_at']) : null,
    );
  }
}

class VaultModel {
  final String id;
  final String name;
  final String? description;
  final String? coverImageUrl;
  final bool isArchived;
  final String ownerId;
  final DateTime createdAt;
  final DateTime updatedAt;
  final int memberCount;
  final int mediaCount;
  final String? inviteCode;
  final List<VaultMemberModel> members;

  VaultModel({
    required this.id,
    required this.name,
    this.description,
    this.coverImageUrl,
    this.isArchived = false,
    required this.ownerId,
    required this.createdAt,
    required this.updatedAt,
    this.memberCount = 1,
    this.mediaCount = 0,
    this.inviteCode,
    this.members = const [],
  });

  factory VaultModel.fromJson(Map<String, dynamic> json) {
    return VaultModel(
      id: json['id'] ?? '',
      name: json['name'] ?? 'Vault',
      description: json['description'],
      coverImageUrl: json['cover_image_url'],
      isArchived: json['is_archived'] ?? false,
      ownerId: json['owner_id'] ?? '',
      createdAt: DateTime.tryParse(json['created_at'] ?? '') ?? DateTime.now(),
      updatedAt: DateTime.tryParse(json['updated_at'] ?? '') ?? DateTime.now(),
      memberCount: json['member_count'] ?? 1,
      mediaCount: json['media_count'] ?? 0,
      inviteCode: json['invite_code'],
      members: (json['members'] as List<dynamic>?)
              ?.map((m) => VaultMemberModel.fromJson(m))
              .toList() ??
          [],
    );
  }
}
class MemoryModel {
  final String id;
  final String ownerId;
  final String? vaultId;
  final String title;
  final String? description;
  final String? coverMediaId;
  final DateTime memoryDate;
  final String? locationName;
  final DateTime createdAt;
  final DateTime updatedAt;
  final int mediaCount;
  final List<MediaModel> media;

  MemoryModel({
    required this.id,
    required this.ownerId,
    this.vaultId,
    required this.title,
    this.description,
    this.coverMediaId,
    required this.memoryDate,
    this.locationName,
    required this.createdAt,
    required this.updatedAt,
    this.mediaCount = 0,
    this.media = const [],
  });

  factory MemoryModel.fromJson(Map<String, dynamic> json) {
    return MemoryModel(
      id: json['id'] ?? '',
      ownerId: json['owner_id'] ?? '',
      vaultId: json['vault_id'],
      title: json['title'] ?? 'Untitled Memory',
      description: json['description'],
      coverMediaId: json['cover_media_id'],
      memoryDate: DateTime.tryParse(json['memory_date'] ?? '') ?? DateTime.now(),
      locationName: json['location_name'],
      createdAt: DateTime.tryParse(json['created_at'] ?? '') ?? DateTime.now(),
      updatedAt: DateTime.tryParse(json['updated_at'] ?? '') ?? DateTime.now(),
      mediaCount: json['media_count'] ?? 0,
      media: (json['media'] as List<dynamic>?)
              ?.map((m) => MediaModel.fromJson(m))
              .toList() ??
          [],
    );
  }
}
class MediaModel {
  final String id;
  final String? vaultId;
  final String? memoryId;
  final String ownerId;
  final String filename;
  final String storagePath;
  final String url;
  final String? thumbnailUrl;
  final String mediaType; // image, video
  final int fileSize;
  final String? mimeType;
  final DateTime createdAt;

  MediaModel({
    required this.id,
    this.vaultId,
    this.memoryId,
    required this.ownerId,
    required this.filename,
    required this.storagePath,
    required this.url,
    this.thumbnailUrl,
    this.mediaType = 'image',
    required this.fileSize,
    this.mimeType,
    required this.createdAt,
  });

  bool get isVideo => mediaType == 'video';

  factory MediaModel.fromJson(Map<String, dynamic> json) {
    return MediaModel(
      id: json['id'] ?? '',
      vaultId: json['vault_id'],
      memoryId: json['memory_id'],
      ownerId: json['owner_id'] ?? '',
      filename: json['filename'] ?? '',
      storagePath: json['storage_path'] ?? '',
      url: json['url'] ?? '',
      thumbnailUrl: json['thumbnail_url'],
      mediaType: json['media_type'] ?? 'image',
      fileSize: json['file_size'] ?? 0,
      mimeType: json['mime_type'],
      createdAt: DateTime.tryParse(json['created_at'] ?? '') ?? DateTime.now(),
    );
  }
}

class InviteModel {
  final String inviteCode;
  final String inviteLink;
  final String vaultId;
  final String vaultName;
  final String? vaultDescription;
  final String? ownerName;
  final int memberCount;

  InviteModel({
    required this.inviteCode,
    required this.inviteLink,
    required this.vaultId,
    required this.vaultName,
    this.vaultDescription,
    this.ownerName,
    this.memberCount = 0,
  });

  factory InviteModel.fromJson(Map<String, dynamic> json) {
    return InviteModel(
      inviteCode: json['invite_code'] ?? '',
      inviteLink: json['invite_link'] ?? '',
      vaultId: json['vault_id'] ?? '',
      vaultName: json['vault_name'] ?? '',
      vaultDescription: json['vault_description'],
      ownerName: json['owner_name'],
      memberCount: json['member_count'] ?? 0,
    );
  }
}

class NotificationModel {
  final String id;
  final String title;
  final String message;
  final String type;
  final Map<String, dynamic> data;
  final bool isRead;
  final DateTime createdAt;

  NotificationModel({
    required this.id,
    required this.title,
    required this.message,
    required this.type,
    this.data = const {},
    this.isRead = false,
    required this.createdAt,
  });

  factory NotificationModel.fromJson(Map<String, dynamic> json) {
    return NotificationModel(
      id: json['id'] ?? '',
      title: json['title'] ?? '',
      message: json['message'] ?? '',
      type: json['type'] ?? 'general',
      data: json['data'] as Map<String, dynamic>? ?? {},
      isRead: json['is_read'] ?? false,
      createdAt: DateTime.tryParse(json['created_at'] ?? '') ?? DateTime.now(),
    );
  }
}

// ── Timeline Models ──────────────────────────────────────────────────────────

class TimelineMediaItem {
  final String id;
  final String url;
  final String? thumbnailUrl;
  final String mediaType;
  final String filename;
  final DateTime createdAt;

  TimelineMediaItem({
    required this.id,
    required this.url,
    this.thumbnailUrl,
    required this.mediaType,
    required this.filename,
    required this.createdAt,
  });

  factory TimelineMediaItem.fromJson(Map<String, dynamic> json) {
    return TimelineMediaItem(
      id: json['id'] ?? '',
      url: json['url'] ?? '',
      thumbnailUrl: json['thumbnail_url'],
      mediaType: json['media_type'] ?? 'image',
      filename: json['filename'] ?? '',
      createdAt: DateTime.tryParse(json['created_at'] ?? '') ?? DateTime.now(),
    );
  }
}

class TimelineDayGroup {
  final String dateLabel;
  final DateTime date;
  final List<MemoryModel> memories;

  TimelineDayGroup({
    required this.dateLabel,
    required this.date,
    this.memories = const [],
  });

  factory TimelineDayGroup.fromJson(Map<String, dynamic> json) {
    return TimelineDayGroup(
      dateLabel: json['date_label'] ?? '',
      date: DateTime.tryParse(json['date'] ?? '') ?? DateTime.now(),
      memories: (json['memories'] as List<dynamic>?)
              ?.map((m) => MemoryModel.fromJson(m))
              .toList() ??
          [],
    );
  }
}

class TimelineMonthGroup {
  final String month;
  final List<TimelineDayGroup> days;

  TimelineMonthGroup({required this.month, this.days = const []});

  factory TimelineMonthGroup.fromJson(Map<String, dynamic> json) {
    return TimelineMonthGroup(
      month: json['month'] ?? '',
      days: (json['days'] as List<dynamic>?)
              ?.map((d) => TimelineDayGroup.fromJson(d))
              .toList() ??
          [],
    );
  }
}

class TimelineYearGroup {
  final String year;
  final List<TimelineMonthGroup> months;

  TimelineYearGroup({required this.year, this.months = const []});

  factory TimelineYearGroup.fromJson(Map<String, dynamic> json) {
    return TimelineYearGroup(
      year: json['year'] ?? '',
      months: (json['months'] as List<dynamic>?)
              ?.map((m) => TimelineMonthGroup.fromJson(m))
              .toList() ??
          [],
    );
  }
}

class TimelineResponse {
  final List<TimelineYearGroup> groups;
  final int totalItems;

  TimelineResponse({this.groups = const [], this.totalItems = 0});

  factory TimelineResponse.fromJson(Map<String, dynamic> json) {
    return TimelineResponse(
      groups: (json['groups'] as List<dynamic>?)
              ?.map((g) => TimelineYearGroup.fromJson(g))
              .toList() ??
          [],
      totalItems: json['total_items'] ?? 0,
    );
  }
}

// ── AI / Chat Models ─────────────────────────────────────────────────────────

class AiMessageModel {
  final String id;
  final String conversationId;
  final String role; // user | assistant
  final String content;
  final List<String> relatedMemoryIds;
  final DateTime createdAt;

  AiMessageModel({
    required this.id,
    required this.conversationId,
    required this.role,
    required this.content,
    this.relatedMemoryIds = const [],
    required this.createdAt,
  });

  bool get isUser => role == 'user';

  factory AiMessageModel.fromJson(Map<String, dynamic> json) {
    return AiMessageModel(
      id: json['id'] ?? '',
      conversationId: json['conversation_id'] ?? '',
      role: json['role'] ?? 'user',
      content: json['content'] ?? '',
      relatedMemoryIds: List<String>.from(json['related_memory_ids'] ?? []),
      createdAt: DateTime.tryParse(json['created_at'] ?? '') ?? DateTime.now(),
    );
  }
}

class AiConversationModel {
  final String id;
  final String title;
  final DateTime createdAt;
  final DateTime updatedAt;
  final int messageCount;

  AiConversationModel({
    required this.id,
    required this.title,
    required this.createdAt,
    required this.updatedAt,
    this.messageCount = 0,
  });

  factory AiConversationModel.fromJson(Map<String, dynamic> json) {
    return AiConversationModel(
      id: json['id'] ?? '',
      title: json['title'] ?? '',
      createdAt: DateTime.tryParse(json['created_at'] ?? '') ?? DateTime.now(),
      updatedAt: DateTime.tryParse(json['updated_at'] ?? '') ?? DateTime.now(),
      messageCount: json['message_count'] ?? 0,
    );
  }
}


class VideoJobModel {
  final String id;
  final String status;
  final String? resultUrl;
  final String? errorMessage;

  VideoJobModel({required this.id, required this.status, this.resultUrl, this.errorMessage});

  factory VideoJobModel.fromJson(Map<String, dynamic> json) {
    return VideoJobModel(
      id: json['id'] ?? '',
      status: json['status'] ?? 'pending',
      resultUrl: json['result_url'],
      errorMessage: json['error_message'],
    );
  }
}
