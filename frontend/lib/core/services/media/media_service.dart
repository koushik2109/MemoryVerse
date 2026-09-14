import 'dart:io';

enum MediaItemType { image, video }

enum MediaPermissionState {
  notDetermined,
  granted,
  limited,
  denied,
  restricted,
  permanentlyDenied,
}

class DiscoveredMediaItem {
  final String id;
  final MediaItemType type;
  final String mediaReference;
  final DateTime timestamp;
  final Map<String, dynamic> metadata;
  final String? thumbnailPath;
  final File? file;

  const DiscoveredMediaItem({
    required this.id,
    required this.type,
    required this.mediaReference,
    required this.timestamp,
    this.metadata = const {},
    this.thumbnailPath,
    this.file,
  });

  bool get isVideo => type == MediaItemType.video;
  String get filename => metadata['filename'] ?? mediaReference.split(Platform.pathSeparator).last;
  int get fileSize => metadata['file_size'] ?? (file != null ? (file!.existsSync() ? file!.lengthSync() : 0) : 0);
  String? get locationName => metadata['location_name'];
  List<String> get tags => List<String>.from(metadata['tags'] ?? []);

  DiscoveredMediaItem copyWith({
    String? id,
    MediaItemType? type,
    String? mediaReference,
    DateTime? timestamp,
    Map<String, dynamic>? metadata,
    String? thumbnailPath,
    File? file,
  }) {
    return DiscoveredMediaItem(
      id: id ?? this.id,
      type: type ?? this.type,
      mediaReference: mediaReference ?? this.mediaReference,
      timestamp: timestamp ?? this.timestamp,
      metadata: metadata ?? this.metadata,
      thumbnailPath: thumbnailPath ?? this.thumbnailPath,
      file: file ?? this.file,
    );
  }

  Map<String, dynamic> toCandidateJson() {
    return {
      'id': id,
      'filename': filename,
      'media_type': isVideo ? 'video' : 'image',
      'timestamp': timestamp.toIso8601String(),
      'location_name': locationName,
      'tags': tags,
      'file_size': fileSize,
      'duration': metadata['duration'],
      'width': metadata['width'],
      'height': metadata['height'],
    };
  }
}

class MediaFilterQuery {
  final String prompt;
  final DateTime? dateFrom;
  final DateTime? dateTo;
  final String? location;
  final List<String> keywords;

  const MediaFilterQuery({
    required this.prompt,
    this.dateFrom,
    this.dateTo,
    this.location,
    this.keywords = const [],
  });
}

abstract class MediaService {
  String get platformName;
  bool get isDesktopEnvironment;

  Future<MediaPermissionState> checkPermission();
  Future<MediaPermissionState> requestPermission();
  Future<bool> openSettings();

  Future<List<DiscoveredMediaItem>> discoverMedia({
    int limit = 200,
    DateTime? since,
  });

  Future<List<DiscoveredMediaItem>> queryMedia(MediaFilterQuery query);
}
