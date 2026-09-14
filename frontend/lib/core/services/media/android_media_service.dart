import 'dart:io';
import 'package:permission_handler/permission_handler.dart';
import 'package:memory_verse/core/services/media/media_service.dart';

class AndroidMediaService implements MediaService {
  @override
  String get platformName => 'Android';

  @override
  bool get isDesktopEnvironment => false;

  @override
  Future<MediaPermissionState> checkPermission() async {
    // On Android 13+ (SDK 33+), check photos and videos.
    // On Android 12 and below, check storage.
    final photosStatus = await Permission.photos.status;
    final videosStatus = await Permission.videos.status;
    final storageStatus = await Permission.storage.status;

    if (photosStatus.isGranted || videosStatus.isGranted || storageStatus.isGranted) {
      return MediaPermissionState.granted;
    }
    if (photosStatus.isLimited) {
      return MediaPermissionState.limited;
    }
    if (photosStatus.isPermanentlyDenied || storageStatus.isPermanentlyDenied) {
      return MediaPermissionState.permanentlyDenied;
    }
    if (photosStatus.isRestricted || storageStatus.isRestricted) {
      return MediaPermissionState.restricted;
    }
    if (photosStatus.isDenied && storageStatus.isDenied) {
      return MediaPermissionState.denied;
    }
    return MediaPermissionState.notDetermined;
  }

  @override
  Future<MediaPermissionState> requestPermission() async {
    // Request photos & videos permissions (handles Android 13+ and legacy storage gracefully)
    Map<Permission, PermissionStatus> statuses = await [
      Permission.photos,
      Permission.videos,
      Permission.storage,
    ].request();

    final photos = statuses[Permission.photos] ?? PermissionStatus.denied;
    final storage = statuses[Permission.storage] ?? PermissionStatus.denied;

    if (photos.isGranted || storage.isGranted) {
      return MediaPermissionState.granted;
    }
    if (photos.isLimited) {
      return MediaPermissionState.limited;
    }
    if (photos.isPermanentlyDenied || storage.isPermanentlyDenied) {
      return MediaPermissionState.permanentlyDenied;
    }
    if (photos.isRestricted || storage.isRestricted) {
      return MediaPermissionState.restricted;
    }
    return MediaPermissionState.denied;
  }

  @override
  Future<bool> openSettings() async {
    return await openAppSettings();
  }

  @override
  Future<List<DiscoveredMediaItem>> discoverMedia({
    int limit = 200,
    DateTime? since,
  }) async {
    final perm = await checkPermission();
    if (perm != MediaPermissionState.granted && perm != MediaPermissionState.limited) {
      return [];
    }

    final items = <DiscoveredMediaItem>[];
    final allowedExtensions = {
      '.jpg', '.jpeg', '.png', '.webp', '.heic', '.mp4', '.mov', '.3gp'
    };

    // Standard Android media directories
    final candidateDirs = [
      Directory('/storage/emulated/0/DCIM/Camera'),
      Directory('/storage/emulated/0/DCIM'),
      Directory('/storage/emulated/0/Pictures'),
      Directory('/storage/emulated/0/Movies'),
      Directory('/storage/emulated/0/Download'),
    ];

    for (final dir in candidateDirs) {
      if (!dir.existsSync()) continue;

      try {
        final entities = dir.listSync(recursive: false, followLinks: false);
        for (final entity in entities) {
          if (items.length >= limit) break;
          if (entity is! File) continue;

          final ext = entity.path.toLowerCase().split('.').last;
          if (!allowedExtensions.contains('.$ext')) continue;

          final stat = entity.statSync();
          final isVideo = {'.mp4', '.mov', '.3gp'}.contains('.$ext');
          final modified = stat.modified;

          if (since != null && modified.isBefore(since)) continue;

          final fname = entity.path.split(Platform.pathSeparator).last;
          items.add(DiscoveredMediaItem(
            id: 'android_${entity.path.hashCode}',
            type: isVideo ? MediaItemType.video : MediaItemType.image,
            mediaReference: entity.path,
            timestamp: modified,
            file: entity,
            thumbnailPath: isVideo ? null : entity.path,
            metadata: {
              'filename': fname,
              'file_size': stat.size,
              'extension': ext,
              'source': 'android_mediastore',
            },
          ));
        }
      } catch (_) {
        // Skip directories that cannot be read
      }
    }

    // Sort newest first
    items.sort((a, b) => b.timestamp.compareTo(a.timestamp));
    return items.take(limit).toList();
  }

  @override
  Future<List<DiscoveredMediaItem>> queryMedia(MediaFilterQuery query) async {
    final allMedia = await discoverMedia(limit: 300);
    return allMedia;
  }
}
