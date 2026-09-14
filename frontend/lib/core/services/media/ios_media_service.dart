import 'dart:io';
import 'package:permission_handler/permission_handler.dart';
import 'package:memory_verse/core/services/media/media_service.dart';

class IOSMediaService implements MediaService {
  @override
  String get platformName => 'iOS';

  @override
  bool get isDesktopEnvironment => false;

  @override
  Future<MediaPermissionState> checkPermission() async {
    final status = await Permission.photos.status;
    if (status.isGranted) return MediaPermissionState.granted;
    if (status.isLimited) return MediaPermissionState.limited;
    if (status.isPermanentlyDenied) return MediaPermissionState.permanentlyDenied;
    if (status.isRestricted) return MediaPermissionState.restricted;
    if (status.isDenied) return MediaPermissionState.denied;
    return MediaPermissionState.notDetermined;
  }

  @override
  Future<MediaPermissionState> requestPermission() async {
    final status = await Permission.photos.request();
    if (status.isGranted) return MediaPermissionState.granted;
    if (status.isLimited) return MediaPermissionState.limited;
    if (status.isPermanentlyDenied) return MediaPermissionState.permanentlyDenied;
    if (status.isRestricted) return MediaPermissionState.restricted;
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
      '.jpg', '.jpeg', '.png', '.heic', '.mov', '.mp4'
    };

    // On iOS, sandboxed documents & media caches
    try {
      final docDir = Directory.current;
      if (docDir.existsSync()) {
        final entities = docDir.listSync(recursive: true, followLinks: false);
        for (final entity in entities) {
          if (items.length >= limit) break;
          if (entity is! File) continue;

          final ext = entity.path.toLowerCase().split('.').last;
          if (!allowedExtensions.contains('.$ext')) continue;

          final stat = entity.statSync();
          final isVideo = {'.mov', '.mp4'}.contains('.$ext');
          final modified = stat.modified;

          if (since != null && modified.isBefore(since)) continue;

          final fname = entity.path.split(Platform.pathSeparator).last;
          items.add(DiscoveredMediaItem(
            id: 'ios_${entity.path.hashCode}',
            type: isVideo ? MediaItemType.video : MediaItemType.image,
            mediaReference: entity.path,
            timestamp: modified,
            file: entity,
            thumbnailPath: isVideo ? null : entity.path,
            metadata: {
              'filename': fname,
              'file_size': stat.size,
              'extension': ext,
              'source': 'ios_photolibrary',
            },
          ));
        }
      }
    } catch (_) {}

    items.sort((a, b) => b.timestamp.compareTo(a.timestamp));
    return items.take(limit).toList();
  }

  @override
  Future<List<DiscoveredMediaItem>> queryMedia(MediaFilterQuery query) async {
    return await discoverMedia(limit: 300);
  }
}
