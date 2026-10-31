import 'dart:io';

import 'package:photo_manager/photo_manager.dart';
import 'package:memory_verse/core/services/media/media_service.dart';

class IOSMediaService implements MediaService {
  @override
  String get platformName => 'iOS';

  @override
  bool get isDesktopEnvironment => false;

  @override
  Future<MediaPermissionState> checkPermission() async {
    return _mapPermission(await PhotoManager.getPermissionState(
      requestOption: const PermissionRequestOption(),
    ));
  }

  @override
  Future<MediaPermissionState> requestPermission() async {
    return _mapPermission(await PhotoManager.requestPermissionExtend());
  }

  @override
  Future<bool> openLimitedAccessPicker() async {
    try {
      await PhotoManager.presentLimited(type: RequestType.common);
      return true;
    } catch (_) {
      return false;
    }
  }

  MediaPermissionState _mapPermission(PermissionState state) {
    switch (state) {
      case PermissionState.authorized:
        return MediaPermissionState.granted;
      case PermissionState.limited:
        return MediaPermissionState.limited;
      case PermissionState.restricted:
        return MediaPermissionState.restricted;
      case PermissionState.denied:
        return MediaPermissionState.denied;
      case PermissionState.notDetermined:
        return MediaPermissionState.notDetermined;
    }
  }

  @override
  Future<bool> openSettings() async {
    await PhotoManager.openSetting();
    return true;
  }

  @override
  Future<List<DiscoveredMediaItem>> discoverMedia({
    int? limit,
    DateTime? since,
  }) async {
    final perm = await checkPermission();
    if (!perm.hasMediaAccess) {
      return [];
    }

    final items = <DiscoveredMediaItem>[];
    try {
      final paths = await PhotoManager.getAssetPathList(
        type: RequestType.common,
        hasAll: true,
        onlyAll: true,
      );
      if (paths.isEmpty) return [];

      const pageSize = 100;
      for (var page = 0;; page++) {
        if (limit != null && items.length >= limit) break;
        final entities = await paths.first.getAssetListPaged(
          page: page,
          size: pageSize,
        );
        if (entities.isEmpty) break;

        final eligible = entities
            .where((asset) =>
                since == null || !asset.createDateTime.isBefore(since))
            .toList();
        final remaining =
            limit == null ? eligible.length : limit - items.length;
        const conversionBatchSize = 8;
        for (var start = 0; start < eligible.length && start < remaining;) {
          var end = start + conversionBatchSize;
          if (end > eligible.length) end = eligible.length;
          if (end > remaining) end = remaining;
          final batch = eligible.sublist(start, end);
          final converted = await Future.wait(batch.map((asset) async {
            try {
              return await _toDiscoveredItem(asset, asset.createDateTime);
            } catch (_) {
              // A single iCloud/unavailable asset must not abort the full scan.
              return null;
            }
          }));
          items.addAll(converted.whereType<DiscoveredMediaItem>());
          start = end;
        }
      }
    } catch (_) {}

    items.sort((a, b) => b.timestamp.compareTo(a.timestamp));
    return limit == null ? items : items.take(limit).toList();
  }

  @override
  Future<List<DiscoveredMediaItem>> queryMedia(MediaFilterQuery query) async {
    return await discoverMedia();
  }

  Future<DiscoveredMediaItem> _toDiscoveredItem(
    AssetEntity asset,
    DateTime created,
  ) async {
    File? file;
    try {
      file = await asset.file;
    } catch (_) {
      // Keep the metadata-only item when the original is not locally available.
    }

    int? fileSize;
    try {
      fileSize = await asset.fileSize;
    } catch (_) {}

    var filename = asset.title;
    if (filename == null || filename.isEmpty) {
      try {
        final asyncTitle = await asset.titleAsync;
        if (asyncTitle.isNotEmpty) filename = asyncTitle;
      } catch (_) {}
    }

    // Use the current `latLng` API; the async fallback also handles assets
    // whose location is resolved lazily by the native PhotoManager plugin.
    var location = asset.latLng;
    try {
      location ??= await asset.latlngAsync();
    } catch (_) {}

    final isVideo = asset.type == AssetType.video;
    return DiscoveredMediaItem(
      id: 'ios_asset_${asset.id}',
      type: isVideo ? MediaItemType.video : MediaItemType.image,
      mediaReference: file?.path ?? 'asset://${asset.id}',
      timestamp: created,
      file: file,
      thumbnailPath: file?.path,
      metadata: {
        'filename': filename ?? 'asset_${asset.id}',
        'file_size': fileSize ?? 0,
        'width': asset.width,
        'height': asset.height,
        'duration': asset.duration,
        'relative_path': asset.relativePath,
        'asset_id': asset.id,
        'latitude': location?.latitude,
        'longitude': location?.longitude,
        'source': 'ios_photolibrary',
      },
    );
  }
}
