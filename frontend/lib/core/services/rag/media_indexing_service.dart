import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/core/services/media/media_service.dart';
import 'package:memory_verse/core/services/media/media_service_provider.dart';

class MediaIndexEntry {
  final DiscoveredMediaItem item;
  final DateTime indexedAt;

  const MediaIndexEntry({
    required this.item,
    required this.indexedAt,
  });
}

class MediaIndexingService {
  final MediaService _mediaService;
  final Map<String, MediaIndexEntry> _index = {};
  DateTime? _lastFullScanTime;

  MediaIndexingService(this._mediaService);

  bool get isIndexed => _index.isNotEmpty;
  int get indexedCount => _index.length;
  DateTime? get lastScanTime => _lastFullScanTime;

  List<DiscoveredMediaItem> get allIndexedItems =>
      _index.values.map((e) => e.item).toList();

  /// Performs an incremental scan. If never scanned, runs a full discovery.
  /// Otherwise, discovers items modified since the last scan.
  Future<List<DiscoveredMediaItem>> syncMediaIndex({
    bool forceFullScan = false,
    int limit = 300,
  }) async {
    final since = forceFullScan ? null : _lastFullScanTime;
    final newlyDiscovered = await _mediaService.discoverMedia(
      limit: limit,
      since: since,
    );

    final now = DateTime.now();
    for (final item in newlyDiscovered) {
      _index[item.id] = MediaIndexEntry(
        item: item,
        indexedAt: now,
      );
    }

    _lastFullScanTime = now;
    
    // Sort all indexed items newest first
    final sorted = _index.values.map((e) => e.item).toList();
    sorted.sort((a, b) => b.timestamp.compareTo(a.timestamp));
    return sorted;
  }

  void clearIndex() {
    _index.clear();
    _lastFullScanTime = null;
  }
}

final mediaIndexingServiceProvider = Provider<MediaIndexingService>((ref) {
  final mediaService = ref.watch(mediaServiceProvider);
  return MediaIndexingService(mediaService);
});
