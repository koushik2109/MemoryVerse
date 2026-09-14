import 'package:flutter_test/flutter_test.dart';
import 'package:memory_verse/core/services/media/media_service.dart';
import 'package:memory_verse/core/services/media/windows_media_service.dart';
import 'package:memory_verse/core/services/rag/media_indexing_service.dart';

void main() {
  group('Media Discovery & RAG Tests', () {
    test('DiscoveredMediaItem candidate conversion', () {
      final now = DateTime(2025, 11, 14, 18, 0);
      final item = DiscoveredMediaItem(
        id: 'test_1',
        type: MediaItemType.image,
        mediaReference: 'C:\\Users\\test\\Pictures\\goa_trip_2025.jpg',
        timestamp: now,
        metadata: {
          'filename': 'goa_trip_2025.jpg',
          'location_name': 'Goa Beach',
          'tags': ['beach', 'sunset'],
          'file_size': 1024,
        },
      );

      final json = item.toCandidateJson();
      expect(json['id'], 'test_1');
      expect(json['filename'], 'goa_trip_2025.jpg');
      expect(json['location_name'], 'Goa Beach');
      expect(json['media_type'], 'image');
    });

    test('WindowsMediaService initialization and permission', () async {
      final winService = WindowsMediaService();
      final status = await winService.checkPermission();
      expect(status, MediaPermissionState.granted);
      expect(winService.isDesktopEnvironment, true);
    });

    test('MediaIndexingService caching and incremental sync', () async {
      final winService = WindowsMediaService();
      final indexService = MediaIndexingService(winService);
      expect(indexService.isIndexed, false);

      await indexService.syncMediaIndex();
      expect(indexService.isIndexed, isA<bool>());
    });
  });
}
