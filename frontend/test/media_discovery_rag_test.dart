import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/api/api_client.dart';
import 'package:memory_verse/core/providers/upload_controller.dart';
import 'package:memory_verse/core/services/media/media_service.dart';
import 'package:memory_verse/core/services/media/windows_media_service.dart';
import 'package:memory_verse/core/services/rag/media_indexing_service.dart';
import 'package:memory_verse/core/services/rag/rag_retrieval_service.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';

class _FakeMediaService implements MediaService {
  final List<DiscoveredMediaItem> items;
  int? requestedLimit;

  _FakeMediaService(this.items);

  @override
  String get platformName => 'test';

  @override
  bool get isDesktopEnvironment => true;

  @override
  Future<MediaPermissionState> checkPermission() async =>
      MediaPermissionState.granted;

  @override
  Future<MediaPermissionState> requestPermission() async =>
      MediaPermissionState.granted;

  @override
  Future<bool> openSettings() async => true;

  @override
  Future<bool> openLimitedAccessPicker() async => true;

  @override
  Future<List<DiscoveredMediaItem>> discoverMedia({
    int? limit,
    DateTime? since,
  }) async {
    requestedLimit = limit;
    final discovered = since == null
        ? items
        : items.where((item) => !item.timestamp.isBefore(since)).toList();
    return limit == null ? discovered : discovered.take(limit).toList();
  }

  @override
  Future<List<DiscoveredMediaItem>> queryMedia(MediaFilterQuery query) async =>
      items;
}

class _RecordingAiRepository extends AiRepository {
  final List<List<File>> curatedBatches = [];
  final List<List<Map<String, dynamic>>> candidateBatches = [];

  _RecordingAiRepository() : super(ApiClient());

  @override
  Future<Map<String, dynamic>> curateLocalMedia({
    required String prompt,
    required List<File> files,
    List<Map<String, dynamic>>? candidateMetadata,
    String? title,
    String? description,
    double threshold = 0.50,
  }) async {
    curatedBatches.add(files);
    return {
      'matched_media': files.map((file) {
        final name = file.path.split(Platform.pathSeparator).last;
        return {
          'filename': name,
          'relevance_score': _scoreFor(name),
        };
      }).toList(),
    };
  }

  @override
  Future<Map<String, dynamic>> filterMedia({
    required String prompt,
    required List<Map<String, dynamic>> candidates,
    int topK = 30,
    double threshold = 0.15,
  }) async {
    candidateBatches.add(candidates);
    return {
      'matched_media': candidates.map((candidate) {
        final id = candidate['id'].toString();
        return {
          'id': id,
          'filename': candidate['filename'],
          'relevance_score': _scoreFor(id),
        };
      }).toList(),
    };
  }

  double _scoreFor(String value) {
    final match = RegExp(r'(\d+)').firstMatch(value);
    return (int.tryParse(match?.group(1) ?? '0') ?? 0) / 100.0;
  }
}

class _RecordingApiClient extends ApiClient {
  Map<String, dynamic>? capturedQuery;

  @override
  Future<Response<T>> post<T>(
    String path, {
    dynamic data,
    Map<String, dynamic>? queryParameters,
  }) async {
    capturedQuery = queryParameters;
    return Response<T>(
      requestOptions: RequestOptions(path: path),
      data: {'job_id': 'job_1'} as T,
    );
  }
}

class _FakeUploadRepository extends MediaRepository {
  var _uploadCount = 0;

  _FakeUploadRepository() : super(ApiClient());

  @override
  Future<MediaModel> uploadMedia({
    required File file,
    required String filename,
    required String mediaType,
    String? vaultId,
    String? memoryId,
    CancelToken? cancelToken,
    ProgressCallback? onSendProgress,
  }) async {
    _uploadCount += 1;
    return MediaModel(
      id: 'uploaded_$_uploadCount',
      ownerId: 'user_1',
      filename: filename,
      storagePath: 'user_1/$filename',
      url: 'https://example.com/$filename',
      fileSize: 1,
      createdAt: DateTime(2025, 1, 1),
    );
  }
}

List<DiscoveredMediaItem> _makeItems(
  int count, {
  Directory? directory,
  bool withFiles = false,
}) {
  return List.generate(count, (index) {
    final filename = 'media_${index.toString().padLeft(3, '0')}.jpg';
    final path = directory == null
        ? '/unavailable/$filename'
        : '${directory.path}${Platform.pathSeparator}$filename';
    final file = withFiles ? File(path) : null;
    return DiscoveredMediaItem(
      id: 'item_$index',
      type: MediaItemType.image,
      mediaReference: path,
      timestamp: DateTime(2025, 1, 1).add(Duration(minutes: index)),
      file: file,
      metadata: {
        'filename': filename,
        'folder': 'Goa Beach',
        'relative_path': filename,
        'location_name': 'Goa Beach',
        'tags': ['beach'],
      },
    );
  });
}

void main() {
  dotenv.testLoad(fileInput: 'API_BASE_URL=http://127.0.0.1:8000');

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
          'folder': 'Goa Beach',
          'relative_path': 'Goa Beach/goa_trip_2025.jpg',
          'latitude': 15.4909,
          'longitude': 73.8278,
          'tags': ['beach', 'sunset'],
          'file_size': 1024,
        },
      );

      final json = item.toCandidateJson();
      expect(json['id'], 'test_1');
      expect(json['filename'], 'goa_trip_2025.jpg');
      expect(json['location_name'], 'Goa Beach');
      expect(json['location'], 'Goa Beach');
      expect(json['latitude'], 15.4909);
      expect(json['longitude'], 73.8278);
      expect(json['folder'], 'Goa Beach');
      expect(json['relative_path'], 'Goa Beach/goa_trip_2025.jpg');
      expect((json['metadata'] as Map)['location_name'], 'Goa Beach');
      expect(json['media_type'], 'image');
    });

    test('WindowsMediaService initialization and permission', () async {
      final winService = WindowsMediaService();
      final status = await winService.checkPermission();
      expect(status, MediaPermissionState.granted);
      expect(winService.isDesktopEnvironment, true);
    });

    test('full sync accumulates every discovered item without a default cap',
        () async {
      final mediaService = _FakeMediaService(_makeItems(450));
      final indexService = MediaIndexingService(mediaService);

      final indexed = await indexService.syncMediaIndex(forceFullScan: true);

      expect(mediaService.requestedLimit, isNull);
      expect(indexed, hasLength(450));
      expect(indexService.indexedCount, 450);
    });

    test('metadata candidates are batched and selected globally by topK',
        () async {
      final mediaService = _FakeMediaService(_makeItems(85));
      final indexService = MediaIndexingService(mediaService);
      final aiRepository = _RecordingAiRepository();
      final service = RagRetrievalService(indexService, aiRepository);

      final result = await service.retrieveMemoriesForPrompt(
        prompt: 'Goa beach',
        topK: 5,
      );

      expect(aiRepository.candidateBatches, hasLength(3));
      expect(aiRepository.candidateBatches.every((batch) => batch.length <= 40),
          isTrue);
      expect(
        aiRepository.candidateBatches.expand((batch) => batch),
        hasLength(85),
      );
      expect(result.totalScanned, 85);
      expect(result.matches, hasLength(5));
      expect(result.matches.map((match) => match.item.id), [
        'item_84',
        'item_83',
        'item_82',
        'item_81',
        'item_80',
      ]);
    });

    test('local files are all sent in bounded visual batches', () async {
      final directory =
          await Directory.systemTemp.createTemp('memoryverse_media_test');
      try {
        final items = _makeItems(85, directory: directory, withFiles: true);
        for (final item in items) {
          await item.file!.writeAsString('test');
        }

        final mediaService = _FakeMediaService(items);
        final aiRepository = _RecordingAiRepository();
        final service = RagRetrievalService(
          MediaIndexingService(mediaService),
          aiRepository,
        );

        final result = await service.retrieveMemoriesForPrompt(
          prompt: 'Goa beach',
          topK: 3,
        );

        expect(aiRepository.curatedBatches, hasLength(3));
        expect(aiRepository.curatedBatches.every((batch) => batch.length <= 40),
            isTrue);
        expect(
          aiRepository.curatedBatches.expand((batch) => batch),
          hasLength(85),
        );
        expect(result.matches.map((match) => match.item.filename), [
          'media_084.jpg',
          'media_083.jpg',
          'media_082.jpg',
        ]);
      } finally {
        await directory.delete(recursive: true);
      }
    });

    test('generateVideo sends the completed backend media IDs', () async {
      final api = _RecordingApiClient();
      final repository = MediaRepository(api);

      final jobId = await repository.generateVideo(
        'memory_1',
        mediaIds: ['media_a', 'media_b'],
      );

      expect(jobId, 'job_1');
      expect(api.capturedQuery?['media_ids'], ['media_a', 'media_b']);
    });

    test('upload completion exposes backend media IDs for video generation',
        () async {
      final directory =
          await Directory.systemTemp.createTemp('memoryverse_upload_test');
      final container = ProviderContainer(
        overrides: [
          mediaRepositoryProvider.overrideWithValue(_FakeUploadRepository()),
        ],
      );

      try {
        final first = File('${directory.path}${Platform.pathSeparator}one.jpg');
        final second =
            File('${directory.path}${Platform.pathSeparator}two.jpg');
        await first.writeAsString('one');
        await second.writeAsString('two');

        final controller = container.read(uploadControllerProvider.notifier);
        final taskIds = controller.addUploads(
          files: [first, second],
          mediaTypes: ['image', 'image'],
          memoryId: 'memory_1',
        );

        expect(await controller.waitForUploads(taskIds), isTrue);
        expect(controller.completedMediaIds(taskIds), [
          'uploaded_1',
          'uploaded_2',
        ]);
      } finally {
        container.dispose();
        await directory.delete(recursive: true);
      }
    });
  });
}
