import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/providers/ai_director_controller.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';

class _FakeAiDirectorRepository implements AiDirectorRepository {
  final Map<String, Completer<AISessionModel>> pendingSessions = {};

  @override
  Future<AISessionModel> createOrGetSession(
    String memoryId, {
    String? initialPrompt,
  }) {
    return pendingSessions[memoryId]?.future ??
        Future.value(_testSessionFor(memoryId));
  }

  @override
  Future<AISessionModel> sendMessage(String sessionId, String message) async =>
      _testSessionFor(sessionId);

  @override
  Future<Map<String, dynamic>> approveAndGenerate(
    String sessionId, {
    VideoSpecificationModel? spec,
  }) async =>
      {'job_id': 'job-$sessionId'};

  @override
  Future<Map<String, dynamic>> requestRevision(
    String sessionId,
    String prompt,
  ) async =>
      {};
}

AISessionModel _testSessionFor(String memoryId) => AISessionModel(
      sessionId: 'session-$memoryId',
      memoryId: memoryId,
      title: 'Memory $memoryId',
      specification: const VideoSpecificationModel(),
    );

void main() {
  late ProviderContainer container;
  late _FakeAiDirectorRepository repository;
  late ProviderSubscription<AiDirectorState> subscription;

  setUp(() {
    repository = _FakeAiDirectorRepository();
    container = ProviderContainer(
      overrides: [aiDirectorRepositoryProvider.overrideWithValue(repository)],
    );
    subscription = container.listen(
      aiDirectorControllerProvider,
      (_, __) {},
      fireImmediately: true,
    );
  });

  tearDown(() {
    subscription.close();
    container.dispose();
  });

  test('clears the previous memory while a new session is loading', () async {
    final controller = container.read(aiDirectorControllerProvider.notifier);
    await controller.initSession('memory-a');
    expect(container.read(aiDirectorControllerProvider).session?.memoryId,
        'memory-a');

    final pending = Completer<AISessionModel>();
    repository.pendingSessions['memory-b'] = pending;
    final loadNewMemory = controller.initSession('memory-b');

    expect(container.read(aiDirectorControllerProvider).session, isNull);
    expect(container.read(aiDirectorControllerProvider).isLoading, isTrue);

    pending.completeError(StateError('offline'));
    await loadNewMemory;

    final state = container.read(aiDirectorControllerProvider);
    expect(state.session, isNull);
    expect(state.isLoading, isFalse);
    expect(state.error, contains('offline'));
  });

  test('an older session response cannot replace the newly selected memory',
      () async {
    final controller = container.read(aiDirectorControllerProvider.notifier);
    final memoryAResponse = Completer<AISessionModel>();
    final memoryBResponse = Completer<AISessionModel>();
    repository.pendingSessions['memory-a'] = memoryAResponse;
    repository.pendingSessions['memory-b'] = memoryBResponse;

    final loadMemoryA = controller.initSession('memory-a');
    final loadMemoryB = controller.initSession('memory-b');

    memoryBResponse.complete(_testSessionFor('memory-b'));
    await loadMemoryB;
    memoryAResponse.complete(_testSessionFor('memory-a'));
    await loadMemoryA;

    expect(container.read(aiDirectorControllerProvider).session?.memoryId,
        'memory-b');
  });
}
