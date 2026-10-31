import 'dart:async';
import 'dart:io';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/providers/home_assistant_controller.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';

class _FakeAiRepository implements AiRepository {
  final Map<String, Completer<Map<String, dynamic>>> pendingReplies = {};
  final Map<String, Map<String, dynamic>> replies = {};

  @override
  Future<Map<String, dynamic>> chat({
    required String message,
    String? conversationId,
  }) async =>
      throw UnimplementedError();

  @override
  Future<List<AiConversationModel>> listConversations() async =>
      throw UnimplementedError();

  @override
  Future<List<AiMessageModel>> getMessages(String conversationId) async =>
      throw UnimplementedError();

  @override
  Future<Map<String, dynamic>> filterMedia({
    required String prompt,
    required List<Map<String, dynamic>> candidates,
    int topK = 30,
    double threshold = 0.15,
  }) async =>
      throw UnimplementedError();

  @override
  Future<Map<String, dynamic>> curateLocalMedia({
    required String prompt,
    required List<File> files,
    List<Map<String, dynamic>>? candidateMetadata,
    String? title,
    String? description,
    double threshold = 0.50,
  }) async =>
      throw UnimplementedError();

  @override
  Future<Map<String, dynamic>> askHomeAssistant({
    required String message,
    Map<String, dynamic>? clientContext,
  }) async {
    final pending = pendingReplies[message];
    if (pending != null) return pending.future;
    return replies[message] ?? _reply('No matching action.', action: 'NONE');
  }
}

Map<String, dynamic> _reply(
  String message, {
  required String action,
  Map<String, dynamic> params = const {},
}) =>
    {
      'message': message,
      'intent': action,
      'structured_action': {'action': action, 'params': params},
    };

void main() {
  late ProviderContainer container;
  late _FakeAiRepository repository;
  late ProviderSubscription<HomeAssistantState> subscription;

  setUp(() {
    repository = _FakeAiRepository();
    container = ProviderContainer(
      overrides: [aiRepositoryProvider.overrideWithValue(repository)],
    );
    subscription = container.listen(
      homeAssistantProvider,
      (_, __) {},
      fireImmediately: true,
    );
  });

  tearDown(() {
    subscription.close();
    container.dispose();
  });

  test('clears an old action while a new query is pending', () async {
    final controller = container.read(homeAssistantProvider.notifier);
    repository.replies['open'] = _reply(
      'Opening memory.',
      action: 'OPEN_MEMORY',
      params: {'memory_id': 'memory-a'},
    );
    await controller.submitQuery('open');
    expect(container.read(homeAssistantProvider).action, 'OPEN_MEMORY');

    final pendingReply = Completer<Map<String, dynamic>>();
    repository.pendingReplies['help'] = pendingReply;
    final nextQuery = controller.submitQuery('help');
    final loadingState = container.read(homeAssistantProvider);
    expect(loadingState.action, isNull);
    expect(loadingState.actionParams, isEmpty);
    expect(loadingState.responseMessage, isNull);

    pendingReply.complete(_reply('Here is some help.', action: 'NONE'));
    await nextQuery;
    expect(container.read(homeAssistantProvider).action, isNull);
  });

  test('ignores an older response after a newer query has completed', () async {
    final controller = container.read(homeAssistantProvider.notifier);
    final olderReply = Completer<Map<String, dynamic>>();
    repository.pendingReplies['older'] = olderReply;
    repository.replies['latest'] = _reply('Latest answer.', action: 'NONE');

    final olderQuery = controller.submitQuery('older');
    await controller.submitQuery('latest');
    olderReply.complete(_reply(
      'Stale action.',
      action: 'OPEN_MEMORY',
      params: {'memory_id': 'memory-stale'},
    ));
    await olderQuery;

    final state = container.read(homeAssistantProvider);
    expect(state.query, 'latest');
    expect(state.responseMessage, 'Latest answer.');
    expect(state.action, isNull);
  });

  test('clearing the assistant invalidates its pending response', () async {
    final controller = container.read(homeAssistantProvider.notifier);
    final pendingReply = Completer<Map<String, dynamic>>();
    repository.pendingReplies['pending'] = pendingReply;

    final query = controller.submitQuery('pending');
    controller.clear();
    pendingReply.complete(_reply('Late answer.', action: 'NONE'));
    await query;

    final state = container.read(homeAssistantProvider);
    expect(state.query, isNull);
    expect(state.responseMessage, isNull);
    expect(state.isLoading, isFalse);
  });
}
