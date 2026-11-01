import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/features/ai/presentation/ai_screen.dart';

class _AiRepository implements AiRepository {
  final conversation = AiConversationModel(
    id: 'conversation-1',
    title: 'Family trip',
    createdAt: DateTime.utc(2026),
    updatedAt: DateTime.utc(2026),
  );
  final messages = [
    AiMessageModel(
      id: 'message-1',
      conversationId: 'conversation-1',
      role: 'user',
      content: 'Tell me about the trip.',
      createdAt: DateTime.utc(2026),
    ),
    AiMessageModel(
      id: 'message-2',
      conversationId: 'conversation-1',
      role: 'assistant',
      content: 'I found the family trip memory.',
      createdAt: DateTime.utc(2026),
    ),
  ];
  String? sentConversationId;

  @override
  Future<List<AiConversationModel>> listConversations() async => [conversation];

  @override
  Future<List<AiMessageModel>> getMessages(String conversationId) async =>
      messages;

  @override
  Future<Map<String, dynamic>> chat({
    required String message,
    String? conversationId,
  }) async {
    sentConversationId = conversationId;
    return {
      'conversation_id': conversation.id,
      'assistant_message': {
        'id': 'message-4',
        'conversation_id': conversation.id,
        'role': 'assistant',
        'content': 'It was a lovely trip.',
        'created_at': DateTime.utc(2026).toIso8601String(),
      },
    };
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  testWidgets('restores the latest chat and continues its conversation',
      (tester) async {
    final repository = _AiRepository();

    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          aiRepositoryProvider.overrideWithValue(repository),
          userProfileProvider.overrideWith(
            (ref) async => UserModel(id: 'user-1', email: 'user@example.test'),
          ),
        ],
        child: MaterialApp(
          theme: ThemeData.dark().copyWith(
            extensions: const [AppColors.dark],
          ),
          home: const AiScreen(),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('Tell me about the trip.'), findsOneWidget);
    expect(find.text('I found the family trip memory.'), findsOneWidget);

    await tester.enterText(find.byType(TextField), 'What happened next?');
    await tester.tap(find.byIcon(Icons.send_rounded));
    await tester.pumpAndSettle();

    expect(repository.sentConversationId, 'conversation-1');
    expect(find.text('It was a lovely trip.'), findsOneWidget);
  });
}
