import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/features/memories/presentation/widgets/media_permission_view.dart';

void main() {
  testWidgets('MediaPermissionView renders successfully in ProviderScope', (WidgetTester tester) async {
    await tester.pumpWidget(
      ProviderScope(
        child: MaterialApp(
          theme: ThemeData.dark().copyWith(
            extensions: const [AppColors.dark],
          ),
          home: const Scaffold(
            body: MediaPermissionView(),
          ),
        ),
      ),
    );

    // Verify widget builds without exceptions
    expect(find.byType(MediaPermissionView), findsOneWidget);
  });
}


