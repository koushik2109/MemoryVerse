import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/services/media/media_service.dart';
import 'package:memory_verse/core/services/media/media_service_provider.dart';
import 'package:memory_verse/features/memories/presentation/widgets/media_permission_view.dart';

class _LimitedMediaService implements MediaService {
  bool limitedPickerOpened = false;

  @override
  String get platformName => 'test';

  @override
  bool get isDesktopEnvironment => false;

  @override
  Future<MediaPermissionState> checkPermission() async =>
      MediaPermissionState.limited;

  @override
  Future<MediaPermissionState> requestPermission() async =>
      MediaPermissionState.limited;

  @override
  Future<bool> openSettings() async => true;

  @override
  Future<bool> openLimitedAccessPicker() async {
    limitedPickerOpened = true;
    return true;
  }

  @override
  Future<List<DiscoveredMediaItem>> discoverMedia({
    int? limit,
    DateTime? since,
  }) async =>
      [];

  @override
  Future<List<DiscoveredMediaItem>> queryMedia(MediaFilterQuery query) async =>
      [];
}

void main() {
  testWidgets('MediaPermissionView renders successfully in ProviderScope',
      (WidgetTester tester) async {
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

  testWidgets('limited access offers selected-media discovery and reselection',
      (WidgetTester tester) async {
    final service = _LimitedMediaService();
    var discoveryStarted = false;

    await tester.pumpWidget(
      ProviderScope(
        overrides: [mediaServiceProvider.overrideWithValue(service)],
        child: MaterialApp(
          theme: ThemeData.dark().copyWith(
            extensions: const [AppColors.dark],
          ),
          home: Scaffold(
            body: MediaPermissionView(
              onPermissionGranted: () => discoveryStarted = true,
            ),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('Limited Media Access'), findsOneWidget);
    expect(find.text('Manage Selection'), findsOneWidget);
    expect(find.text('Use Selected Media'), findsOneWidget);

    await tester.tap(find.text('Use Selected Media'));
    expect(discoveryStarted, isTrue);

    await tester.tap(find.text('Manage Selection'));
    await tester.pumpAndSettle();
    expect(service.limitedPickerOpened, isTrue);
  });

  test('limited access is usable while full access is distinct', () {
    expect(MediaPermissionState.limited.hasMediaAccess, isTrue);
    expect(MediaPermissionState.limited.hasFullMediaAccess, isFalse);
    expect(MediaPermissionState.granted.hasFullMediaAccess, isTrue);
    expect(MediaPermissionState.denied.hasMediaAccess, isFalse);
  });
}
