import 'dart:io';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/core/services/media/media_service.dart';
import 'package:memory_verse/core/services/media/android_media_service.dart';
import 'package:memory_verse/core/services/media/ios_media_service.dart';
import 'package:memory_verse/core/services/media/windows_media_service.dart';

final mediaServiceProvider = Provider<MediaService>((ref) {
  if (Platform.isAndroid) {
    return AndroidMediaService();
  } else if (Platform.isIOS) {
    return IOSMediaService();
  } else {
    // Windows, macOS, Linux dev/testing environment
    return WindowsMediaService();
  }
});

/// StateNotifier/Notifier for current permission state and reactive updates
final mediaPermissionStateProvider = StateNotifierProvider<MediaPermissionNotifier, AsyncValue<MediaPermissionState>>((ref) {
  final service = ref.watch(mediaServiceProvider);
  return MediaPermissionNotifier(service);
});

class MediaPermissionNotifier extends StateNotifier<AsyncValue<MediaPermissionState>> {
  final MediaService _service;

  MediaPermissionNotifier(this._service) : super(const AsyncValue.loading()) {
    checkPermission();
  }

  Future<void> checkPermission() async {
    state = const AsyncValue.loading();
    try {
      final status = await _service.checkPermission();
      state = AsyncValue.data(status);
    } catch (e, st) {
      state = AsyncValue.error(e, st);
    }
  }

  Future<MediaPermissionState> requestPermission() async {
    state = const AsyncValue.loading();
    try {
      final status = await _service.requestPermission();
      state = AsyncValue.data(status);
      return status;
    } catch (e, st) {
      state = AsyncValue.error(e, st);
      return MediaPermissionState.denied;
    }
  }

  Future<void> openSettings() async {
    await _service.openSettings();
  }
}
