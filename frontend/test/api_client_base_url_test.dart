import 'package:flutter_test/flutter_test.dart';
import 'package:memory_verse/core/api/api_client.dart';

void main() {
  group('ApiClient.resolveBaseUrl', () {
    test('uses the Android emulator host by default', () {
      expect(
        ApiClient.resolveBaseUrl(
          environment: const {
            'API_BASE_URL_ANDROID': '',
            'API_BASE_URL': '',
          },
          isWeb: false,
          isAndroid: true,
          isIOS: false,
        ),
        'http://10.0.2.2:8000/api/v1',
      );
    });

    test('preserves a physical Android device LAN override', () {
      expect(
        ApiClient.resolveBaseUrl(
          environment: const {
            'API_BASE_URL': 'http://10.0.2.2:8000/api/v1',
            'API_BASE_URL_ANDROID': 'http://192.168.1.25:8000',
          },
          isWeb: false,
          isAndroid: true,
          isIOS: false,
        ),
        'http://192.168.1.25:8000/api/v1',
      );
    });

    test('maps the Android emulator URL to the iOS simulator host', () {
      expect(
        ApiClient.resolveBaseUrl(
          environment: const {
            'API_BASE_URL': 'http://10.0.2.2:8000/api/v1',
          },
          isWeb: false,
          isAndroid: false,
          isIOS: true,
        ),
        'http://127.0.0.1:8000/api/v1',
      );
    });

    test('preserves a physical iOS device LAN override', () {
      expect(
        ApiClient.resolveBaseUrl(
          environment: const {
            'API_BASE_URL_IOS': 'http://192.168.1.25:8000/api/v1',
          },
          isWeb: false,
          isAndroid: false,
          isIOS: true,
        ),
        'http://192.168.1.25:8000/api/v1',
      );
    });

    test('uses loopback for desktop and web defaults', () {
      for (final isWeb in [false, true]) {
        expect(
          ApiClient.resolveBaseUrl(
            environment: const {},
            isWeb: isWeb,
            isAndroid: false,
            isIOS: false,
          ),
          'http://127.0.0.1:8000/api/v1',
        );
      }
    });
  });
}
