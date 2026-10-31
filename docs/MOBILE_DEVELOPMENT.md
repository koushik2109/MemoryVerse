# Flutter development on Linux, Android, and iOS

This guide covers API connectivity for local emulators and physical phones. Keep generated Flutter and Android files machine-local; they are recreated for the current operating system.

## Regenerate platform metadata after changing machines

From `frontend/`, run:

```sh
flutter clean
flutter pub get
```

`android/local.properties`, `.dart_tool/`, `build/`, `ios/Flutter/Generated.xcconfig`, and `ios/Flutter/flutter_export_environment.sh` are generated or machine-specific. Do not copy them from Windows to Linux or commit them. `flutter pub get` regenerates Flutter's configuration for the current checkout.

## Android emulator on Linux

The default API address is `http://10.0.2.2:8000/api/v1`, which the Android emulator maps to the development machine. Leave `API_BASE_URL_ANDROID` blank in the local `frontend/.env`. Start the backend and the emulator, then run `flutter run -d <device-id>` from `frontend/`.

The Android debug manifest permits cleartext requests for local development. The main manifest does not enable cleartext traffic for release builds; use HTTPS for deployed backends.

## Physical Android phone

Connect the phone and development machine to the same trusted network. Set `API_BASE_URL_ANDROID` in the ignored local `frontend/.env` to the machine's LAN address, for example `http://192.168.1.25:8000/api/v1`. Start the backend with `MEMORYVERSE_BIND_HOST=0.0.0.0` so the phone can reach it, and allow port 8000 through the host firewall for that private network if needed.

## iOS simulator and physical iPhone

An iOS simulator uses the development machine's loopback address. Leave `API_BASE_URL_IOS` blank, or set it to `http://127.0.0.1:8000/api/v1`.

A physical iPhone needs the development machine's LAN address in `API_BASE_URL_IOS` and the backend must be started with `MEMORYVERSE_BIND_HOST=0.0.0.0`. Keep both devices on the same network and approve the iOS local-network prompt. The iOS project allows local-network HTTP during development only; use HTTPS for deployed services. Building or running an iOS target requires macOS, Xcode, and CocoaPods.

## Other targets

Web and Linux desktop default to `127.0.0.1`. Set `API_BASE_URL_WEB` or the shared `API_BASE_URL` in the ignored `.env` when the backend is on another host. Platform-specific variables override the shared URL; blank overrides are ignored.
