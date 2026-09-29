import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/core/providers/theme_provider.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:memory_verse/app.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
    ),
  );

  String supabaseUrl = 'https://cmllsgjtgpdtzwqdnwdl.supabase.co';
  String supabaseAnonKey = 'sb_publishable_NlSazEm5x0vevHmsc54gmw_Ipd2HKvb';

  try {
    await dotenv.load(fileName: '.env');
    final envUrl = dotenv.env['SUPABASE_URL']?.trim();
    final envKey = dotenv.env['SUPABASE_ANON_KEY']?.trim();
    if (envUrl != null && envUrl.isNotEmpty) supabaseUrl = envUrl;
    if (envKey != null && envKey.isNotEmpty) supabaseAnonKey = envKey;
  } catch (e) {
    debugPrint('Warning: Could not load .env file, using default configuration: $e');
  }

  // Clean any quotes
  supabaseUrl = supabaseUrl.replaceAll('"', '').replaceAll("'", '').trim();
  supabaseAnonKey = supabaseAnonKey.replaceAll('"', '').replaceAll("'", '').trim();

  try {
    await Supabase.initialize(
      url: supabaseUrl,
      anonKey: supabaseAnonKey,
      authOptions: const FlutterAuthClientOptions(
        authFlowType: AuthFlowType.pkce,
      ),
    );
    debugPrint('Supabase initialized successfully with $supabaseUrl');
  } catch (e) {
    debugPrint('Warning: Supabase.initialize error: $e');
  }

  runApp(
    ProviderScope(
      overrides: [
        themeModeProvider.overrideWith(
          (ref) => ThemeModeNotifier(ThemeMode.dark),
        ),
      ],
      child: const MemoryVerseApp(),
    ),
  );
}
