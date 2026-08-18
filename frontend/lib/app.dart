import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/core/design/theme.dart';
import 'package:memory_verse/core/navigation/router.dart';

class MemoryVerseApp extends ConsumerWidget {
  const MemoryVerseApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final router = ref.watch(routerProvider);

    return MaterialApp.router(
      title: 'MemoryVerse',
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.light,
      theme: AppTheme.light,
      routerConfig: router,
    );
  }
}
