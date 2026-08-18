import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:memory_verse/core/navigation/router.dart';
import 'package:memory_verse/core/presentation/widgets/aurora_hero.dart';

class LandingScreen extends StatelessWidget {
  const LandingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return AuroraHero(
      onPrimaryCta: () {
        context.push(Routes.signUp);
      },
      onSecondaryCta: () {
        // Just show a snackbar for now
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Scroll down to see how it works!')),
        );
      },
      onSignIn: () {
        context.push(Routes.signIn);
      },
    );
  }
}
