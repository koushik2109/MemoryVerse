import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/memory.dart';
import 'package:flutter/material.dart';

class MemoriesNotifier extends Notifier<List<Memory>> {
  @override
  List<Memory> build() {
    return [
      Memory(
        id: '1',
        title: 'Trip to the Mountains',
        description: 'A beautiful weekend at the peak.',
        date: DateTime(2023, 10, 24),
        icon: Icons.landscape_rounded,
      ),
      Memory(
        id: '2',
        title: 'First AI Project',
        description: 'Started working on MemoryVerse.',
        date: DateTime(2023, 10, 25),
        icon: Icons.memory_rounded,
      ),
      Memory(
        id: '3',
        title: 'Graduation Day',
        description: 'Finally finished the degree!',
        date: DateTime(2023, 5, 20),
        icon: Icons.school_rounded,
      ),
    ];
  }

  void addMemory(Memory memory) {
    state = [...state, memory];
  }
}

final memoriesProvider = NotifierProvider<MemoriesNotifier, List<Memory>>(
  () => MemoriesNotifier(),
);
