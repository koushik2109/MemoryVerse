import 'package:flutter/material.dart';

class Memory {
  final String id;
  final String title;
  final String description;
  final DateTime date;
  final IconData icon;

  Memory({
    required this.id,
    required this.title,
    required this.description,
    required this.date,
    required this.icon,
  });

  factory Memory.fromJson(Map<String, dynamic> json) {
    return Memory(
      id: json['id'] as String,
      title: json['title'] as String,
      description: json['description'] as String,
      date: DateTime.parse(json['date'] as String),
      icon: _getIconData(json['icon_name'] as String?),
    );
  }

  static IconData _getIconData(String? name) {
    switch (name) {
      case 'landscape': return Icons.landscape_rounded;
      case 'memory': return Icons.memory_rounded;
      case 'school': return Icons.school_rounded;
      case 'coffee': return Icons.coffee_rounded;
      case 'music': return Icons.music_note_rounded;
      case 'code': return Icons.code_rounded;
      default: return Icons.notes_rounded;
    }
  }
}
