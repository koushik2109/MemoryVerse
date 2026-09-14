import 'dart:io';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/core/services/media/media_service.dart';
import 'package:memory_verse/core/services/rag/media_indexing_service.dart';

class ScoredMediaMatch {
  final DiscoveredMediaItem item;
  final double relevanceScore;
  final String? matchReason;
  final List<String> matchedTags;
  final bool recommended;

  const ScoredMediaMatch({
    required this.item,
    required this.relevanceScore,
    this.matchReason,
    this.matchedTags = const [],
    this.recommended = true,
  });
}

class RagRetrievalResult {
  final String prompt;
  final String suggestedTitle;
  final String? suggestedDescription;
  final DateTime? suggestedDate;
  final String? suggestedLocation;
  final List<ScoredMediaMatch> matches;
  final int totalScanned;
  final int totalRecommended;

  const RagRetrievalResult({
    required this.prompt,
    required this.suggestedTitle,
    this.suggestedDescription,
    this.suggestedDate,
    this.suggestedLocation,
    required this.matches,
    required this.totalScanned,
    this.totalRecommended = 0,
  });

  List<DiscoveredMediaItem> get matchedItems => matches.map((m) => m.item).toList();
  List<DiscoveredMediaItem> get recommendedItems =>
      matches.where((m) => m.recommended).map((m) => m.item).toList();
}

class RagRetrievalService {
  final MediaIndexingService _indexingService;
  final AiRepository _aiRepository;

  RagRetrievalService(this._indexingService, this._aiRepository);

  Future<RagRetrievalResult> retrieveMemoriesForPrompt({
    required String prompt,
    int topK = 30,
  }) async {
    // 1. Sync / index accessible media items
    final allMedia = await _indexingService.syncMediaIndex(limit: 200);

    if (allMedia.isEmpty) {
      return RagRetrievalResult(
        prompt: prompt,
        suggestedTitle: prompt.trim().isEmpty ? 'New Memory' : _formatTitle(prompt),
        suggestedDescription: null,
        suggestedDate: DateTime.now(),
        suggestedLocation: null,
        matches: [],
        totalScanned: 0,
        totalRecommended: 0,
      );
    }

    // 2. Collect local physical files if present
    final localFiles = <File>[];
    for (final m in allMedia) {
      if (m.file != null && m.file!.existsSync()) {
        localFiles.add(m.file!);
      } else if (m.mediaReference.isNotEmpty) {
        final f = File(m.mediaReference);
        if (f.existsSync()) localFiles.add(f);
      }
    }

    try {
      Map<String, dynamic> response;
      if (localFiles.isNotEmpty) {
        // Visual CLIP curation on real image bytes
        response = await _aiRepository.curateLocalMedia(
          prompt: prompt,
          files: localFiles.take(40).toList(),
        );
      } else {
        // Fallback to metadata RAG filter
        final candidatePayloads = allMedia.map((m) => m.toCandidateJson()).toList();
        response = await _aiRepository.filterMedia(
          prompt: prompt,
          candidates: candidatePayloads,
          topK: topK,
        );
      }

      final matchedList = (response['matched_media'] as List<dynamic>?) ?? [];
      final mediaMap = <String, DiscoveredMediaItem>{};
      for (final m in allMedia) {
        mediaMap[m.id] = m;
        mediaMap[m.filename] = m;
        if (m.file != null) {
          mediaMap[m.file!.path.split(Platform.pathSeparator).last] = m;
        }
      }

      final matches = <ScoredMediaMatch>[];
      for (final raw in matchedList) {
        final id = raw['id'] as String?;
        final fname = raw['filename'] as String?;
        final targetItem = (id != null ? mediaMap[id] : null) ?? (fname != null ? mediaMap[fname] : null);

        if (targetItem != null) {
          final isRec = (raw['recommended'] as bool?) ??
              (((raw['relevance_score'] as num?)?.toDouble() ?? 0.0) >= 0.50);
          matches.add(ScoredMediaMatch(
            item: targetItem,
            relevanceScore: (raw['relevance_score'] as num?)?.toDouble() ?? 0.5,
            matchReason: raw['reason'] as String?,
            matchedTags: List<String>.from(raw['matched_tags'] ?? []),
            recommended: isRec,
          ));
        }
      }

      DateTime? parsedDate;
      if (response['suggested_date'] != null) {
        parsedDate = DateTime.tryParse(response['suggested_date']);
      }

      final recCount = (response['total_recommended'] as int?) ??
          matches.where((m) => m.recommended).length;

      return RagRetrievalResult(
        prompt: prompt,
        suggestedTitle: response['suggested_title'] ?? _formatTitle(prompt),
        suggestedDescription: response['suggested_description'],
        suggestedDate: parsedDate ?? (matches.isNotEmpty ? matches.first.item.timestamp : DateTime.now()),
        suggestedLocation: response['suggested_location'],
        matches: matches,
        totalScanned: allMedia.length,
        totalRecommended: recCount,
      );
    } catch (_) {
      // 4. Client-side local heuristic fallback (e.g. offline or during connection errors)
      return _localHeuristicRetrieval(prompt, allMedia, topK);
    }
  }

  RagRetrievalResult _localHeuristicRetrieval(
    String prompt,
    List<DiscoveredMediaItem> candidates,
    int topK,
  ) {
    final cleanPrompt = prompt.toLowerCase();
    final words = cleanPrompt.split(RegExp(r'\s+')).where((w) => w.length > 2).toSet();

    // Check year in prompt
    final yearRegex = RegExp(r'\b(19\d\d|20\d\d)\b');
    final yearMatch = yearRegex.firstMatch(cleanPrompt);
    final targetYear = yearMatch != null ? int.tryParse(yearMatch.group(1)!) : null;

    final scored = <ScoredMediaMatch>[];
    for (final item in candidates) {
      double score = 0.1;
      final tags = <String>[];
      final itemText = '${item.filename} ${item.locationName ?? ''} ${item.tags.join(' ')}'.toLowerCase();

      for (final w in words) {
        if (itemText.contains(w)) {
          score += 0.3;
          tags.add(w);
        }
      }

      if (targetYear != null && item.timestamp.year == targetYear) {
        score += 0.4;
        tags.add('$targetYear');
      }

      if (score >= 0.15 || words.isEmpty) {
        scored.add(ScoredMediaMatch(
          item: item,
          relevanceScore: score.clamp(0.0, 1.0),
          matchReason: tags.isNotEmpty ? 'Matched: ${tags.join(', ')}' : 'Context relevance',
          matchedTags: tags,
        ));
      }
    }

    scored.sort((a, b) => b.relevanceScore.compareTo(a.relevanceScore));
    final topMatches = scored.take(topK).toList();

    return RagRetrievalResult(
      prompt: prompt,
      suggestedTitle: _formatTitle(prompt),
      suggestedDescription: 'Automatically curated memories for "$prompt".',
      suggestedDate: topMatches.isNotEmpty ? topMatches.first.item.timestamp : DateTime.now(),
      suggestedLocation: null,
      matches: topMatches,
      totalScanned: candidates.length,
    );
  }

  String _formatTitle(String prompt) {
    if (prompt.trim().isEmpty) return 'New Memory';
    final words = prompt.trim().split(' ');
    return words.map((w) => w.isNotEmpty ? '${w[0].toUpperCase()}${w.substring(1)}' : '').join(' ');
  }
}

final ragRetrievalServiceProvider = Provider<RagRetrievalService>((ref) {
  final indexingService = ref.watch(mediaIndexingServiceProvider);
  final aiRepo = ref.watch(aiRepositoryProvider);
  return RagRetrievalService(indexingService, aiRepo);
});
