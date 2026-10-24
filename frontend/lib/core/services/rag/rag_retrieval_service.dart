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
  final String? locationCandidate;
  final double locationConfidence;
  final String? locationDecisionStatus;

  const ScoredMediaMatch({
    required this.item,
    required this.relevanceScore,
    this.matchReason,
    this.matchedTags = const [],
    this.recommended = true,
    this.locationCandidate,
    this.locationConfidence = 0.0,
    this.locationDecisionStatus,
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

  List<DiscoveredMediaItem> get matchedItems =>
      matches.map((m) => m.item).toList();
  List<DiscoveredMediaItem> get recommendedItems =>
      matches.where((m) => m.recommended).map((m) => m.item).toList();
}

class RagRetrievalService {
  // Keep each request bounded while still visiting every indexed item.
  static const int _retrievalBatchSize = 40;

  final MediaIndexingService _indexingService;
  final AiRepository _aiRepository;

  RagRetrievalService(this._indexingService, this._aiRepository);

  Future<RagRetrievalResult> retrieveMemoriesForPrompt({
    required String prompt,
    int topK = 30,
  }) async {
    // 1. Sync / index accessible media items
    final allMedia = await _indexingService.syncMediaIndex();

    if (allMedia.isEmpty) {
      return RagRetrievalResult(
        prompt: prompt,
        suggestedTitle:
            prompt.trim().isEmpty ? 'New Memory' : _formatTitle(prompt),
        suggestedDescription: null,
        suggestedDate: DateTime.now(),
        suggestedLocation: null,
        matches: [],
        totalScanned: 0,
        totalRecommended: 0,
      );
    }

    try {
      // 2. Run bounded requests until every indexed item has been considered.
      final response = await _retrieveInBatches(
        prompt: prompt,
        allMedia: allMedia,
        topK: topK,
      );

      final matchedList = (response['matched_media'] as List<dynamic>?) ?? [];
      final mediaById = <String, DiscoveredMediaItem>{};
      final mediaByFilename = <String, List<DiscoveredMediaItem>>{};
      for (final m in allMedia) {
        mediaById[m.id] = m;
        mediaByFilename.putIfAbsent(m.filename, () => []).add(m);
        if (m.file != null) {
          final fileName = m.file!.path.split(Platform.pathSeparator).last;
          mediaByFilename.putIfAbsent(fileName, () => []).add(m);
        }
      }

      final matches = <ScoredMediaMatch>[];
      final matchedItemIds = <String>{};
      for (final raw in matchedList) {
        if (raw is! Map) continue;
        final id = raw['id'] as String?;
        final fname = raw['filename'] as String?;
        final targetItem = (id != null ? mediaById[id] : null) ??
            _firstUnmatchedByFilename(
              mediaByFilename[fname],
              matchedItemIds,
            );

        if (targetItem != null && matchedItemIds.add(targetItem.id)) {
          final isRec = (raw['recommended'] as bool?) ??
              (((raw['relevance_score'] as num?)?.toDouble() ?? 0.0) >= 0.50);
          matches.add(ScoredMediaMatch(
            item: targetItem,
            relevanceScore: (raw['relevance_score'] as num?)?.toDouble() ?? 0.5,
            matchReason: raw['reason'] as String?,
            matchedTags: List<String>.from(raw['matched_tags'] ?? []),
            recommended: isRec,
            locationCandidate: raw['location_candidate'] as String?,
            locationConfidence:
                (raw['location_confidence'] as num?)?.toDouble() ?? 0.0,
            locationDecisionStatus: raw['location_decision_status'] as String?,
          ));
        }
      }

      matches.sort((a, b) => b.relevanceScore.compareTo(a.relevanceScore));
      final selectedMatches = matches.take(topK).toList();

      DateTime? parsedDate;
      if (response['suggested_date'] != null) {
        parsedDate = DateTime.tryParse(response['suggested_date'].toString());
      }

      final recCount = (response['total_recommended'] as num?)?.toInt() ??
          selectedMatches.where((m) => m.recommended).length;

      return RagRetrievalResult(
        prompt: prompt,
        suggestedTitle: response['suggested_title'] ?? _formatTitle(prompt),
        suggestedDescription: response['suggested_description'],
        suggestedDate: parsedDate ??
            (selectedMatches.isNotEmpty
                ? selectedMatches.first.item.timestamp
                : DateTime.now()),
        suggestedLocation: response['suggested_location'],
        matches: selectedMatches,
        totalScanned: allMedia.length,
        totalRecommended: recCount,
      );
    } catch (_) {
      // 4. Client-side local heuristic fallback (e.g. offline or during connection errors)
      return _localHeuristicRetrieval(prompt, allMedia, topK);
    }
  }

  Future<Map<String, dynamic>> _retrieveInBatches({
    required String prompt,
    required List<DiscoveredMediaItem> allMedia,
    required int topK,
  }) async {
    final responses = <Map<String, dynamic>>[];
    final metadataOnly = <Map<String, dynamic>>[];
    final localEntries = <Map<String, dynamic>>[];

    for (final item in allMedia) {
      final file = _existingFileFor(item);
      if (file != null) {
        localEntries.add({
          'file': file,
          'metadata': item.toCandidateJson(),
        });
      } else {
        metadataOnly.add(item.toCandidateJson());
      }
    }

    // Visual curation is preferred for files available to the client. Every
    // file is sent in a bounded request instead of truncating at the first 40.
    for (final batch in _chunk(localEntries, _retrievalBatchSize)) {
      responses.add(await _aiRepository.curateLocalMedia(
        prompt: prompt,
        files: batch.map((entry) => entry['file'] as File).toList(),
        candidateMetadata: batch
            .map((entry) => Map<String, dynamic>.from(entry['metadata'] as Map))
            .toList(),
      ));
    }

    // Items without readable local bytes still participate through metadata.
    // This also serves as the primary path on platforms that expose metadata
    // without a filesystem path.
    final metadataCandidates = localEntries.isEmpty
        ? allMedia.map((item) => item.toCandidateJson()).toList()
        : metadataOnly;
    for (final batch in _chunk(metadataCandidates, _retrievalBatchSize)) {
      responses.add(await _aiRepository.filterMedia(
        prompt: prompt,
        candidates: batch,
        topK: topK,
      ));
    }

    return _combineResponses(responses);
  }

  File? _existingFileFor(DiscoveredMediaItem item) {
    if (item.file != null && item.file!.existsSync()) return item.file;
    if (item.mediaReference.isEmpty) return null;

    final file = File(item.mediaReference);
    return file.existsSync() ? file : null;
  }

  Iterable<List<T>> _chunk<T>(List<T> values, int size) sync* {
    for (var start = 0; start < values.length; start += size) {
      final end = (start + size < values.length) ? start + size : values.length;
      yield values.sublist(start, end);
    }
  }

  Map<String, dynamic> _combineResponses(List<Map<String, dynamic>> responses) {
    final matchedMedia = <dynamic>[];
    var totalRecommended = 0;
    var hasRecommendedCount = false;
    final combined = <String, dynamic>{};

    for (final response in responses) {
      final rawMatches = response['matched_media'];
      if (rawMatches is List) {
        for (final raw in rawMatches) {
          if (raw is! Map) continue;
          final normalized = Map<String, dynamic>.from(raw);
          // Visual curation currently returns filenames rather than a native
          // asset ID. The same basename can exist in different folders, so
          // retain duplicates here and resolve them against indexed items in
          // the caller, where the full path is available.
          matchedMedia.add(normalized);
        }
      }

      final recommended = response['total_recommended'];
      if (recommended is num) {
        totalRecommended += recommended.toInt();
        hasRecommendedCount = true;
      }

      for (final key in const [
        'suggested_title',
        'suggested_description',
        'suggested_date',
        'suggested_location',
      ]) {
        if (!combined.containsKey(key) && response[key] != null) {
          combined[key] = response[key];
        }
      }
    }

    combined['matched_media'] = matchedMedia;
    if (hasRecommendedCount) combined['total_recommended'] = totalRecommended;
    return combined;
  }

  DiscoveredMediaItem? _firstUnmatchedByFilename(
    List<DiscoveredMediaItem>? candidates,
    Set<String> matchedItemIds,
  ) {
    if (candidates == null) return null;
    for (final candidate in candidates) {
      if (!matchedItemIds.contains(candidate.id)) return candidate;
    }
    return null;
  }

  RagRetrievalResult _localHeuristicRetrieval(
    String prompt,
    List<DiscoveredMediaItem> candidates,
    int topK,
  ) {
    final cleanPrompt = prompt.toLowerCase();
    final words =
        cleanPrompt.split(RegExp(r'\s+')).where((w) => w.length > 2).toSet();

    // Check year in prompt
    final yearRegex = RegExp(r'\b(19\d\d|20\d\d)\b');
    final yearMatch = yearRegex.firstMatch(cleanPrompt);
    final targetYear =
        yearMatch != null ? int.tryParse(yearMatch.group(1)!) : null;

    final scored = <ScoredMediaMatch>[];
    for (final item in candidates) {
      double score = 0.1;
      final tags = <String>[];
      final itemText = [
        item.filename,
        item.locationName ?? '',
        item.metadata['location'] ?? '',
        item.metadata['folder'] ?? '',
        item.metadata['relative_path'] ?? '',
        item.tags.join(' '),
      ].join(' ').toLowerCase();

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
          matchReason: tags.isNotEmpty
              ? 'Matched: ${tags.join(', ')}'
              : 'Context relevance',
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
      suggestedDate: topMatches.isNotEmpty
          ? topMatches.first.item.timestamp
          : DateTime.now(),
      suggestedLocation: null,
      matches: topMatches,
      totalScanned: candidates.length,
    );
  }

  String _formatTitle(String prompt) {
    if (prompt.trim().isEmpty) return 'New Memory';
    final words = prompt.trim().split(' ');
    return words
        .map(
            (w) => w.isNotEmpty ? '${w[0].toUpperCase()}${w.substring(1)}' : '')
        .join(' ');
  }
}

final ragRetrievalServiceProvider = Provider<RagRetrievalService>((ref) {
  final indexingService = ref.watch(mediaIndexingServiceProvider);
  final aiRepo = ref.watch(aiRepositoryProvider);
  return RagRetrievalService(indexingService, aiRepo);
});
