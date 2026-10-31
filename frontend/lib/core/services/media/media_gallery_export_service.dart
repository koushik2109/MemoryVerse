import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:photo_manager/photo_manager.dart';

/// Saves downloaded media through the system media library rather than writing
/// directly into shared-storage paths that Android scoped storage blocks.
class MediaGalleryExportService {
  const MediaGalleryExportService._();

  static bool get isSupportedPlatform =>
      !kIsWeb && (Platform.isAndroid || Platform.isIOS);

  static String folderFor({required bool isVideo}) =>
      isVideo ? 'Movies' : 'Pictures';

  static Future<String?> saveDownloadedFile({
    required String filePath,
    required String fileName,
    required bool isVideo,
  }) async {
    if (!isSupportedPlatform) return null;

    final relativePath =
        isVideo ? 'Movies/MemoryVerse' : 'Pictures/MemoryVerse';
    if (isVideo) {
      await PhotoManager.editor.saveVideo(
        File(filePath),
        title: fileName,
        relativePath: relativePath,
      );
    } else {
      await PhotoManager.editor.saveImageWithPath(
        filePath,
        title: fileName,
        relativePath: relativePath,
      );
    }
    return folderFor(isVideo: isVideo);
  }
}
