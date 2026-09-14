import 'dart:io';
import 'package:file_picker/file_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:memory_verse/core/services/media/media_service.dart';

class WindowsMediaService implements MediaService {
  static const String _prefCustomPathKey = 'windows_custom_media_folder';
  String? _customFolderPath;

  WindowsMediaService();

  @override
  String get platformName => 'Windows (Dev & Testing)';

  @override
  bool get isDesktopEnvironment => true;

  String? get activeCustomFolder => _customFolderPath;

  Future<void> init() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      _customFolderPath = prefs.getString(_prefCustomPathKey);
    } catch (_) {}
  }

  Future<String?> selectCustomFolder() async {
    final selectedDirectory = await FilePicker.platform.getDirectoryPath(
      dialogTitle: 'Select Media Folder for MemoryVerse Testing',
    );
    if (selectedDirectory != null) {
      _customFolderPath = selectedDirectory;
      try {
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString(_prefCustomPathKey, selectedDirectory);
      } catch (_) {}
    }
    return selectedDirectory;
  }

  @override
  Future<MediaPermissionState> checkPermission() async {
    // Windows file system access for user libraries / selected directories is always available.
    return MediaPermissionState.granted;
  }

  @override
  Future<MediaPermissionState> requestPermission() async {
    return MediaPermissionState.granted;
  }

  @override
  Future<bool> openSettings() async {
    return true;
  }

  List<Directory> _getWindowsMediaDirectories() {
    final dirs = <Directory>[];

    // 1. If user configured a custom testing directory, prioritize it
    if (_customFolderPath != null && _customFolderPath!.isNotEmpty) {
      final customDir = Directory(_customFolderPath!);
      if (customDir.existsSync()) {
        dirs.add(customDir);
        return dirs;
      }
    }

    // 2. Discover default Windows User directories
    final userProfile = Platform.environment['USERPROFILE'];
    if (userProfile != null) {
      final pictures = Directory('$userProfile\\Pictures');
      final videos = Directory('$userProfile\\Videos');
      final oneDrivePictures = Directory('$userProfile\\OneDrive\\Pictures');
      final oneDriveVideos = Directory('$userProfile\\OneDrive\\Videos');

      if (pictures.existsSync()) dirs.add(pictures);
      if (videos.existsSync()) dirs.add(videos);
      if (oneDrivePictures.existsSync()) dirs.add(oneDrivePictures);
      if (oneDriveVideos.existsSync()) dirs.add(oneDriveVideos);
    }

    // 3. Fallback to workspace assets or current directory
    final currentAssets = Directory('${Directory.current.path}\\assets');
    if (currentAssets.existsSync()) dirs.add(currentAssets);

    return dirs;
  }

  @override
  Future<List<DiscoveredMediaItem>> discoverMedia({
    int limit = 200,
    DateTime? since,
  }) async {
    if (_customFolderPath == null) {
      await init();
    }

    final items = <DiscoveredMediaItem>[];
    final allowedExtensions = {
      '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.mp4', '.mov', '.avi', '.mkv'
    };

    final dirs = _getWindowsMediaDirectories();

    for (final dir in dirs) {
      if (!dir.existsSync()) continue;

      try {
        // Recursively inspect folder up to reasonable depth (2 levels)
        final entities = dir.listSync(recursive: true, followLinks: false);
        for (final entity in entities) {
          if (items.length >= limit) break;
          if (entity is! File) continue;

          final ext = entity.path.toLowerCase().split('.').last;
          if (!allowedExtensions.contains('.$ext')) continue;

          final stat = entity.statSync();
          final isVideo = {'.mp4', '.mov', '.avi', '.mkv'}.contains('.$ext');
          final modified = stat.modified;

          if (since != null && modified.isBefore(since)) continue;

          final pathParts = entity.path.split(Platform.pathSeparator);
          final fname = pathParts.last;
          final parentFolder = pathParts.length > 1 ? pathParts[pathParts.length - 2] : '';

          // Extract lightweight tags from folder name and file name
          final folderTag = parentFolder.replaceAll(RegExp(r'[^a-zA-Z0-9]'), ' ').trim();
          final tags = <String>[];
          if (folderTag.isNotEmpty && folderTag.toLowerCase() != 'pictures' && folderTag.toLowerCase() != 'videos') {
            tags.add(folderTag);
          }

          items.add(DiscoveredMediaItem(
            id: 'win_${entity.path.hashCode}',
            type: isVideo ? MediaItemType.video : MediaItemType.image,
            mediaReference: entity.path,
            timestamp: modified,
            file: entity,
            thumbnailPath: isVideo ? null : entity.path,
            metadata: {
              'filename': fname,
              'file_size': stat.size,
              'extension': ext,
              'folder': parentFolder,
              'tags': tags,
              'source': 'windows_filesystem',
            },
          ));
        }
      } catch (_) {
        // Skip unreadable files/folders
      }
    }

    // Sort newest first
    items.sort((a, b) => b.timestamp.compareTo(a.timestamp));
    return items.take(limit).toList();
  }

  @override
  Future<List<DiscoveredMediaItem>> queryMedia(MediaFilterQuery query) async {
    final all = await discoverMedia(limit: 300);
    return all;
  }
}
