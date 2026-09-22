import 'dart:async';
import 'package:dio/dio.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;

final apiClientProvider = Provider<ApiClient>((ref) => ApiClient());

class ApiClient {
  late final Dio _dio;
  Completer<bool>? _refreshCompleter;

  /// Async mutex to prevent concurrent refresh token calls and 401 loops
  Future<bool> _refreshTokenMutex() async {
    if (_refreshCompleter != null) {
      return _refreshCompleter!.future;
    }
    _refreshCompleter = Completer<bool>();
    try {
      final res = await Supabase.instance.client.auth.refreshSession();
      final success = res.session != null;
      _refreshCompleter!.complete(success);
      return success;
    } catch (_) {
      _refreshCompleter!.complete(false);
      return false;
    } finally {
      _refreshCompleter = null;
    }
  }

  static String get baseUrl {
    String url = dotenv.env['API_BASE_URL'] ?? 'http://10.0.2.2:8000/api/v1';
    if (kIsWeb || (!kIsWeb && !Platform.isAndroid)) {
      url = url.replaceAll('10.0.2.2', '127.0.0.1');
    } else if (!kIsWeb && Platform.isAndroid) {
      url = url.replaceAll('localhost', '10.0.2.2').replaceAll('127.0.0.1', '10.0.2.2');
    }
    if (!url.endsWith('/api/v1')) {
      url = url.endsWith('/') ? '${url}api/v1' : '$url/api/v1';
    }
    return url;
  }

  ApiClient() {
    String url = baseUrl;

    _dio = Dio(
      BaseOptions(
        baseUrl: url,
        connectTimeout: const Duration(seconds: 15),
        receiveTimeout: const Duration(seconds: 30),
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
      ),
    );

    _dio.interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) async {
          final session = Supabase.instance.client.auth.currentSession;
          if (session != null) {
            options.headers['Authorization'] = 'Bearer ${session.accessToken}';
          }
          return handler.next(options);
        },
        onError: (DioException error, handler) async {
          if (error.response?.statusCode == 401) {
            // Use mutex to refresh token only once across concurrent failing requests
            final refreshed = await _refreshTokenMutex();
            if (refreshed) {
              final newSession = Supabase.instance.client.auth.currentSession;
              if (newSession != null) {
                error.requestOptions.headers['Authorization'] =
                    'Bearer ${newSession.accessToken}';
                try {
                  final clonedReq = await _dio.fetch(error.requestOptions);
                  return handler.resolve(clonedReq);
                } catch (retryError) {
                  if (retryError is DioException) {
                    return handler.next(retryError);
                  }
                }
              }
            } else {
              // If refresh failed, sign out so router redirects to sign-in
              await Supabase.instance.client.auth.signOut();
            }
          }
          return handler.next(error);
        },
      ),
    );
  }

  Dio get dio => _dio;

  Future<Response<T>> get<T>(
    String path, {
    Map<String, dynamic>? queryParameters,
  }) async {
    try {
      return await _dio.get<T>(path, queryParameters: queryParameters);
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  Future<Response<T>> post<T>(
    String path, {
    dynamic data,
    Map<String, dynamic>? queryParameters,
  }) async {
    try {
      return await _dio.post<T>(
        path,
        data: data,
        queryParameters: queryParameters,
      );
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  Future<Response<T>> put<T>(String path, {dynamic data}) async {
    try {
      return await _dio.put<T>(path, data: data);
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  Future<Response<T>> patch<T>(String path, {dynamic data}) async {
    try {
      return await _dio.patch<T>(path, data: data);
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  Future<Response<T>> delete<T>(String path) async {
    try {
      return await _dio.delete<T>(path);
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  /// Send a multipart/form-data request (for file uploads).
  Future<Response<T>> postForm<T>(
    String path, {
    required Map<String, dynamic> formData,
    CancelToken? cancelToken,
    ProgressCallback? onSendProgress,
  }) async {
    try {
      final form = FormData.fromMap(formData);
      return await _dio.post<T>(
        path,
        data: form,
        options: Options(
          contentType: 'multipart/form-data',
          sendTimeout: const Duration(seconds: 120),
          receiveTimeout: const Duration(seconds: 120),
        ),
        cancelToken: cancelToken,
        onSendProgress: onSendProgress,
      );
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  String _handleError(DioException e) {
    if (e.type == DioExceptionType.connectionTimeout ||
        e.type == DioExceptionType.receiveTimeout) {
      return 'Connection timed out. Please check your internet connection.';
    }
    if (e.type == DioExceptionType.connectionError) {
      return 'Cannot reach the server. Make sure the backend is running.';
    }
    if (e.response != null) {
      final statusCode = e.response?.statusCode;
      final body = e.response?.data;
      // Try to extract a meaningful error message
      String? msg;
      if (body is Map) {
        msg =
            body['detail'] as String? ??
            (body['error'] as Map?)?['message'] as String?;
      }
      if (msg != null && msg.isNotEmpty) return msg;
      switch (statusCode) {
        case 401:
          return 'Your session has expired. Please sign in again.';
        case 403:
          return 'You don\'t have permission to perform this action.';
        case 404:
          return 'The requested resource was not found.';
        case 409:
          return 'A conflict occurred. The resource may already exist.';
        case 422:
          return 'Invalid input. Please check your entries.';
        case 500:
          return 'Server error. Please try again later.';
        default:
          return 'Server error ($statusCode). Please try again.';
      }
    }
    return 'Network error. Please check your connection and try again.';
  }
}
