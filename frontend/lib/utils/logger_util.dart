import 'package:logger/logger.dart';

class Log {
  final Logger _logger;

  Log() : _logger = Logger();

  // Debug log
  void debug(String message) {
    _logger.d(message);
  }

  // Error log
  void error(String message, [dynamic error, StackTrace? stackTrace]) {
    _logger.e(message, error: error, stackTrace: stackTrace);
  }

  // Info log
  void info(String message) {
    _logger.i(message);
  }

  // Warning log
  void warning(String message) {
    _logger.w(message);
  }

  // Verbose log
  void verbose(String message) {
    _logger.v(message);
  }

  // What could be potentially fatal log
  void wtf(String message) {
    _logger.wtf(message);
  }
}
