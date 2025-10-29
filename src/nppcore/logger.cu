#include "logger.h"
#include <cstdlib>
#include <ctime>

namespace mpp {

Logger::Logger() : level_(LogLevel::INFO), enabled_(true) {
  // Check environment variable for log level
  const char* env_level = std::getenv("MPP_LOG_LEVEL");
  if (env_level) {
    if (strcmp(env_level, "DEBUG") == 0) {
      level_ = LogLevel::DEBUG;
    } else if (strcmp(env_level, "INFO") == 0) {
      level_ = LogLevel::INFO;
    } else if (strcmp(env_level, "WARNING") == 0) {
      level_ = LogLevel::WARNING;
    } else if (strcmp(env_level, "ERROR") == 0) {
      level_ = LogLevel::ERROR;
    } else if (strcmp(env_level, "NONE") == 0) {
      level_ = LogLevel::NONE;
    }
  }

  // Check if logging is disabled
  const char* env_disable = std::getenv("MPP_LOG_DISABLE");
  if (env_disable && strcmp(env_disable, "1") == 0) {
    enabled_ = false;
  }
}

Logger& Logger::getInstance() {
  static Logger instance;
  return instance;
}

void Logger::setLevel(LogLevel level) {
  level_ = level;
}

void Logger::log(LogLevel level, const char* level_str, const char* format, va_list args) {
  if (!enabled_ || level < level_) {
    return;
  }

  // Print timestamp
  time_t now = time(nullptr);
  struct tm* timeinfo = localtime(&now);
  char time_buffer[32];
  strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", timeinfo);

  // Print level and timestamp
  fprintf(stderr, "[%s] [%s] ", level_str, time_buffer);

  // Print message
  vfprintf(stderr, format, args);
  fprintf(stderr, "\n");
  fflush(stderr);
}

void Logger::debug(const char* format, ...) {
  va_list args;
  va_start(args, format);
  log(LogLevel::DEBUG, "DEBUG", format, args);
  va_end(args);
}

void Logger::info(const char* format, ...) {
  va_list args;
  va_start(args, format);
  log(LogLevel::INFO, "INFO", format, args);
  va_end(args);
}

void Logger::warning(const char* format, ...) {
  va_list args;
  va_start(args, format);
  log(LogLevel::WARNING, "WARNING", format, args);
  va_end(args);
}

void Logger::error(const char* format, ...) {
  va_list args;
  va_start(args, format);
  log(LogLevel::ERROR, "ERROR", format, args);
  va_end(args);
}

} // namespace mpp
