#ifndef MPP_LOGGER_H
#define MPP_LOGGER_H

#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace mpp {

enum class LogLevel { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, NONE = 4 };

class Logger {
public:
  static Logger &getInstance();

  void setLevel(LogLevel level);
  LogLevel getLevel() const { return level_; }

  void debug(const char *format, ...);
  void info(const char *format, ...);
  void warning(const char *format, ...);
  void error(const char *format, ...);

  // Enable/disable logging
  void enable() { enabled_ = true; }
  void disable() { enabled_ = false; }
  bool isEnabled() const { return enabled_; }

private:
  Logger();
  ~Logger() = default;
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  void log(LogLevel level, const char *level_str, const char *format, va_list args);

  LogLevel level_;
  bool enabled_;
};

// Convenience macros for logging
#define LOG_DEBUG(...) mpp::Logger::getInstance().debug(__VA_ARGS__)
#define LOG_INFO(...) mpp::Logger::getInstance().info(__VA_ARGS__)
#define LOG_WARNING(...) mpp::Logger::getInstance().warning(__VA_ARGS__)
#define LOG_ERROR(...) mpp::Logger::getInstance().error(__VA_ARGS__)

// Check if debug logging is enabled before expensive operations
#define IS_DEBUG_ENABLED() (mpp::Logger::getInstance().getLevel() <= mpp::LogLevel::DEBUG)

} // namespace mpp

#endif // MPP_LOGGER_H
