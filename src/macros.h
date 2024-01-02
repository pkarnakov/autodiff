#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

#define FILELINE (std::string() + __FILE__ + ":" + std::to_string(__LINE__))

#define NAMEVALUE(x)                 \
  ([&]() -> std::string {            \
    std::stringstream namevalue_s;   \
    namevalue_s << #x << '=' << (x); \
    return namevalue_s.str();        \
  }())

#define AUTODIFF_CAT(x, y) x##y
#define AUTODIFF_XCAT(x, y) AUTODIFF_CAT(x, y)
#define USEFLAG(x) AUTODIFF_XCAT(0, _USE_##x##_)
#define AUTODIFF_ID_1(x) x
#define AUTODIFF_ID_0(x)

#define fassert_1(x)                                                      \
  do {                                                                    \
    if (!(x)) {                                                           \
      throw std::runtime_error(FILELINE + ": assertion failed '" #x "'"); \
    }                                                                     \
  } while (0);

#define fassert_2(x, msg)                                                   \
  do {                                                                      \
    if (!(x)) {                                                             \
      throw std::runtime_error(FILELINE + ": assertion failed '" #x "'\n" + \
                               (msg));                                      \
    }                                                                       \
  } while (0);

#define GET_COUNT(_1, _2, _3, N, ...) N
#define fassert(...)                                                       \
  AUTODIFF_ID_1(AUTODIFF_ID_1(GET_COUNT(__VA_ARGS__, fassert_3, fassert_2, \
                                        fassert_1, fassert_0))(__VA_ARGS__))
#define fassert_equal(...)                                                   \
  AUTODIFF_ID_1(                                                             \
      AUTODIFF_ID_1(GET_COUNT(__VA_ARGS__, fassert_equal_3, fassert_equal_2, \
                              fassert_equal_1, fassert_equal_0))(__VA_ARGS__))

#define fassert_equal_2(x, y)                                         \
  do {                                                                \
    const auto fasrteq_x = (x);                                       \
    const auto fasrteq_y = (y);                                       \
    if (!(fasrteq_x == fasrteq_y)) {                                  \
      std::stringstream fasrteq_s;                                    \
      fasrteq_s << FILELINE << ": assertion failed, expected equal "; \
      fasrteq_s << #x << "='" << fasrteq_x << "' and " << #y << "='"  \
                << fasrteq_y << "'";                                  \
      throw std::runtime_error(fasrteq_s.str());                      \
    }                                                                 \
  } while (0);

#define fassert_equal_3(x, y, msg)                                    \
  do {                                                                \
    const auto fasrteq_x = (x);                                       \
    const auto fasrteq_y = (y);                                       \
    if (!(fasrteq_x == fasrteq_y)) {                                  \
      std::stringstream fasrteq_s;                                    \
      fasrteq_s << FILELINE << ": assertion failed, expected equal "; \
      fasrteq_s << #x << "='" << fasrteq_x << "' and " << #y << "='"  \
                << fasrteq_y << "'" << msg;                           \
      throw std::runtime_error(fasrteq_s.str());                      \
    }                                                                 \
  } while (0);
