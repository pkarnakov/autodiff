#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

#define FILELINE (std::string() + __FILE__ + ":" + std::to_string(__LINE__))
#define fassert_equal(x, y)                                           \
  do {                                                                \
    const auto fasrteq_x = (x);                                       \
    const auto fasrteq_y = (y);                                       \
    if (!(fasrteq_x == fasrteq_y)) {                                  \
      std::stringstream fasrteq_s;                                    \
      fasrteq_s << FILELINE << ": assertion failed, expected equal "; \
      fasrteq_s << #x << "=" << fasrteq_x << " and " << #y << "="     \
                << fasrteq_y;                                         \
      throw std::runtime_error(fasrteq_s.str());                      \
    }                                                                 \
  } while (0);
