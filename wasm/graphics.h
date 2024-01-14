#include <cstdint>
#include <vector>

#include "matrix.h"

// Returns index in [min, max] from fraction in (0, 1).
// The indices split (0, 1) into `max - min + 1` equal subintervals.
inline int FractionToIndex(float fraction, int min, int max) {
  int i = (1 - fraction) * min + fraction * max + 0.5;
  return i < min ? min : i > max ? max : i;
}

template <class T>
inline T Clip(T u, T min, T max) {
  return u < min ? min : u > max ? max : u;
}

void DrawFieldsOnBitmap(const Matrix<float>& u, const Matrix<float>& mask,
                        std::vector<uint32_t>& bitmap);
void CopyToCanvas(uint32_t* buffer, int width, int height);

template <class Scal>
void DrawCircleOnMask(Matrix<Scal>& mask, float x, float y, float r);
