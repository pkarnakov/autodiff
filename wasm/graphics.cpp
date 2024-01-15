#include "graphics.h"

#include <emscripten.h>

#include <cmath>

#include "macros.h"

// Draws fields on a bitmap.
// u: solution field with values in [0, 1]; drawn with colors [C0,C1,C2].
// mask: mask field with values in [0, 1]; drawn with color C3.
// bitmap: buffer for pixels ABGR, will be resized to `u.size()`.
void DrawFieldsOnBitmap(const Matrix<float>& u, const Matrix<float>& mask,
                        std::vector<uint32_t>& bitmap) {
  const float C0[] = {0.34, 0.17, 0.54};  // Blue.
  const float C1[] = {0.97, 0.97, 0.97};  // White.
  const float C2[] = {0.85, 0.5, 0.07};   // Orange.
  const float C3[] = {0, 0.95, 0.6};      // Green.
  const size_t nx = u.nrow();
  const size_t ny = u.ncol();
  fassert_equal(mask.nrow(), nx);
  fassert_equal(mask.ncol(), ny);
  bitmap.resize(u.size());
  auto linear = [](float x, float x0, float x1, float u0, float u1) {
    return ((x1 - x) * u0 + (x - x0) * u1) / (x1 - x0);
  };
  auto color = [&](float v, int k) {
    return (v < 0.5 ? linear(v, 0, 0.5, C0[k], C1[k])
                    : linear(v, 0.5, 1, C1[k], C2[k]));
  };
  auto blend = [](float c, float ca, float a) -> float {
    return c * (1 - a) + ca * a;
  };
  auto round = [](float c) -> uint8_t {
    return Clip<float>(c * 255 + 0.5, 0, 255);
  };
  for (size_t ix = 0; ix < nx; ++ix) {
    for (size_t iy = 0; iy < ny; ++iy) {
      const float vu = Clip<float>(u(ix, iy), 0, 1);
      const float vmask = Clip<float>(mask(ix, iy), 0, 1);
      const uint8_t r = round(blend(color(vu, 0), C3[0], vmask * 0.5));
      const uint8_t g = round(blend(color(vu, 1), C3[1], vmask * 0.5));
      const uint8_t b = round(blend(color(vu, 2), C3[2], vmask * 0.5));
      bitmap[(ny - iy - 1) * nx + ix] = (0xFF << 24) | (b << 16) | (g << 8) | r;
    }
  }
}

// Copies ABGR pixels from buffer of size `width * height`
// to canvas `g_shared_canvas` defined in JS.
void CopyToCanvas(uint32_t* buffer, int width, int height) {
  EM_ASM_(
      {
        let data = Module.HEAPU8.slice($0, $0 + $1 * $2 * 4);
        let tctx = g_shared_canvas.getContext("2d");
        let image = tctx.getImageData(0, 0, $1, $2);
        image.data.set(data);
        tctx.putImageData(image, 0, 0);
      },
      buffer, width, height);
}

// Draws a smooth circle on a field.
// mask: target field.
// x, y: circle center in fractions (0, 1).
// r: radius of circle in cells.
template <class Scal>
void DrawCircleOnMask(Matrix<Scal>& mask, float x, float y, float r) {
  const size_t nx = mask.nrow();
  const size_t ny = mask.ncol();
  const int ixc = FractionToIndex(x, 0, nx - 1);
  const int iyc = FractionToIndex(y, 0, ny - 1);
  const int w = std::ceil(r);
  auto kernel = [r](int dx, int dy) -> Scal {  //
    return Clip<Scal>(3 * (1 - std::sqrt(sqr(dx) + sqr(dy)) / r), 0, 1);
  };
  const int ix0 = Clip<int>(ixc - w, 0, nx);
  const int ix1 = Clip<int>(ixc + w + 1, 0, nx);
  const int jx0 = Clip<int>(iyc - w, 0, ny);
  const int jx1 = Clip<int>(iyc + w + 1, 0, ny);
  for (int ix = ix0; ix < ix1; ++ix) {
    for (int iy = jx0; iy < jx1; ++iy) {
      const Scal a = kernel(ix - ixc, iy - iyc);
      mask(ix, iy) = Clip<Scal>(a + mask(ix, iy) * (1 - a), 0, 1);
    }
  }
}

template void DrawCircleOnMask(Matrix<float>& mask, float x, float y, float r);
template void DrawCircleOnMask(Matrix<double>& mask, float x, float y, float r);
