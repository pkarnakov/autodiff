#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iosfwd>
#include <numeric>
#include <vector>

#include "macros.h"

template <class T>
T sqr(T x) {
  return x * x;
}

template <class T>
T tanh(T x) {
  using std::exp;
  return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

template <class T>
class Matrix {
 public:
  template <class U>
  friend class Matrix;

  // Constructor.
  Matrix() : nrow_(0), ncol_(0) {}
  Matrix(const Matrix&) = default;
  explicit Matrix(size_t n) : Matrix(n, n) {}
  Matrix(size_t nrow, size_t ncol)
      : nrow_(nrow), ncol_(ncol), data_(nrow * ncol) {}
  Matrix(size_t nrow, size_t ncol, T value)
      : nrow_(nrow), ncol_(ncol), data_(nrow * ncol, value) {}
  template <class U>
  Matrix(const Matrix<U>& other)
      : nrow_(other.nrow_),
        ncol_(other.ncol_),
        data_(other.data_.begin(), other.data_.end()) {}

  // Element access.
  T& operator()(size_t i, size_t j) {
    return data_[i * ncol_ + j];
  }
  const T& operator()(size_t i, size_t j) const {
    return data_[i * ncol_ + j];
  }
  T& operator[](size_t i) {
    return data_[i];
  }
  const T& operator[](size_t i) const {
    return data_[i];
  }
  size_t nrow() const {
    return nrow_;
  }
  size_t ncol() const {
    return ncol_;
  }
  size_t size() const {
    return nrow_ * ncol_;
  }
  const T* data() const {
    return data_.data();
  }
  T* data() {
    return data_.data();
  }

  // Assignment.
  Matrix& operator=(const Matrix&) = default;
  Matrix& operator+=(const Matrix& other) {
    if (nrow_ == 0 && ncol_ == 0) {
      *this = other;
      return *this;
    }
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      data_[i] += other.data_[i];
    }
    return *this;
  }
  Matrix& operator-=(const Matrix& other) {
    if (nrow_ == 0 && ncol_ == 0) {
      *this = -other;
      return *this;
    }
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      data_[i] -= other.data_[i];
    }
    return *this;
  }
  Matrix& operator*=(const T& a) {
    for (size_t i = 0; i < data_.size(); ++i) {
      data_[i] *= a;
    }
    return *this;
  }
  Matrix& operator/=(const T& a) {
    for (size_t i = 0; i < data_.size(); ++i) {
      data_[i] /= a;
    }
    return *this;
  }
  void fill(const T& value) {
    std::fill(data_.begin(), data_.end(), value);
  }
  void clear() {
    fill(T(0));
  }

  // Member functions.
  Matrix operator-() const {
    Matrix res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = -data_[i];
    }
    return res;
  }
  Matrix operator+(const T& a) const {
    Matrix res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = data_[i] + a;
    }
    return res;
  }
  Matrix operator-(const T& a) const {
    Matrix res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = data_[i] - a;
    }
    return res;
  }
  Matrix operator*(const T& a) const {
    Matrix res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = data_[i] * a;
    }
    return res;
  }
  Matrix operator/(const T& a) const {
    Matrix res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = data_[i] / a;
    }
    return res;
  }
  Matrix operator+(const Matrix& other) const {
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    Matrix res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = data_[i] + other[i];
    }
    return res;
  }
  Matrix operator-(const Matrix& other) const {
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    Matrix res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = data_[i] - other[i];
    }
    return res;
  }
  Matrix operator*(const Matrix& other) const {
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    Matrix res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = data_[i] * other[i];
    }
    return res;
  }
  Matrix operator/(const Matrix& other) const {
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    Matrix res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = data_[i] / other[i];
    }
    return res;
  }
  Matrix transpose() const {
    Matrix res(ncol_, nrow_);
    for (size_t i = 0; i < nrow_; ++i) {
      for (size_t j = 0; j < ncol_; ++j) {
        res(j, i) = (*this)(i, j);
      }
    }
    return res;
  }
  Matrix roll(int shift_row, int shift_col) const {
    using Idx = size_t[2];
    Matrix res(nrow_, ncol_);
    if (nrow_ == 0 || ncol_ == 0) {
      return res;
    }
    const Idx shape = {nrow_, ncol_};
    auto& src = *this;
    auto& dst = res;
    // Rectangle to copy:
    Idx idst;  // Starting position in dst.
    Idx isrc;  // Starting position in src.
    Idx icnt;  // Elements count.

    // Flips the rectangle along axis k.
    auto flip = [&](int k) {
      idst[k] = (idst[k] == 0 ? icnt[k] : 0);
      isrc[k] = (isrc[k] == 0 ? icnt[k] : 0);
      icnt[k] = shape[k] - icnt[k];
    };
    // Copies the rectangle from src to dst.
    auto copy = [&]() {
      if (icnt[0] == 0 || icnt[1] == 0) {
        return;
      }
      for (auto k : {0, 1}) {
        fassert(idst[k] + icnt[k] <= shape[k]);
        fassert(isrc[k] + icnt[k] <= shape[k]);
      }
      for (size_t i = 0; i < icnt[0]; ++i) {
        for (size_t j = 0; j < icnt[1]; ++j) {
          dst(idst[0] + i, idst[1] + j) = src(isrc[0] + i, isrc[1] + j);
        }
      }
    };

    // Normalize shift to positive values and select one rectangle to copy.
    int shift[2] = {shift_row, shift_col};
    for (auto k : {0, 1}) {
      if (shift[k] < 0) {
        shift[k] = shape[k] - (-shift[k]) % shape[k];
      } else {
        shift[k] = shift[k] % shape[k];
      }
      idst[k] = shift[k];
      isrc[k] = 0;
      icnt[k] = size_t(shape[k] - shift[k]);
    }

    copy();

    flip(0);
    copy();

    flip(1);
    copy();

    flip(0);
    copy();

    return res;
  }
  // Restricts array to the next coarser level.
  Matrix restrict() const {
    fassert(nrow_ % 2 == 0);
    fassert(ncol_ % 2 == 0);
    Matrix res(nrow_ / 2, ncol_ / 2);
    for (size_t i = 0; i < res.nrow_; ++i) {
      for (size_t j = 0; j < res.ncol_; ++j) {
        T a{};
        for (int di = 0; di < 2; ++di) {
          for (int dj = 0; dj < 2; ++dj) {
            a += (*this)(2 * i + di, 2 * j + dj);
          }
        }
        res(i, j) = a * 0.25;
      }
    }
    return res;
  }
  // Adjoint of restrict().
  Matrix restrict_adjoint() const {
    Matrix res(nrow_ * 2, ncol_ * 2);
    for (size_t i = 0; i < nrow_; ++i) {
      for (size_t j = 0; j < ncol_; ++j) {
        for (int di = 0; di < 2; ++di) {
          for (int dj = 0; dj < 2; ++dj) {
            res(2 * i + di, 2 * j + dj) = (*this)(i, j) * 0.25;
          }
        }
      }
    }
    return res;
  }
  // Interpolates array to the next finer level.
  Matrix interpolate() const {
    const Matrix& u = *this;
    Matrix ufine(u.nrow_ * 2, u.ncol_ * 2);
    auto interp = [&u](size_t i, size_t ip, size_t j, size_t jp, T dx, T dy) {
      return (u(i, j) * (1 - dx) + u(ip, j) * dx) * (1 - dy) +
             (u(i, jp) * (1 - dx) + u(ip, jp) * dx) * dy;
    };
    // Inner cells.
    for (size_t i = 0; i + 1 < u.nrow_; ++i) {
      for (size_t j = 0; j + 1 < u.ncol_; ++j) {
        for (int di = 0; di < 2; ++di) {
          for (int dj = 0; dj < 2; ++dj) {
            const T dx = 0.25 + di * 0.5;
            const T dy = 0.25 + dj * 0.5;
            ufine(2 * i + 1 + di, 2 * j + 1 + dj) =
                interp(i, i + 1, j, j + 1, dx, dy);
          }
        }
      }
    }
    // Boundary cells with linear extrapolation.
    for (size_t fi = 0; fi < ufine.nrow_; ++fi) {
      for (size_t fj = 0; fj < ufine.ncol_; ++fj) {
        const size_t i = (fi == 0 || u.nrow_ == 1 ? 0
                          : fi == ufine.nrow_ - 1 ? u.nrow_ - 2
                                                  : (fi - 1) / 2);
        const size_t j = (fj == 0 || u.ncol_ == 1 ? 0
                          : fj == ufine.ncol_ - 1 ? u.ncol_ - 2
                                                  : (fj - 1) / 2);
        const size_t ip = (i + 1 < u.nrow_ ? i + 1 : i);
        const size_t jp = (j + 1 < u.ncol_ ? j + 1 : j);
        const T dx = 0.5 * (fi - 2 * i) - 0.25;
        const T dy = 0.5 * (fj - 2 * j) - 0.25;
        ufine(fi, fj) = interp(i, ip, j, jp, dx, dy);
        if (fi > 0 && fi + 1 < ufine.nrow_) {
          // Jump to the opposite side.
          fj += ufine.ncol_ - 2;
        }
      }
    }
    return ufine;
  }
  // Adjoint of interpolate().
  Matrix interpolate_adjoint() const {
    fassert(nrow_ % 2 == 0);
    fassert(ncol_ % 2 == 0);
    const Matrix& ufine = *this;
    Matrix u(ufine.nrow_ / 2, ufine.ncol_ / 2, T(0));
    auto interp = [&u](size_t i, size_t ip, size_t j, size_t jp, T dx, T dy,
                       T value) {
      u(i, j) += (1 - dx) * (1 - dy) * value;
      u(ip, j) += dx * (1 - dy) * value;
      u(i, jp) += (1 - dx) * dy * value;
      u(ip, jp) += dx * dy * value;
    };
    // Inner cells.
    for (size_t i = 0; i + 1 < u.nrow_; ++i) {
      for (size_t j = 0; j + 1 < u.ncol_; ++j) {
        for (int di = 0; di < 2; ++di) {
          for (int dj = 0; dj < 2; ++dj) {
            const T dx = 0.25 + di * 0.5;
            const T dy = 0.25 + dj * 0.5;
            interp(i, i + 1, j, j + 1, dx, dy,
                   ufine(2 * i + 1 + di, 2 * j + 1 + dj));
          }
        }
      }
    }
    // Boundary cells with linear extrapolation.
    for (size_t fi = 0; fi < ufine.nrow_; ++fi) {
      for (size_t fj = 0; fj < ufine.ncol_; ++fj) {
        const size_t i = (fi == 0 || u.nrow_ == 1 ? 0
                          : fi == ufine.nrow_ - 1 ? u.nrow_ - 2
                                                  : (fi - 1) / 2);
        const size_t j = (fj == 0 || u.ncol_ == 1 ? 0
                          : fj == ufine.ncol_ - 1 ? u.ncol_ - 2
                                                  : (fj - 1) / 2);
        const size_t ip = (i + 1 < u.nrow_ ? i + 1 : i);
        const size_t jp = (j + 1 < u.ncol_ ? j + 1 : j);
        const T dx = 0.5 * (fi - 2 * i) - 0.25;
        const T dy = 0.5 * (fj - 2 * j) - 0.25;
        interp(i, ip, j, jp, dx, dy, ufine(fi, fj));
        if (fi > 0 && fi + 1 < ufine.nrow_) {
          // Jump to the opposite side.
          fj += ufine.ncol_ - 2;
        }
      }
    }
    return u;
  }
  Matrix conv(const T& a, const T& axm, const T& axp, const T& aym,
              const T& ayp) const {
    Matrix res(nrow_, ncol_);
    auto& u = *this;
    for (size_t ix = 0; ix < nrow_; ++ix) {
      for (size_t iy = 0; iy < ncol_; ++iy) {
        const size_t ixm = (ix == 0 ? nrow_ - 1 : ix - 1);
        const size_t ixp = (ix + 1 == nrow_ ? 0 : ix + 1);
        const size_t iym = (iy == 0 ? ncol_ - 1 : iy - 1);
        const size_t iyp = (iy + 1 == ncol_ ? 0 : iy + 1);
        res(ix, iy) = a * u(ix, iy) + axm * u(ixm, iy) + axp * u(ixp, iy) +
                      aym * u(ix, iym) + ayp * u(ix, iyp);
      }
    }
    return res;
  }

  // Reduction.
  T sum() const {
    return std::accumulate(data_.begin(), data_.end(), T{});
  }
  T mean() const {
    return sum() / size();
  }
  T dot(const Matrix& other) const {
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    return (*this * other).sum();
  }
  T min() const {
    return *std::min_element(data_.begin(), data_.end());
  }
  T max() const {
    return *std::max_element(data_.begin(), data_.end());
  }
  template <class F>
  auto apply(F func) const -> Matrix<decltype(func(T()))> const {
    Matrix<decltype(func(T()))> res(ncol_, nrow_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = func((*this)[i]);
    }
    return res;
  }
  Matrix matmul(const Matrix& other) const {
    fassert_equal(ncol_, other.nrow_);
    const auto other_t = other.transpose();
    Matrix res(nrow_, other.ncol_);
    for (size_t i = 0; i < res.nrow_; ++i) {
      for (size_t j = 0; j < res.ncol_; ++j) {
        T s{};
        for (size_t k = 0; k < ncol_; ++k) {
          s += (*this)(i, k) * other_t(j, k);
        }
        res(i, j) = s;
      }
    }
    return res;
  }
  Matrix hstack(const Matrix& other) const {
    fassert_equal(nrow_, other.nrow_);
    Matrix res(nrow_, ncol_ + other.ncol_);
    for (size_t i = 0; i < nrow_; ++i) {
      for (size_t j = 0; j < ncol_; ++j) {
        res(i, j) = (*this)(i, j);
      }
      for (size_t j = 0; j < other.ncol_; ++j) {
        res(i, ncol_ + j) = other(i, j);
      }
    }
    return res;
  }
  Matrix vstack(const Matrix& other) const {
    fassert_equal(ncol_, other.ncol_);
    Matrix res(nrow_ + other.nrow_, ncol_);
    for (size_t i = 0; i < nrow_; ++i) {
      for (size_t j = 0; j < ncol_; ++j) {
        res(i, j) = (*this)(i, j);
      }
    }
    for (size_t i = 0; i < other.nrow_; ++i) {
      for (size_t j = 0; j < ncol_; ++j) {
        res(nrow_ + i, j) = other(i, j);
      }
    }
    return res;
  }

  // Friend functions.
  friend Matrix sin(const Matrix& matr) {
    using std::sin;
    return matr.apply([](T x) { return sin(x); });
  }
  friend Matrix cos(const Matrix& matr) {
    using std::cos;
    return matr.apply([](T x) { return cos(x); });
  }
  friend Matrix exp(const Matrix& matr) {
    using std::exp;
    return matr.apply([](T x) { return exp(x); });
  }
  friend Matrix log(const Matrix& matr) {
    using std::log;
    return matr.apply([](T x) { return log(x); });
  }
  friend Matrix sqr(const Matrix& matr) {
    return matr.apply([](T x) { return sqr(x); });
  }
  friend Matrix sqrt(const Matrix& matr) {
    using std::sqrt;
    return matr.apply([](T x) { return sqrt(x); });
  }
  friend T sum(const Matrix& matr) {
    return matr.sum();
  }
  friend T mean(const Matrix& matr) {
    return matr.mean();
  }
  friend T rms(const Matrix& matr) {
    using std::sqrt;
    return sqrt(sqr(matr).mean());
  }
  friend T dot(const Matrix& matr, const Matrix& other) {
    return matr.dot(other);
  }
  friend Matrix transpose(const Matrix& matr) {
    return matr.transpose();
  }
  friend Matrix restrict(const Matrix& matr) {
    return matr.restrict();
  }
  friend Matrix restrict_adjoint(const Matrix& matr) {
    return matr.restrict_adjoint();
  }
  friend Matrix interpolate(const Matrix& matr) {
    return matr.interpolate();
  }
  friend Matrix interpolate_adjoint(const Matrix& matr) {
    return matr.interpolate_adjoint();
  }
  friend Matrix operator+(const T& a, const Matrix& matr) {
    Matrix res(matr.nrow_, matr.ncol_);
    for (size_t i = 0; i < matr.data_.size(); ++i) {
      res[i] = a + matr[i];
    }
    return res;
  }
  friend Matrix operator-(const T& a, const Matrix& matr) {
    Matrix res(matr.nrow_, matr.ncol_);
    for (size_t i = 0; i < matr.data_.size(); ++i) {
      res[i] = a - matr[i];
    }
    return res;
  }
  friend Matrix operator*(const T& a, const Matrix& matr) {
    Matrix res(matr.nrow_, matr.ncol_);
    for (size_t i = 0; i < matr.data_.size(); ++i) {
      res[i] = a * matr[i];
    }
    return res;
  }
  friend Matrix operator/(const T& a, const Matrix& matr) {
    Matrix res(matr.nrow_, matr.ncol_);
    for (size_t i = 0; i < matr.data_.size(); ++i) {
      res[i] = a / matr[i];
    }
    return res;
  }
  friend Matrix roll(const Matrix& matr, int shift_row, int shift_col) {
    return matr.roll(shift_row, shift_col);
  }
  template <class U>
  friend Matrix conv(const Matrix& matr, const U& a, const U& axm, const U& axp,
                     const U& aym, const U& ayp) {
    return matr.conv(a, axm, axp, aym, ayp);
  }

  // Static functions.
  static Matrix zeros(size_t nrow, size_t ncol) {
    return Matrix(nrow, ncol, T{});
  }
  static Matrix zeros(size_t n) {
    return Matrix::zeros(n, n);
  }
  static Matrix ones(size_t nrow, size_t ncol) {
    return Matrix(nrow, ncol, T(1));
  }
  static Matrix ones(size_t n) {
    return Matrix::ones(n, n);
  }
  template <class U>
  static Matrix zeros_like(const Matrix<U>& other) {
    return Matrix(other.nrow_, other.ncol_, T{});
  }
  template <class U>
  static Matrix ones_like(const Matrix<U>& other) {
    return Matrix(other.nrow_, other.ncol_, T(1));
  }
  static Matrix eye(size_t nrow, size_t ncol) {
    Matrix res(nrow, ncol);
    for (size_t i = 0; i < nrow; ++i) {
      for (size_t j = 0; j < ncol; ++j) {
        res(i, j) = (i == j ? T(1) : T(0));
      }
    }
    return res;
  }
  static Matrix eye(size_t n) {
    return Matrix::eye(n, n);
  }
  static Matrix diag(const std::vector<T>& data) {
    auto res = Matrix::zeros(data.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      res(i, i) = data[i];
    }
    return res;
  }
  static Matrix iota(size_t nrow, size_t ncol) {
    Matrix res(nrow, ncol);
    for (size_t i = 0; i < res.size(); ++i) {
      res[i] = i;
    }
    return res;
  }
  static Matrix iota(size_t n) {
    return Matrix::iota(n, n);
  }

 private:
  size_t nrow_;
  size_t ncol_;
  std::vector<T> data_;
};

////////////////////////////////////////
// Output.
////////////////////////////////////////

template <class T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& matr) {
  out << '[';
  for (size_t i = 0; i < matr.nrow(); ++i) {
    out << '[';
    for (size_t j = 0; j < matr.ncol(); ++j) {
      out << matr(i, j);
      if (j + 1 < matr.ncol()) {
        out << ", ";
      }
    }
    out << ']';
    if (i + 1 < matr.nrow()) {
      out << ",";
    }
  }
  out << ']';
  return out;
}

template <class T>
std::string MatrixToStr(const Matrix<T>& matr, int width = 3, int precision = 6,
                        bool fixed = false) {
  std::stringstream out;
  if (fixed) {
    out << std::fixed;
  }
  out.precision(precision);
  for (size_t i = 0; i < matr.nrow(); ++i) {
    for (size_t j = 0; j < matr.ncol(); ++j) {
      out << std::setw(width) << matr(i, j);
      if (j + 1 < matr.ncol()) {
        out << ' ';
      }
    }
    if (i + 1 < matr.nrow()) {
      out << '\n';
    }
  }
  return out.str();
};
