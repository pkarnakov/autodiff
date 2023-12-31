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
  // Constructor.
  Matrix() : nrow_(0), ncol_(0) {}
  explicit Matrix(size_t n) : Matrix(n, n) {}
  Matrix(size_t nrow, size_t ncol)
      : nrow_(nrow), ncol_(ncol), data_(nrow * ncol) {}
  Matrix(size_t nrow, size_t ncol, T a)
      : nrow_(nrow), ncol_(ncol), data_(nrow * ncol, a) {}
  template <class U>
  Matrix(const Matrix<U>& other)
      : nrow_(other.nrow_),
        ncol_(other.ncol_),
        data_(other.data_.begin(), other.data_.end()) {}

  template <class U>
  friend class Matrix;

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
    Idx icnt;  // Elements to copy.

    // Flips the rectangle along axis k.
    auto flip = [&](int k) {
      idst[k] = (idst[k] == 0 ? icnt[k] : 0);
      isrc[k] = (isrc[k] == 0 ? icnt[k] : 0);
      icnt[k] = shape[k] - icnt[k];
    };
    // Copies the rectangle to dst from src.
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
  T sum() const {
    return std::accumulate(data_.begin(), data_.end(), T(0));
  }
  T mean() const {
    return sum() / size();
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

  // Static functions.
  static Matrix zeros(size_t nrow, size_t ncol) {
    return Matrix(nrow, ncol, T(0));
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
    return Matrix(other.nrow_, other.ncol_, T(0));
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
