#pragma once

#include <cmath>
#include <iosfwd>
#include <vector>

#include "macros.h"

template <class T>
class Matrix {
 public:
  // Constructor.
  Matrix() : nrow_(0), ncol_(0) {}
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
  T sum() const {
    T res(0);
    for (size_t i = 0; i < data_.size(); ++i) {
      res += data_[i];
    }
    return res;
  }
  T mean() const {
    return sum() / (nrow_ * ncol_);
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
