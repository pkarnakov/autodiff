#pragma once

#include <cmath>
#include <iosfwd>
#include <vector>

#include "macros.h"

template <class U, class V>
using Mix = decltype(U() + V());

template <class T>
class Matrix {
 public:
  // Constructor.
  Matrix() : nrow_(0), ncol_(0) {}
  Matrix(size_t nrow, size_t ncol)
      : nrow_(nrow), ncol_(ncol), data_(nrow * ncol) {}
  Matrix(size_t nrow, size_t ncol, T a)
      : nrow_(nrow), ncol_(ncol), data_(nrow * ncol, a) {}

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

  // Assignment.
  Matrix& operator=(const Matrix&) = default;
  Matrix& operator+=(const Matrix& other) {
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      data_[i] += other.data_[i];
    }
    return *this;
  }
  Matrix& operator-=(const Matrix& other) {
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
  template <class U>
  Matrix<Mix<U, T>> operator+(const Matrix<U>& other) const {
    Matrix<Mix<U, T>> res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = data_[i] + other[i];
    }
    return res;
  }
  template <class U>
  Matrix<Mix<U, T>> operator-(const Matrix<U>& other) const {
    Matrix<Mix<U, T>> res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = data_[i] - other[i];
    }
    return res;
  }
  template <class U>
  Matrix<Mix<U, T>> operator*(const Matrix<U>& other) const {
    using R = Mix<U, T>;
    fassert_equal(ncol_, other.nrow_);
    const auto other_t = other.transpose();
    Matrix<R> res(nrow_, other.ncol_, R(0));
    for (size_t i = 0; i < res.nrow_; ++i) {
      for (size_t j = 0; j < res.ncol_; ++j) {
        R s(0);
        for (size_t k = 0; k < ncol_; ++k) {
          s += (*this)(i, k) * other_t(j, k);
        }
        res(i, j) = s;
      }
    }
    return res;
  }
  template <class U>
  Matrix<Mix<U, T>> operator+(const U& a) const {
    Matrix<Mix<U, T>> res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = data_[i] + a;
    }
    return res;
  }
  Matrix operator-() const {
    Matrix res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = -data_[i];
    }
    return res;
  }
  template <class U>
  Matrix<Mix<U, T>> operator-(const U& a) const {
    Matrix<Mix<U, T>> res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = data_[i] - a;
    }
    return res;
  }
  template <class U>
  Matrix<Mix<U, T>> operator*(const U& a) const {
    Matrix<Mix<U, T>> res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = data_[i] * a;
    }
    return res;
  }
  template <class U>
  Matrix<Mix<U, T>> operator/(const U& a) const {
    Matrix<Mix<U, T>> res(nrow_, ncol_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = data_[i] / a;
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
  Matrix apply(F func) const {
    Matrix res(ncol_, nrow_);
    for (size_t i = 0; i < data_.size(); ++i) {
      res[i] = func((*this)[i]);
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
  /* FIXME: Preceeds matrix-matrix product if T=Matrix.
  template <class U>
  friend Matrix<Mix<U, T>> operator*(const U& a, const Matrix& matr) {
    Matrix<Mix<U, T>> res(matr.nrow_, matr.ncol_);
    for (size_t i = 0; i < matr.data_.size(); ++i) {
      res[i] = a * matr[i];
    }
    return res;
  }
  */
  friend std::ostream& operator<<(std::ostream& out, const Matrix& matr) {
    for (size_t i = 0; i < matr.nrow_; ++i) {
      for (size_t j = 0; j < matr.ncol_; ++j) {
        out << matr.data_[i * matr.ncol_ + j] << " ";
      }
      out << "\n";
    }
    return out;
  }
  static Matrix zeros(size_t nrow, size_t ncol) {
    return Matrix(nrow, ncol, T(0));
  }
  static Matrix eye(size_t nrow, size_t ncol) {
    Matrix<T> res(nrow, ncol);
    for (size_t i = 0; i < nrow; ++i) {
      for (size_t j = 0; j < ncol; ++j) {
        res(i, j) = (i == j ? 1 : 0);
      }
    }
    return res;
  }

 private:
  size_t nrow_;
  size_t ncol_;
  std::vector<T> data_;
};
