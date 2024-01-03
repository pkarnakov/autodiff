#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iosfwd>
#include <numeric>
#include <vector>

#include "macros.h"
#include "matrix.h"
#include "opencl.h"

template <class T>
class MatrixCL {
 public:
  using CL = OpenCL;

  // Constructor.
  MatrixCL(size_t nrow, size_t ncol, CL& cl_)
      : cl(cl_), nrow_(nrow), ncol_(ncol), data_(cl.context(), nrow * ncol) {}
  MatrixCL(MatrixCL&&) = default;
  MatrixCL(const Matrix<T>& other, CL& cl_)
      : MatrixCL(other.nrow(), other.ncol(), cl_) {
    data_.EnqueueWrite(cl.queue(), other.data());
  }

  // Element access.
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
  MatrixCL& operator=(const MatrixCL&) = delete;
  void fill(T value) const {
    return cl.Fill(data_, value);
  }

  explicit operator Matrix<T>() const {
    Matrix<T> res(nrow_, ncol_, 2);
    data_.EnqueueRead(cl.queue(), res.data());
    return res;
  }

  // Member functions.
  MatrixCL operator+(const MatrixCL& other) const {
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    MatrixCL res(nrow_, ncol_, cl);
    cl.Add(data_, other.data_, res.data_);
    return res;
  }
  MatrixCL operator-(const MatrixCL& other) const {
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    MatrixCL res(nrow_, ncol_, cl);
    cl.Sub(data_, other.data_, res.data_);
    return res;
  }
  MatrixCL operator*(const MatrixCL& other) const {
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    MatrixCL res(nrow_, ncol_, cl);
    cl.Mul(data_, other.data_, res.data_);
    return res;
  }
  MatrixCL operator/(const MatrixCL& other) const {
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    MatrixCL res(nrow_, ncol_, cl);
    cl.Div(data_, other.data_, res.data_);
    return res;
  }

  // Reduction.
  T sum() const {
    return cl.Sum(data_);
  }
  T mean() const {
    return sum() / size();
  }
  T dot(const MatrixCL& other) const {
    fassert_equal(&cl, &other.cl);
    return cl.Dot(data_, other.data_);
  }
  T max() const {
    return cl.Max(data_);
  }

 private:
  CL& cl;
  size_t nrow_;
  size_t ncol_;
  CL::Buffer<T> data_;
};
