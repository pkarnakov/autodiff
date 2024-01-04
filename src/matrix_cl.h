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

  class Entry {
   public:
    Entry(MatrixCL& matr, int i, int j) : matr_(matr), i_(i), j_(j) {}
    Entry& operator=(T value) {
      matr_.write(i_, j_, value);
      return *this;
    }
    operator T() const {
      return matr_.read(i_, j_);
    }

   private:
    MatrixCL& matr_;
    int i_;
    int j_;
  };

  // Constructor.
  MatrixCL(size_t nrow, size_t ncol, CL& cl_)
      : cl(cl_), nrow_(nrow), ncol_(ncol), data_(cl.context(), nrow * ncol) {}
  MatrixCL(MatrixCL&&) = default;
  MatrixCL(const Matrix<T>& other, CL& cl_)
      : MatrixCL(other.nrow(), other.ncol(), cl_) {
    data_.EnqueueWrite(cl.queue(), other.data());
  }

  // Element access.
  T read(int i, int j) const {
    return cl.ReadAt<T>(data_, i, j);
  }
  void write(int i, int j, T value) {
    return cl.WriteAt(data_, i, j, value);
  }
  T operator()(int i, int j) const {
    return read(i, j);
  }
  Entry operator()(int i, int j) {
    return Entry(*this, i, j);
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
  MatrixCL operator+(T a) const {
    MatrixCL res(nrow_, ncol_, cl);
    cl.Add(data_, a, res.data_);
    return res;
  }
  MatrixCL operator-(T a) const {
    MatrixCL res(nrow_, ncol_, cl);
    cl.Sub(data_, a, res.data_);
    return res;
  }
  MatrixCL operator*(T a) const {
    MatrixCL res(nrow_, ncol_, cl);
    cl.Mul(data_, a, res.data_);
    return res;
  }
  MatrixCL operator/(T a) const {
    MatrixCL res(nrow_, ncol_, cl);
    cl.Div(data_, a, res.data_);
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
  T min() const {
    return cl.Min(data_);
  }

  // Friend functions.
  friend MatrixCL sin(const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl.Sin(matr.data_, res.data_);
    return res;
  }
  friend MatrixCL cos(const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl.Cos(matr.data_, res.data_);
    return res;
  }
  friend MatrixCL exp(const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl.Exp(matr.data_, res.data_);
    return res;
  }
  friend MatrixCL log(const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl.Log(matr.data_, res.data_);
    return res;
  }
  friend MatrixCL operator+(T a, const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl.Add(matr.data_, a, res.data_);
    return res;
  }
  friend MatrixCL operator-(T a, const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl.Sub(a, matr.data_, res.data_);
    return res;
  }
  friend MatrixCL operator*(T a, const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl.Mul(matr.data_, a, res.data_);
    return res;
  }
  friend MatrixCL operator/(T a, const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl.Div(a, matr.data_, res.data_);
    return res;
  }

 private:
  CL& cl;
  size_t nrow_;
  size_t ncol_;
  CL::Buffer<T> data_;
};
