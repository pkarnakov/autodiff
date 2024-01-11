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
  using MSize = OpenCL::MSize;
  using MInt = OpenCL::MInt;
  template <class U>
  friend class Matrix;

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
  MatrixCL() = default;
  MatrixCL(MSize nw, CL* cl_)
      : cl(cl_), nw_(nw), data_(cl->context(), nw_[0] * nw_[1]) {}
  MatrixCL(MSize nw, CL& cl_) : MatrixCL(nw, &cl_) {}
  MatrixCL(size_t nrow, size_t ncol, CL* cl_) : MatrixCL({nrow, ncol}, cl_) {}
  MatrixCL(size_t nrow, size_t ncol, CL& cl_) : MatrixCL({nrow, ncol}, &cl_) {}
  MatrixCL(MSize nw, T value, CL* cl_) : MatrixCL(nw, cl_) {
    this->fill(value);
  }
  MatrixCL(size_t nrow, size_t ncol, T value, CL* cl_)
      : MatrixCL({nrow, ncol}, value, cl_) {}
  MatrixCL(const MatrixCL& other) : MatrixCL(other.nw_, other.cl) {
    fassert(data_.handle);
    fassert(other.data_.handle);
    data_.EnqueueWriteBuffer(cl->queue(), other.data_);
  }
  MatrixCL(MatrixCL&&) = default;
  MatrixCL(const Matrix<T>& other, CL& cl_)
      : MatrixCL(other.nrow(), other.ncol(), cl_) {
    data_.EnqueueWrite(cl->queue(), other.data());
  }

  // Element access.
  T read(int i, int j) const {
    return cl->ReadAt<T>(nw_, data_, {i, j});
  }
  void write(int i, int j, T value) {
    return cl->WriteAt(nw_, data_, {i, j}, value);
  }
  T operator()(int i, int j) const {
    return read(i, j);
  }
  Entry operator()(int i, int j) {
    return Entry(*this, i, j);
  }
  size_t nrow() const {
    return nw_[0];
  }
  size_t ncol() const {
    return nw_[1];
  }
  size_t size() const {
    return nw_[0] * nw_[1];
  }

  // Assignment.
  MatrixCL& operator=(const MatrixCL& other) {
    fassert(other.cl, "Can't assign add an empty matrix");
    if (cl == nullptr) {
      cl = other.cl;
      nw_ = other.nw_;
      CL::Buffer<T> buf(cl->context(), nw_[0] * nw_[1]);
      data_.swap(buf);
    }
    fassert_equal(nw_[0], other.nw_[0]);
    fassert_equal(nw_[1], other.nw_[1]);
    fassert(data_.handle);
    fassert(other.data_.handle);
    data_.EnqueueWriteBuffer(cl->queue(), other.data_);
    return *this;
  }
  MatrixCL& operator+=(const MatrixCL& other) {
    fassert(other.cl, "Can't assign add an empty matrix");
    if (cl == nullptr) {
      cl = other.cl;
      nw_ = other.nw_;
      CL::Buffer<T> buf(cl->context(), nw_[0] * nw_[1]);
      data_.swap(buf);
    }
    fassert_equal(nw_[0], other.nw_[0]);
    fassert_equal(nw_[1], other.nw_[1]);
    cl->AssignAdd(nw_, data_, other.data_);
    return *this;
  }
  MatrixCL& operator-=(const MatrixCL& other) {
    fassert(other.cl, "Can't assign add an empty matrix");
    if (cl == nullptr) {
      cl = other.cl;
      nw_ = other.nw_;
      CL::Buffer<T> buf(cl->context(), nw_[0] * nw_[1]);
      data_.swap(buf);
    }
    fassert_equal(nw_[0], other.nw_[0]);
    fassert_equal(nw_[1], other.nw_[1]);
    cl->AssignSub(nw_, data_, other.data_);
    return *this;
  }
  void fill(T value) const {
    return cl->Fill(nw_, data_, value);
  }
  void clear() const {
    if (cl) {
      cl->Fill(nw_, data_, T(0));
    }
  }
  explicit operator Matrix<T>() const {
    Matrix<T> res(nw_[0], nw_[1]);
    data_.EnqueueRead(cl->queue(), res.data());
    return res;
  }

  // Member functions.
  MatrixCL operator-() const {
    MatrixCL res(nw_, cl);
    cl->Mul(nw_, data_, -1, res.data_);
    return res;
  }
  MatrixCL operator+(const MatrixCL& other) const {
    fassert_equal(cl, other.cl);
    fassert_equal(nw_[0], other.nw_[0]);
    fassert_equal(nw_[1], other.nw_[1]);
    MatrixCL res(nw_, cl);
    cl->Add(nw_, data_, other.data_, res.data_);
    return res;
  }
  MatrixCL operator-(const MatrixCL& other) const {
    fassert_equal(cl, other.cl);
    fassert_equal(nw_[0], other.nw_[0]);
    fassert_equal(nw_[1], other.nw_[1]);
    MatrixCL res(nw_, cl);
    cl->Sub(nw_, data_, other.data_, res.data_);
    return res;
  }
  MatrixCL operator*(const MatrixCL& other) const {
    fassert_equal(cl, other.cl);
    fassert_equal(nw_[0], other.nw_[0]);
    fassert_equal(nw_[1], other.nw_[1]);
    MatrixCL res(nw_, cl);
    cl->Mul(nw_, data_, other.data_, res.data_);
    return res;
  }
  MatrixCL operator/(const MatrixCL& other) const {
    fassert_equal(cl, other.cl);
    fassert_equal(nw_[0], other.nw_[0]);
    fassert_equal(nw_[1], other.nw_[1]);
    MatrixCL res(nw_, cl);
    cl->Div(nw_, data_, other.data_, res.data_);
    return res;
  }
  MatrixCL operator+(T a) const {
    MatrixCL res(nw_, cl);
    cl->Add(nw_, data_, a, res.data_);
    return res;
  }
  MatrixCL operator-(T a) const {
    MatrixCL res(nw_, cl);
    cl->Sub(nw_, data_, a, res.data_);
    return res;
  }
  MatrixCL operator*(T a) const {
    MatrixCL res(nw_, cl);
    cl->Mul(nw_, data_, a, res.data_);
    return res;
  }
  MatrixCL operator/(T a) const {
    MatrixCL res(nw_, cl);
    cl->Div(nw_, data_, a, res.data_);
    return res;
  }
  MatrixCL roll(int shift_row, int shift_col) const {
    MatrixCL res(nw_, cl);
    cl->Roll(nw_, data_, {shift_col, shift_row}, res.data_);
    return res;
  }
  MatrixCL restrict() const {
    fassert(nw_[0] % 2 == 0);
    fassert(nw_[1] % 2 == 0);
    MatrixCL res(nw_[0] / 2, nw_[1] / 2, cl);
    cl->Restrict(res.nw_, data_, res.data_);
    return res;
  }
  MatrixCL restrict_adjoint() const {
    MatrixCL res(nw_[0] * 2, nw_[1] * 2, cl);
    cl->RestrictAdjoint(nw_, data_, res.data_);
    return res;
  }
  MatrixCL interpolate() const {
    MatrixCL res(nw_[0] * 2, nw_[1] * 2, cl);
    cl->Interpolate(res.nw_, data_, res.data_);
    return res;
  }
  MatrixCL interpolate_adjoint() const {
    fassert(nw_[0] % 2 == 0);
    fassert(nw_[1] % 2 == 0);
    MatrixCL res({nw_[0] / 2, nw_[1] / 2}, cl);
    cl->InterpolateAdjoint(res.nw_, data_, res.data_);
    return res;
  }
  MatrixCL conv(const T& a, const T& axm, const T& axp, const T& aym,
                const T& ayp) const {
    MatrixCL res(nw_, cl);
    if (nw_[0] == 0 || nw_[1] == 0) {
      return res;
    }
    // Swapping the order of x and y.
    // TODO: Revise to make x the slow index everywhere.
    cl->Conv(nw_, data_, a, aym, ayp, axm, axp, res.data_);
    return res;
  }

  // Reduction.
  T sum() const {
    return cl->Sum(nw_, data_);
  }
  T mean() const {
    return sum() / size();
  }
  T dot(const MatrixCL& other) const {
    fassert_equal(cl, other.cl);
    return cl->Dot(nw_, data_, other.data_);
  }
  T max() const {
    return cl->Max(nw_, data_);
  }
  T min() const {
    return cl->Min(nw_, data_);
  }

  // Friend functions.
  friend MatrixCL sin(const MatrixCL& matr) {
    MatrixCL res(matr.nw_, matr.cl);
    matr.cl->Sin(matr.nw_, matr.data_, res.data_);
    return res;
  }
  friend MatrixCL cos(const MatrixCL& matr) {
    MatrixCL res(matr.nw_, matr.cl);
    matr.cl->Cos(matr.nw_, matr.data_, res.data_);
    return res;
  }
  friend MatrixCL exp(const MatrixCL& matr) {
    MatrixCL res(matr.nw_, matr.cl);
    matr.cl->Exp(matr.nw_, matr.data_, res.data_);
    return res;
  }
  friend MatrixCL log(const MatrixCL& matr) {
    MatrixCL res(matr.nw_, matr.cl);
    matr.cl->Log(matr.nw_, matr.data_, res.data_);
    return res;
  }
  friend MatrixCL sqr(const MatrixCL& matr) {
    MatrixCL res(matr.nw_, matr.cl);
    matr.cl->Sqr(matr.nw_, matr.data_, res.data_);
    return res;
  }
  friend MatrixCL sqrt(const MatrixCL& matr) {
    MatrixCL res(matr.nw_, matr.cl);
    matr.cl->Sqrt(matr.nw_, matr.data_, res.data_);
    return res;
  }
  friend T sum(const MatrixCL& matr) {
    return matr.sum();
  }
  friend T mean(const MatrixCL& matr) {
    return matr.mean();
  }
  friend T rms(const MatrixCL& matr) {
    using std::sqrt;
    return sqrt(sqr(matr).mean());
  }
  friend T dot(const MatrixCL& matr, const MatrixCL& other) {
    return matr.dot(other);
  }
  friend MatrixCL transpose(const MatrixCL& matr) {
    return matr.transpose();
  }
  friend MatrixCL operator+(T a, const MatrixCL& matr) {
    MatrixCL res(matr.nw_, matr.cl);
    matr.cl->Add(matr.nw_, matr.data_, a, res.data_);
    return res;
  }
  friend MatrixCL operator-(T a, const MatrixCL& matr) {
    MatrixCL res(matr.nw_, matr.cl);
    matr.cl->Sub(matr.nw_, a, matr.data_, res.data_);
    return res;
  }
  friend MatrixCL operator*(T a, const MatrixCL& matr) {
    MatrixCL res(matr.nw_, matr.cl);
    matr.cl->Mul(matr.nw_, matr.data_, a, res.data_);
    return res;
  }
  friend MatrixCL operator/(T a, const MatrixCL& matr) {
    MatrixCL res(matr.nw_, matr.cl);
    matr.cl->Div(matr.nw_, a, matr.data_, res.data_);
    return res;
  }
  friend MatrixCL roll(const MatrixCL& matr, int shift_row, int shift_col) {
    return matr.roll(matr.nw_, {shift_row, shift_col});
  }

  template <class U>
  friend MatrixCL conv(const MatrixCL& matr, const U& a, const U& axm,
                       const U& axp, const U& aym, const U& ayp) {
    return matr.conv(a, axm, axp, aym, ayp);
  }

  // Static functions.
  static MatrixCL zeros(size_t nrow, size_t ncol, CL& cl) {
    return MatrixCL(nrow, ncol, T(0), &cl);
  }
  static MatrixCL zeros(size_t n, CL& cl) {
    return MatrixCL::zeros(n, n, cl);
  }
  static MatrixCL ones(size_t nrow, size_t ncol, CL& cl) {
    return MatrixCL(nrow, ncol, T(1), &cl);
  }
  static MatrixCL ones(size_t n, CL& cl) {
    return MatrixCL::ones(n, n, cl);
  }
  template <class U>
  static MatrixCL zeros_like(const MatrixCL<U>& other) {
    return MatrixCL(other.nw_, T(0), other.cl);
  }
  template <class U>
  static MatrixCL ones_like(const MatrixCL<U>& other) {
    return MatrixCL(other.nw_, T(1), other.cl);
  }

 private:
  CL* cl = nullptr;
  MSize nw_;
  CL::Buffer<T> data_;
};

////////////////////////////////////////
// Output.
////////////////////////////////////////

template <class T>
std::ostream& operator<<(std::ostream& out, const MatrixCL<T>& matr) {
  out << Matrix<T>(matr);
  return out;
}

template <class T>
std::string MatrixToStr(const MatrixCL<T>& matr, int width = 3,
                        int precision = 6, bool fixed = false) {
  return MatrixToStr(Matrix<T>(matr), width, precision, fixed);
};
