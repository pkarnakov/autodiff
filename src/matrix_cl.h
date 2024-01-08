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
  MatrixCL(size_t nrow, size_t ncol, CL* cl_)
      : cl(cl_),
        nrow_(nrow),
        ncol_(ncol),
        data_(cl->context(), nrow_ * ncol_) {}
  MatrixCL(size_t nrow, size_t ncol, CL& cl_) : MatrixCL(nrow, ncol, &cl_) {}
  MatrixCL(size_t nrow, size_t ncol, T value, CL* cl_)
      : MatrixCL(nrow, ncol, cl_) {
    this->fill(value);
  }
  MatrixCL(const MatrixCL& other)
      : MatrixCL(other.nrow_, other.ncol_, other.cl) {
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
    return cl->ReadAt<T>(data_, i, j);
  }
  void write(int i, int j, T value) {
    return cl->WriteAt(data_, i, j, value);
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
  MatrixCL& operator=(const MatrixCL& other) {
    fassert(other.cl, "Can't assign add an empty matrix");
    if (cl == nullptr) {
      cl = other.cl;
      nrow_ = other.nrow_;
      ncol_ = other.ncol_;
      CL::Buffer<T> buf(cl->context(), nrow_ * ncol_);
      data_.swap(buf);
    }
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    fassert(data_.handle);
    fassert(other.data_.handle);
    data_.EnqueueWriteBuffer(cl->queue(), other.data_);
    return *this;
  }
  MatrixCL& operator+=(const MatrixCL& other) {
    fassert(other.cl, "Can't assign add an empty matrix");
    if (cl == nullptr) {
      cl = other.cl;
      nrow_ = other.nrow_;
      ncol_ = other.ncol_;
      CL::Buffer<T> buf(cl->context(), nrow_ * ncol_);
      data_.swap(buf);
    }
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    cl->AssignAdd(data_, other.data_);
    return *this;
  }
  MatrixCL& operator-=(const MatrixCL& other) {
    fassert(other.cl, "Can't assign add an empty matrix");
    if (cl == nullptr) {
      cl = other.cl;
      nrow_ = other.nrow_;
      ncol_ = other.ncol_;
      CL::Buffer<T> buf(cl->context(), nrow_ * ncol_);
      data_.swap(buf);
    }
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    cl->AssignSub(data_, other.data_);
    return *this;
  }
  void fill(T value) const {
    return cl->Fill(data_, value);
  }
  void clear() const {
    if (cl) {
      cl->Fill(data_, T(0));
    }
  }
  explicit operator Matrix<T>() const {
    Matrix<T> res(nrow_, ncol_, 2);
    data_.EnqueueRead(cl->queue(), res.data());
    return res;
  }

  // Member functions.
  MatrixCL operator-() const {
    MatrixCL res(nrow_, ncol_, cl);
    cl->Mul(data_, -1, res.data_);
    return res;
  }
  MatrixCL operator+(const MatrixCL& other) const {
    fassert_equal(cl, other.cl);
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    MatrixCL res(nrow_, ncol_, cl);
    cl->Add(data_, other.data_, res.data_);
    return res;
  }
  MatrixCL operator-(const MatrixCL& other) const {
    fassert_equal(cl, other.cl);
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    MatrixCL res(nrow_, ncol_, cl);
    cl->Sub(data_, other.data_, res.data_);
    return res;
  }
  MatrixCL operator*(const MatrixCL& other) const {
    fassert_equal(cl, other.cl);
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    MatrixCL res(nrow_, ncol_, cl);
    cl->Mul(data_, other.data_, res.data_);
    return res;
  }
  MatrixCL operator/(const MatrixCL& other) const {
    fassert_equal(cl, other.cl);
    fassert_equal(nrow_, other.nrow_);
    fassert_equal(ncol_, other.ncol_);
    MatrixCL res(nrow_, ncol_, cl);
    cl->Div(data_, other.data_, res.data_);
    return res;
  }
  MatrixCL operator+(T a) const {
    MatrixCL res(nrow_, ncol_, cl);
    cl->Add(data_, a, res.data_);
    return res;
  }
  MatrixCL operator-(T a) const {
    MatrixCL res(nrow_, ncol_, cl);
    cl->Sub(data_, a, res.data_);
    return res;
  }
  MatrixCL operator*(T a) const {
    MatrixCL res(nrow_, ncol_, cl);
    cl->Mul(data_, a, res.data_);
    return res;
  }
  MatrixCL operator/(T a) const {
    MatrixCL res(nrow_, ncol_, cl);
    cl->Div(data_, a, res.data_);
    return res;
  }
  MatrixCL roll(int shift_row, int shift_col) const {
    MatrixCL res(nrow_, ncol_, cl);
    cl->Roll(data_, shift_col, shift_row, res.data_);
    return res;
  }
  MatrixCL roll2(int shift_row, int shift_col) const {
    using Idx = std::array<int, 2>;
    MatrixCL res(nrow_, ncol_, cl);
    if (nrow_ == 0 || ncol_ == 0) {
      return res;
    }
    const Idx shape = {int(nrow_), int(ncol_)};
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
      cl->AssignSubarray(res.data_, data_, {idst[1], idst[0]},
                         {isrc[1], isrc[0]}, {icnt[1], icnt[0]});
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
      icnt[k] = shape[k] - shift[k];
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
  MatrixCL conv(const T& a, const T& axm, const T& axp, const T& aym,
                const T& ayp) const {
    MatrixCL res(nrow_, ncol_, cl);
    if (nrow_ == 0 || ncol_ == 0) {
      return res;
    }
    // Swapping the order of x and y.
    // TODO: Revise to make x the slow index everywhere.
    cl->Conv(data_, a, aym, ayp, axm, axp, res.data_);
    return res;
  }

  // Reduction.
  T sum() const {
    return cl->Sum(data_);
  }
  T mean() const {
    return sum() / size();
  }
  T dot(const MatrixCL& other) const {
    fassert_equal(cl, other.cl);
    return cl->Dot(data_, other.data_);
  }
  T max() const {
    return cl->Max(data_);
  }
  T min() const {
    return cl->Min(data_);
  }

  // Friend functions.
  friend MatrixCL sin(const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl->Sin(matr.data_, res.data_);
    return res;
  }
  friend MatrixCL cos(const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl->Cos(matr.data_, res.data_);
    return res;
  }
  friend MatrixCL exp(const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl->Exp(matr.data_, res.data_);
    return res;
  }
  friend MatrixCL log(const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl->Log(matr.data_, res.data_);
    return res;
  }
  friend MatrixCL sqr(const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl->Sqr(matr.data_, res.data_);
    return res;
  }
  friend MatrixCL sqrt(const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl->Sqrt(matr.data_, res.data_);
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
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl->Add(matr.data_, a, res.data_);
    return res;
  }
  friend MatrixCL operator-(T a, const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl->Sub(a, matr.data_, res.data_);
    return res;
  }
  friend MatrixCL operator*(T a, const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl->Mul(matr.data_, a, res.data_);
    return res;
  }
  friend MatrixCL operator/(T a, const MatrixCL& matr) {
    MatrixCL res(matr.nrow_, matr.ncol_, matr.cl);
    matr.cl->Div(a, matr.data_, res.data_);
    return res;
  }
  friend MatrixCL roll(const MatrixCL& matr, int shift_row, int shift_col) {
    return matr.roll(shift_row, shift_col);
  }

  // Static functions.
  static MatrixCL zeros(size_t nrow, size_t ncol, CL& cl) {
    return MatrixCL(nrow, ncol, T(0), &cl);
  }
  template <class U>
  static MatrixCL zeros_like(const MatrixCL<U>& other) {
    return MatrixCL(other.nrow_, other.ncol_, T(0), other.cl);
  }
  template <class U>
  static MatrixCL ones_like(const MatrixCL<U>& other) {
    return MatrixCL(other.nrow_, other.ncol_, T(1), other.cl);
  }

 private:
  CL* cl = nullptr;
  size_t nrow_;
  size_t ncol_;
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
