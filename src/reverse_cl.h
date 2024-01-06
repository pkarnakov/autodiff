#pragma once

#include "matrix_cl.h"
#include "reverse.h"

////////////////////////////////////////
// Specializations for reverse.h
////////////////////////////////////////
template <class T>
struct TypeName<MatrixCL<T>> {
  inline static const std::string value =
      "MatrixCL<" + TypeName<T>::value + ">";
};
template <class T>
void Clear(MatrixCL<T>& matr) {
  matr.clear();
};
template <class T, class E>
Tracer<T, E> sum(const Tracer<MatrixCL<T>, E>& tr_x) {
  return {std::make_shared<NodeUnary<T, MatrixCL<T>, E>>(
      tr_x.node(), [](const MatrixCL<T>& x) { return x.sum(); },
      [](const MatrixCL<T>& x, const T& du) {
        return du * MatrixCL<T>::ones_like(x);
      },
      "sum")};
}
template <class T, class E>
Tracer<T, E> mean(const Tracer<MatrixCL<T>, E>& tr_x) {
  return {std::make_shared<NodeUnary<T, MatrixCL<T>, E>>(
      tr_x.node(), [](const MatrixCL<T>& x) { return x.mean(); },
      [](const MatrixCL<T>& x, const T& du) {
        return du * MatrixCL<T>::ones_like(x) / x.size();
      },
      "mean")};
}
