#pragma once

#include <functional>
#include <iosfwd>
#include <memory>
#include <string>

template <class T>
std::string GetTypeName() {
  return "unknown";
}

template <>
std::string GetTypeName<double>() {
  return "double";
}
template <>
std::string GetTypeName<float>() {
  return "float";
}
template <>
std::string GetTypeName<int>() {
  return "int";
}

template <class T>
class Var {
 public:
  explicit Var(T value) : value_(std::make_unique<T>(value)) {}
  explicit Var(const std::string& name) : name_(name) {}
  Var(T value, const std::string& name)
      : value_(std::make_unique<T>(value)), name_(name) {}

  const T& value() const {
    return *value_;
  }
  Var& operator=(const T& value) {
    *value_ = value;
    return *this;
  }

  // Friend functions.
  friend std::ostream& operator<<(std::ostream& out, const Var& var) {
    out << "Var(";
    if (var.value_) {
      out << *var.value_;
    } else {
      out << "null";
    }
    out << ", " << GetTypeName<T>();
    if (!var.name_.empty()) {
      out << ", " << var.name_;
    }
    out << ")";
    return out;
  }

 private:
  std::unique_ptr<T> value_;
  std::string name_;
};

template <class T>
class Node {
 public:
  virtual ~Node() = default;
  virtual T value() const = 0;
  virtual T grad() const = 0;
};

template <class T>
class NodeVar : public Node<T> {
 public:
  NodeVar(const Var<T>& var) : var_(var) {}

  T value() const override {
    return var_.value();
  }
  T grad() const override {
    return T(1);
  }
  const Var<T>& var() const {
    return var_;
  }

  const Var<T>& var_;
};

template <class T>
struct Tracer {
  Tracer(std::shared_ptr<Node<T>> node_) : node(node_) {}
  Tracer(const Var<T>& var) : node(std::make_shared<NodeVar<T>>(var)) {}
  T value() const {
    return node->value();
  }
  T grad() const {
    return node->grad();
  }

  std::shared_ptr<Node<T>> node;
};

template <class T>
class NodeUnary : public Node<T> {
 public:
  NodeUnary(std::shared_ptr<Node<T>> x, std::function<T(T)> fvalue,
            std::function<T(T, T)> fgrad)
      : x_(x), fvalue_(fvalue), fgrad_(fgrad) {}

  T value() const override {
    return fvalue_(x_->value());
  }
  T grad() const override {
    return fgrad_(x_->value(), x_->grad());
  }

  std::shared_ptr<Node<T>> x_;
  std::function<T(T)> fvalue_;
  std::function<T(T, T)> fgrad_;
};

template <class T>
class NodeBinary : public Node<T> {
 public:
  NodeBinary(std::shared_ptr<Node<T>> x, std::shared_ptr<Node<T>> y,
             std::function<T(T, T)> fvalue, std::function<T(T, T, T, T)> fgrad)
      : x_(x), y_(y), fvalue_(fvalue), fgrad_(fgrad) {}

  T value() const override {
    return fvalue_(x_->value(), y_->value());
  }
  T grad() const override {
    return fgrad_(x_->value(), y_->value(), x_->grad(), y_->grad());
  }

  std::shared_ptr<Node<T>> x_;
  std::shared_ptr<Node<T>> y_;
  std::function<T(T, T)> fvalue_;
  std::function<T(T, T, T, T)> fgrad_;
};

////////////////////////////////////////
// Unary operations.
////////////////////////////////////////

template <class T>
Tracer<T> operator+(Tracer<T> tr_x, T y) {
  return {std::make_shared<NodeUnary<T>>(
      tr_x.node, [y](T x) { return x + y; },  //
      [](T, T dx) { return dx; })};
}

template <class T>
Tracer<T> operator+(T x, Tracer<T> tr_y) {
  return {std::make_shared<NodeUnary<T>>(
      tr_y.node, [x](T y) { return x + y; },  //
      [](T, T dy) { return dy; })};
}

template <class T>
Tracer<T> operator-(Tracer<T> tr_x, T y) {
  return {std::make_shared<NodeUnary<T>>(
      tr_x.node, [y](T x) { return x - y; },  //
      [](T, T dx) { return dx; })};
}

template <class T>
Tracer<T> operator-(T x, Tracer<T> tr_y) {
  return {std::make_shared<NodeUnary<T>>(
      tr_y.node, [x](T y) { return x - y; },  //
      [](T, T dy) { return -dy; })};
}

template <class T>
Tracer<T> operator-(Tracer<T> tr_x) {
  return {std::make_shared<NodeUnary<T>>(
      tr_x.node, [](T x) { return -x; },  //
      [](T, T dx) { return -dx; })};
}

template <class T>
Tracer<T> operator*(Tracer<T> tr_x, T y) {
  return {std::make_shared<NodeUnary<T>>(
      tr_x.node, [y](T x) { return x * y; },  //
      [y](T, T dx) { return dx * y; })};
}

template <class T>
Tracer<T> operator*(T x, Tracer<T> tr_y) {
  return {std::make_shared<NodeUnary<T>>(
      tr_y.node, [x](T y) { return x * y; },  //
      [x](T, T dy) { return x * dy; })};
}

template <class T>
Tracer<T> sqr(Tracer<T> tr_x) {
  return {std::make_shared<NodeUnary<T>>(
      tr_x.node, [](T x) { return x * x; },
      [](T x, T dx) { return 2 * x * dx; })};
}

template <class T>
Tracer<T> sin(Tracer<T> tr_x) {
  using std::cos;
  using std::sin;
  return {std::make_shared<NodeUnary<T>>(
      tr_x.node, [](T x) { return sin(x); },
      [](T x, T dx) { return cos(x) * dx; })};
}

template <class T>
Tracer<T> cos(Tracer<T> tr_x) {
  using std::cos;
  using std::sin;
  return {std::make_shared<NodeUnary<T>>(
      tr_x.node, [](T x) { return cos(x); },
      [](T x, T dx) { return -sin(x) * dx; })};
}

template <class T>
Tracer<T> exp(Tracer<T> tr_x) {
  using std::exp;
  return {std::make_shared<NodeUnary<T>>(
      tr_x.node, [](T x) { return exp(x); },
      [](T x, T dx) { return exp(x) * dx; })};
}

template <class T>
Tracer<T> log(Tracer<T> tr_x) {
  using std::log;
  return {std::make_shared<NodeUnary<T>>(
      tr_x.node, [](T x) { return log(x); },  //
      [](T x, T dx) { return dx / x; })};
}

////////////////////////////////////////
// Binary operations.
////////////////////////////////////////

template <class T>
Tracer<T> operator+(Tracer<T> tr_x, Tracer<T> tr_y) {
  return {std::make_shared<NodeBinary<T>>(
      tr_x.node, tr_y.node, [](T x, T y) { return x + y; },
      [](T, T, T dx, T dy) { return dx + dy; })};
}

template <class T>
Tracer<T> operator-(Tracer<T> tr_x, Tracer<T> tr_y) {
  return {std::make_shared<NodeBinary<T>>(
      tr_x.node, tr_y.node, [](T x, T y) { return x - y; },
      [](T, T, T dx, T dy) { return dx - dy; })};
}

template <class T>
Tracer<T> operator*(Tracer<T> tr_x, Tracer<T> tr_y) {
  return {std::make_shared<NodeBinary<T>>(
      tr_x.node, tr_y.node, [](T x, T y) { return x * y; },
      [](T x, T y, T dx, T dy) { return x * dy + dx * y; })};
}

template <class T>
Tracer<T> operator/(Tracer<T> tr_x, Tracer<T> tr_y) {
  return {std::make_shared<NodeBinary<T>>(
      tr_x.node, tr_y.node, [](T x, T y) { return x / y; },
      [](T x, T y, T dx, T dy) { return dx / y - (x * dy) / (y * y); })};
}
