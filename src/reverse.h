#pragma once

#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>

#include "matrix.h"

template <class T>
class Var {
 public:
  Var(const T& value, const std::string& name = "")
      : value_(std::make_unique<T>(value)), name_(name) {}

  const T& value() const {
    return *value_;
  }
  T& value() {
    return *value_;
  }
  const std::string& name() const {
    return name_;
  }
  Var& operator=(const T& value) {
    *value_ = value;
    return *this;
  }

 private:
  std::unique_ptr<T> value_;
  std::string name_;
};

struct BaseExtra {
  template <class Node>
  static void Print(std::ostream& out, const Node* node);
  template <class Node>
  void TraversePre(Node*) {}
  template <class Node>
  void TraversePost(Node*) {}
};

template <class T, class E>
class Node {
 public:
  Node(const std::string& name = "") : name_(name) {}
  virtual ~Node() = default;
  virtual const T& value() const {
    return value_;
  }
  virtual const T& grad() const {
    return grad_;
  }
  // TODO: Revise updates without age.
  int age() const {
    return age_;
  }
  void set_age(int age) {
    age_ = age;
  }
  void inc_age() {
    ++age_;
  }
  virtual void UpdateValue() = 0;
  virtual void ClearGrad() {
    grad_ = T();
  }
  virtual void AppendGrad(const T&) = 0;
  virtual std::string name() const {
    return name_;
  };
  using Extra = E;
  virtual void Print(std::ostream& out) const {
    Extra::Print(out, this);
  }
  virtual void TraversePre(Extra& extra) const {
    extra.TraversePre(this);
  }
  virtual void TraversePost(Extra& extra) const {
    extra.TraversePost(this);
  }

 protected:
  std::string name_;
  T value_;
  T grad_;
  int age_ = 0;
};

template <class T, class E>
class NodeVar : public Node<T, E> {
 public:
  NodeVar(const Var<T>& var, std::string name = "")
      : Node<T, E>(name), var_(var) {}
  const T& value() const override {
    return var_.value();
  }
  const Var<T>& var() const {
    return var_;
  }
  void UpdateValue() override {}
  void AppendGrad(const T& du) override {
    this->grad_ += du;
  }
  using Extra = E;
  void Print(std::ostream& out) const override {
    Extra::Print(out, this);
  }
  void TraversePre(Extra& extra) const override {
    extra.TraversePre(this);
  }
  void TraversePost(Extra& extra) const override {
    extra.TraversePost(this);
  }

  const Var<T>& var_;
};

template <class T, class X, class E>
class NodeUnary : public Node<T, E> {
 public:
  NodeUnary(std::shared_ptr<Node<X, E>> x, std::function<T(const X&)> fvalue,
            std::function<X(const X&, const T&)> fgrad, std::string name = "")
      : Node<T, E>(name), x_(x), fvalue_(fvalue), fgrad_(fgrad) {}

  const std::shared_ptr<Node<X, E>>& x() const {
    return x_;
  }
  void UpdateValue() override {
    if (x_->age() < this->age()) {
      x_->set_age(this->age());
      x_->UpdateValue();
    }
    this->value_ = fvalue_(x_->value());
  }
  void ClearGrad() override {
    this->grad_ = T();
    x_->ClearGrad();
  }
  void AppendGrad(const T& du) override {
    this->grad_ += du;
    x_->AppendGrad(fgrad_(x_->value(), du));
  }
  using Extra = E;
  void Print(std::ostream& out) const override {
    Extra::Print(out, this);
  }
  void TraversePre(Extra& extra) const override {
    extra.TraversePre(this);
    if (x_->age() < this->age()) {
      x_->set_age(this->age());
      x_->TraversePre(extra);
    }
  }
  void TraversePost(Extra& extra) const override {
    x_->TraversePost(extra);
    extra.TraversePost(this);
  }

 private:
  std::shared_ptr<Node<X, E>> x_;
  std::function<T(const X&)> fvalue_;
  std::function<X(const X&, const T&)> fgrad_;
};

template <class T, class X, class Y, class E>
class NodeBinary : public Node<T, E> {
 public:
  NodeBinary(std::shared_ptr<Node<X, E>> x, std::shared_ptr<Node<Y, E>> y,
             std::function<T(const X&, const Y&)> fvalue,
             std::function<X(const X&, const Y&, const T&)> fgradx,
             std::function<Y(const X&, const Y&, const T&)> fgrady,
             std::string name = "")
      : Node<T, E>(name),
        x_(x),
        y_(y),
        fvalue_(fvalue),
        fgradx_(fgradx),
        fgrady_(fgrady) {}

  const std::shared_ptr<Node<T, E>>& x() const {
    return x_;
  }
  const std::shared_ptr<Node<T, E>>& y() const {
    return y_;
  }

  void UpdateValue() override {
    if (x_->age() < this->age()) {
      x_->set_age(this->age());
      x_->UpdateValue();
    }
    if (y_->age() < this->age()) {
      y_->set_age(this->age());
      y_->UpdateValue();
    }
    this->value_ = fvalue_(x_->value(), y_->value());
  }
  void ClearGrad() override {
    this->grad_ = T();
    x_->ClearGrad();
    y_->ClearGrad();
  }
  void AppendGrad(const T& du) override {
    this->grad_ += du;
    x_->AppendGrad(fgradx_(x_->value(), y_->value(), du));
    y_->AppendGrad(fgrady_(x_->value(), y_->value(), du));
  }
  using Extra = E;
  void Print(std::ostream& out) const override {
    Extra::Print(out, this);
  }
  void TraversePre(Extra& extra) const override {
    extra.TraversePre(this);
    if (x_->age() < this->age()) {
      x_->set_age(this->age());
      x_->TraversePre(extra);
    }
    if (y_->age() < this->age()) {
      y_->set_age(this->age());
      y_->TraversePre(extra);
    }
  }
  void TraversePost(Extra& extra) const override {
    x_->TraversePost(extra);
    y_->TraversePost(extra);
    extra.TraversePost(this);
  }

 private:
  std::shared_ptr<Node<X, E>> x_;
  std::shared_ptr<Node<Y, E>> y_;
  std::function<T(const X&, const Y&)> fvalue_;
  std::function<X(const X&, const Y&, const T&)> fgradx_;
  std::function<Y(const X&, const Y&, const T&)> fgrady_;
};

template <class T, class E>
class Tracer {
 public:
  Tracer(std::shared_ptr<Node<T, E>> node) : node_(node) {}
  Tracer(const Var<T>& var)
      : node_(std::make_shared<NodeVar<T, E>>(var, var.name())) {}
  const T& value() const {
    return node_->value();
  }
  const T& grad() const {
    return node_->grad();
  }
  void UpdateValue() const {
    node_->inc_age();
    node_->UpdateValue();
  }
  void UpdateGrad() const {
    UpdateValue();
    node_->ClearGrad();
    node_->AppendGrad(T(1));
  }
  std::string name() const {
    return node_->name();
  };
  const std::shared_ptr<Node<T, E>>& node() const {
    return node_;
  }
  using Extra = E;
  void Print(std::ostream& out) const {
    node_->Print(out);
  }
  void TraversePre(Extra& extra) const {
    node_->inc_age();
    return node_->TraversePre(extra);
  }
  void TraversePost(Extra& extra) const {
    node_->inc_age();
    return node_->TraversePost(extra);
  }

 private:
  std::shared_ptr<Node<T, E>> node_;
};

template <class E = BaseExtra, class T>
Tracer<T, E> MakeTracer(const Var<T>& var) {
  return Tracer<T, E>(var);
}

////////////////////////////////////////
// Unary operations.
////////////////////////////////////////

template <class T, class E>
Tracer<T, E> ApplyScalarFunction(const Tracer<T, E>& tr_x,
                                 std::function<T(const T&)> fvalue,
                                 std::function<T(const T&)> fgrad,
                                 const std::string& name = "") {
  return {std::make_shared<NodeUnary<T, T, E>>(
      tr_x.node(), [fvalue](const T& x) { return fvalue(x); },
      [fgrad](const T& x, const T& du) { return fgrad(x) * du; }, name)};
}

template <class T, class E, class Scal>
Tracer<T, E> operator+(const Tracer<T, E>& tr_x, const Scal& y) {
  return {std::make_shared<NodeUnary<T, T, E>>(
      tr_x.node(), [y](const T& x) { return x + y; },
      [](const T&, const T& du) { return du; }, "+")};
}

template <class T, class E, class Scal>
Tracer<T, E> operator+(const Scal& x, const Tracer<T, E>& tr_y) {
  return {std::make_shared<NodeUnary<T, T, E>>(
      tr_y.node(), [x](const T& y) { return x + y; },
      [](const T&, const T& du) { return du; }, "+")};
}

template <class T, class E, class Scal>
Tracer<T, E> operator-(const Tracer<T, E>& tr_x, const Scal& y) {
  return {std::make_shared<NodeUnary<T, T, E>>(
      tr_x.node(), [y](const T& x) { return x - y; },
      [](const T&, const T& du) { return du; }, "-")};
}

template <class T, class E, class Scal>
Tracer<T, E> operator-(const Scal& x, const Tracer<T, E>& tr_y) {
  return {std::make_shared<NodeUnary<T, T, E>>(
      tr_y.node(), [x](const T& y) { return x - y; },
      [](const T&, const T& du) { return -du; }, "-")};
}

template <class T, class E>
Tracer<T, E> operator-(const Tracer<T, E>& tr_x) {
  return {std::make_shared<NodeUnary<T, T, E>>(
      tr_x.node(), [](const T& x) { return -x; },
      [](const T&, const T& du) { return -du; }, "-")};
}

template <class T, class E, class Scal>
Tracer<T, E> operator*(const Tracer<T, E>& tr_x, const Scal& y) {
  return {std::make_shared<NodeUnary<T, T, E>>(
      tr_x.node(), [y](const T& x) { return x * y; },
      [y](const T&, const T& du) { return du * y; }, "*")};
}

template <class T, class E, class Scal>
Tracer<T, E> operator*(const Scal& x, const Tracer<T, E>& tr_y) {
  return {std::make_shared<NodeUnary<T, T, E>>(
      tr_y.node(), [x](const T& y) { return x * y; },
      [x](const T&, const T& du) { return x * du; }, "*")};
}

template <class T, class E, class Scal>
Tracer<T, E> operator/(const Tracer<T, E>& tr_x, const Scal& y) {
  return {std::make_shared<NodeUnary<T, T, E>>(
      tr_x.node(), [y](const T& x) { return x / y; },
      [y](const T&, const T& du) { return du / y; }, "/")};
}

template <class T, class E>
Tracer<T, E> transpose(const Tracer<T, E>& tr_x) {
  return {std::make_shared<NodeUnary<T, T, E>>(
      tr_x.node(), [](const T& x) { return x.transpose(); },
      [](const T&, const T& du) { return du.transpose(); }, "T")};
}

template <class T, class E>
Tracer<T, E> interpolate(const Tracer<T, E>& tr_x) {
  return {std::make_shared<NodeUnary<T, T, E>>(
      tr_x.node(), [](const T& x) { return x.interpolate(); },
      [](const T&, const T& du) { return du.interpolate_adjoint(); }, "I")};
}

template <class T, class E>
Tracer<T, E> restrict(const Tracer<T, E>& tr_x) {
  return {std::make_shared<NodeUnary<T, T, E>>(
      tr_x.node(), [](const T& x) { return x.restrict(); },
      [](const T&, const T& du) { return du.restrict_adjoint(); }, "R")};
}

template <class T, class E>
Tracer<T, E> roll(const Tracer<T, E>& tr_x, int shift_row, int shift_col) {
  const auto name = "roll(" + std::to_string(shift_row) + "," +
                    std::to_string(shift_col) + ")";
  return {std::make_shared<NodeUnary<T, T, E>>(
      tr_x.node(),
      [shift_row, shift_col](const T& x) {
        return x.roll(shift_row, shift_col);
      },
      [shift_row, shift_col](const T&, const T& du) {
        return du.roll(-shift_row, -shift_col);
      },
      name)};
}

template <class T, class E>
Tracer<T, E> sum(const Tracer<Matrix<T>, E>& tr_x) {
  return {std::make_shared<NodeUnary<T, Matrix<T>, E>>(
      tr_x.node(), [](const Matrix<T>& x) { return x.sum(); },
      [](const Matrix<T>& x, const T& du) {
        return du * Matrix<T>::ones_like(x);
      },
      "sum")};
}

template <class T, class E>
Tracer<T, E> sum(const Tracer<T, E>& tr_x) {
  return {std::make_shared<NodeUnary<T, T, E>>(
      tr_x.node(), [](const T& x) { return x; },
      [](const T&, const T& du) { return du * T(1); }, "sum")};
}

template <class T, class E>
Tracer<T, E> mean(const Tracer<Matrix<T>, E>& tr_x) {
  return {std::make_shared<NodeUnary<T, Matrix<T>, E>>(
      tr_x.node(), [](const Matrix<T>& x) { return x.mean(); },
      [](const Matrix<T>& x, const T& du) {
        return du * Matrix<T>::ones_like(x) / x.size();
      },
      "mean")};
}

template <class T, class E>
Tracer<T, E> mean(const Tracer<T, E>& tr_x) {
  return {std::make_shared<NodeUnary<T, T, E>>(
      tr_x.node(), [](const T& x) { return x; },
      [](const T&, const T& du) { return du * T(1); }, "mean")};
}

template <class T, class E>
Tracer<T, E> sin(const Tracer<T, E>& tr_x) {
  using std::cos;
  using std::sin;
  return ApplyScalarFunction<T, E>(
      tr_x, [](const T& x) { return sin(x); },
      [](const T& x) { return cos(x); }, "sin");
}

template <class T, class E>
Tracer<T, E> cos(const Tracer<T, E>& tr_x) {
  using std::cos;
  using std::sin;
  return ApplyScalarFunction<T, E>(
      tr_x, [](const T& x) { return cos(x); },
      [](const T& x) { return -sin(x); }, "cos");
}

template <class T, class E>
Tracer<T, E> exp(const Tracer<T, E>& tr_x) {
  using std::exp;
  return ApplyScalarFunction<T, E>(
      tr_x, [](const T& x) { return exp(x); },
      [](const T& x) { return exp(x); }, "exp");
}

template <class T, class E>
Tracer<T, E> log(const Tracer<T, E>& tr_x) {
  using std::log;
  return ApplyScalarFunction<T, E>(
      tr_x, [](const T& x) { return log(x); },  //
      [](const T& x) { return 1 / x; }, "log");
}

template <class T, class E>
Tracer<T, E> sqr(const Tracer<T, E>& tr_x) {
  return ApplyScalarFunction<T, E>(
      tr_x, [](const T& x) { return sqr(x); },  //
      [](const T& x) { return 2 * x; }, "sqr");
}

////////////////////////////////////////
// Binary operations.
////////////////////////////////////////

template <class T, class E>
Tracer<T, E> operator+(Tracer<T, E> tr_x, Tracer<T, E> tr_y) {
  return {std::make_shared<NodeBinary<T, T, T, E>>(
      tr_x.node(), tr_y.node(),  //
      [](const T& x, const T& y) { return x + y; },
      [](const T&, const T&, const T& du) { return du; },
      [](const T&, const T&, const T& du) { return du; }, "+")};
}

template <class T, class E>
Tracer<T, E> operator-(Tracer<T, E> tr_x, Tracer<T, E> tr_y) {
  return {std::make_shared<NodeBinary<T, T, T, E>>(
      tr_x.node(), tr_y.node(),  //
      [](const T& x, const T& y) { return x - y; },
      [](const T&, const T&, const T& du) { return du; },
      [](const T&, const T&, const T& du) { return -du; }, "-")};
}

template <class T, class E>
Tracer<T, E> operator*(Tracer<T, E> tr_x, Tracer<T, E> tr_y) {
  return {std::make_shared<NodeBinary<T, T, T, E>>(
      tr_x.node(), tr_y.node(),  //
      [](const T& x, const T& y) { return x * y; },
      [](const T&, const T& y, const T& du) { return y * du; },
      [](const T& x, const T&, const T& du) { return x * du; }, "*")};
}

template <class T, class E>
Tracer<T, E> operator/(Tracer<T, E> tr_x, Tracer<T, E> tr_y) {
  return {std::make_shared<NodeBinary<T, T, T, E>>(
      tr_x.node(), tr_y.node(),  //
      [](const T& x, const T& y) { return x / y; },
      [](const T&, const T& y, const T& du) { return du / y; },
      [](const T& x, const T& y, const T& du) { return -x * du / (y * y); },
      "/")};
}

////////////////////////////////////////
// Output.
////////////////////////////////////////

template <class T>
struct TypeName {
  inline static const std::string value = "unknown";
};
template <>
struct TypeName<double> {
  inline static const std::string value = "double";
};
template <>
struct TypeName<float> {
  inline static const std::string value = "float";
};
template <>
struct TypeName<int> {
  inline static const std::string value = "int";
};
template <class T>
struct TypeName<Matrix<T>> {
  inline static const std::string value = "Matrix<" + TypeName<T>::value + ">";
};
template <class T>
std::string GetTypeName() {
  return TypeName<T>::value;
}

template <class T>
std::ostream& operator<<(std::ostream& out, const Var<T>& var) {
  out << "Var(" << var.value();
  out << ", " << GetTypeName<T>();
  if (!var.name().empty()) {
    out << ", \"" << var.name() << "\"";
  }
  out << ")";
  return out;
}

template <class T, class E>
std::ostream& operator<<(std::ostream& out, const Node<T, E>& node) {
  node.Print(out);
  return out;
}

template <class T, class E>
std::ostream& operator<<(std::ostream& out, const Tracer<T, E>& tracer) {
  out << "Tracer(";
  tracer.node()->Print(out);
  out << ")";
  return out;
}

// Writes nodes in plain text.
struct PrintImpl {
  template <class T, class E>
  static void Print(std::ostream& out, const Node<T, E>* node) {
    out << "Node(" << node->name() << ", " << GetTypeName<T>() << ")";
  }
  template <class T, class E>
  static void Print(std::ostream& out, const NodeVar<T, E>* node) {
    out << "NodeVar(" << node->var() << ")";
  }
  template <class T, class X, class E>
  static void Print(std::ostream& out, const NodeUnary<T, X, E>* node) {
    out << "NodeUnary(" << node->name() << "[" << node->x()->name() << "]"
        << ", " << GetTypeName<T>() << "[" << GetTypeName<T>() << "]"
        << ")";
  }
  template <class T, class X, class Y, class E>
  static void Print(std::ostream& out, const NodeBinary<T, X, Y, E>* node) {
    out << "NodeBinary(" << node->name() << "[" << node->x()->name() << ","
        << node->y()->name() << "]"
        << ", " << GetTypeName<T>() << "[" << GetTypeName<T>() << ","
        << GetTypeName<T>() << "]"
        << ")";
  }
};

template <class Node>
void BaseExtra::Print(std::ostream& out, const Node* node) {
  PrintImpl::Print(out, node);
}

// Writes the graph in DOT format.
struct DotWriter {
  DotWriter(std::ostream& out_) : out(out_), pad("  ") {
    out << "digraph {\n";
    out << "  rankdir=\"BT\"\n";
    out << "  node [shape=circle, margin=0]\n";
  }
  ~DotWriter() {
    out << "}\n";
  }
  template <class T>
  std::string nodename(T* node) {
    if (!names.count(node)) {
      const size_t i = names.size();
      names[node] = "n" + std::to_string(i);
      out << pad << names[node] << " [label=\"" << node->name() << "\"]\n";
    }
    return names.at(node);
  }
  template <class T, class E>
  void Write(const Node<T, E>*) {}
  template <class T, class E>
  void Write(const NodeVar<T, E>*) {}
  template <class T, class X, class E>
  void Write(const NodeUnary<T, X, E>* node) {
    const auto name = nodename(node);
    const auto xname = nodename(node->x().get());
    out << pad << xname << " -> " << name << '\n';
  }
  template <class T, class X, class Y, class E>
  void Write(const NodeBinary<T, X, Y, E>* node) {
    const auto name = nodename(node);
    const auto xname = nodename(node->x().get());
    const auto yname = nodename(node->y().get());
    out << pad << xname << " -> " << name << '\n';
    out << pad << yname << " -> " << name << '\n';
  }
  std::ostream& out;
  std::map<const void*, std::string> names;
  std::string pad;
};
