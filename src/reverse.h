#pragma once

#include <array>
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>

#include "matrix.h"

template <class T>
class Var {
 public:
  Var(std::unique_ptr<T>&& ptr, const std::string& name = "")
      : value_(std::move(ptr)), name_(name) {}
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
  void Visit(Node*) {}
};

template <class T>
struct ClearImpl {
  static void Clear(T& u) {
    u = 0;
  }
};

template <class T>
struct ClearImpl<Matrix<T>> {
  static void Clear(Matrix<T>& u) {
    u.clear();
  }
};

template <class T>
void Clear(T& u) {
  ClearImpl<T>::Clear(u);
};

template <class E>
class GenericNode;

template <class E>
using NodeOrder = std::vector<GenericNode<E>*>;

template <class E>
class GenericNode {
 public:
  using Extra = E;

  virtual ~GenericNode() = default;
  bool visited() const {
    return visited_;
  }
  void set_visited(bool visited) {
    visited_ = visited;
  }

  virtual void UpdateValue() = 0;
  virtual void UpdateGrad() = 0;
  virtual void ClearGrad() = 0;
  virtual void BuildForwardOrder(NodeOrder<E>& order) = 0;
  virtual void Print(std::ostream& out) const {
    Extra::Print(out, this);
  }
  virtual void Visit(Extra& extra) const = 0;

 protected:
  bool visited_ = false;
};

template <class T, class E>
class Node : public GenericNode<E> {
 public:
  using Extra = E;

  Node(const std::string& name = "") : name_(name) {}
  virtual std::string name() const {
    return name_;
  };
  virtual const T& value() const {
    return value_;
  }
  virtual const T& grad() const {
    return grad_;
  }

  void UpdateValue() override = 0;
  void UpdateGrad() override = 0;
  void ClearGrad() override {
    Clear(grad_);
  }
  virtual void AddGrad(const T& add) {
    grad_ += add;
  }

  void BuildForwardOrder(NodeOrder<E>& order) override = 0;
  void Print(std::ostream& out) const override {
    Extra::Print(out, this);
  }
  void Visit(Extra& extra) const override {
    extra.Visit(this);
  }

 protected:
  std::string name_;
  T value_;
  T grad_;
};

template <class T, class E>
class NodeVar : public Node<T, E> {
 public:
  using Extra = E;

  NodeVar(const Var<T>& var, std::string name = "")
      : Node<T, E>(name), var_(var) {}
  const T& value() const override {
    return var_.value();
  }
  const Var<T>& var() const {
    return var_;
  }

  void UpdateValue() override {}
  void UpdateGrad() override {}

  void BuildForwardOrder(NodeOrder<E>& order) override {
    this->set_visited(true);
    order.push_back(this);
  }
  void Print(std::ostream& out) const override {
    Extra::Print(out, this);
  }
  void Visit(Extra& extra) const override {
    extra.Visit(this);
  }
  const Var<T>& var_;
};

template <class T, class X, class E>
class NodeUnary : public Node<T, E> {
 public:
  using Extra = E;

  NodeUnary(std::shared_ptr<Node<X, E>> x, std::function<T(const X&)> fvalue,
            std::function<X(const X&, const T&)> fgrad, std::string name = "")
      : Node<T, E>(name), x_(x), fvalue_(fvalue), fgrad_(fgrad) {}

  const std::shared_ptr<Node<X, E>>& x() const {
    return x_;
  }
  void UpdateValue() override {
    this->value_ = fvalue_(x_->value());
  }
  void UpdateGrad() override {
    x_->AddGrad(fgrad_(x_->value(), this->grad_));
  }
  void BuildForwardOrder(NodeOrder<E>& order) override {
    this->set_visited(true);
    if (!x_->visited()) {
      x_->BuildForwardOrder(order);
    }
    order.push_back(this);
  }
  void Print(std::ostream& out) const override {
    Extra::Print(out, this);
  }
  void Visit(Extra& extra) const override {
    extra.Visit(this);
  }

 private:
  std::shared_ptr<Node<X, E>> x_;
  std::function<T(const X&)> fvalue_;
  std::function<X(const X&, const T&)> fgrad_;
};

template <class T, class X, class Y, class E>
class NodeBinary : public Node<T, E> {
 public:
  using Extra = E;

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
    this->value_ = fvalue_(x_->value(), y_->value());
  }
  void UpdateGrad() override {
    x_->AddGrad(fgradx_(x_->value(), y_->value(), this->grad_));
    y_->AddGrad(fgrady_(x_->value(), y_->value(), this->grad_));
  }
  void BuildForwardOrder(NodeOrder<E>& order) override {
    this->set_visited(true);
    if (!x_->visited()) {
      x_->BuildForwardOrder(order);
    }
    if (!y_->visited()) {
      y_->BuildForwardOrder(order);
    }
    order.push_back(this);
  }
  void Print(std::ostream& out) const override {
    Extra::Print(out, this);
  }
  void Visit(Extra& extra) const override {
    extra.Visit(this);
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
  using Extra = E;

  Tracer() = default;
  Tracer(std::shared_ptr<Node<T, E>> node) : node_(node) {}
  Tracer(const Var<T>& var)
      : node_(std::make_shared<NodeVar<T, E>>(var, var.name())) {}
  const T& value() const {
    return node_->value();
  }
  const T& grad() const {
    return node_->grad();
  }
  NodeOrder<E> GetFowardOrder() const {
    fassert(!node_->visited());
    NodeOrder<E> order;
    node_->BuildForwardOrder(order);
    return order;
  }
  std::string name() const {
    return node_->name();
  };
  const std::shared_ptr<Node<T, E>>& node() const {
    return node_;
  }
  void UpdateValue(const NodeOrder<Extra>& forward_order) {
    // Check that called from the root node (e.g. scalar loss).
    fassert_equal(node_.get(), forward_order.back());
    for (auto* node : forward_order) {
      node->UpdateValue();
    }
  }
  void UpdateGrad(const NodeOrder<Extra>& forward_order,
                  bool update_value = true) {
    if (forward_order.empty()) {
      return;
    }
    // Check that called from the root node (e.g. scalar loss).
    fassert_equal(node_.get(), forward_order.back());
    const NodeOrder<Extra> reverse_order(forward_order.rbegin(),
                                         forward_order.rend());
    if (update_value) {
      UpdateValue(forward_order);
    }
    for (auto* node : reverse_order) {
      node->ClearGrad();
    }
    node_->AddGrad(T(1));
    for (auto* node : reverse_order) {
      node->UpdateGrad();
    }
  }
  void UpdateGrad() {
    const auto order = GetFowardOrder();
    UpdateGrad(order);
    ClearVisited(order);
  }
  void Print(std::ostream& out) const {
    node_->Print(out);
  }
  void Visit(Extra& extra) const {
    return node_->Visit(extra);
  }

 private:
  std::shared_ptr<Node<T, E>> node_;
};

template <class E = BaseExtra, class T>
Tracer<T, E> MakeTracer(const Var<T>& var) {
  return Tracer<T, E>(var);
}

template <class Extra>
void Traverse(const NodeOrder<Extra>& order, Extra& extra) {
  for (auto* node : order) {
    node->Visit(extra);
  }
}

template <class Extra>
void ClearVisited(const NodeOrder<Extra>& order) {
  for (auto* node : order) {
    node->set_visited(false);
  }
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

template <class T, class A, class E>
Tracer<Matrix<T>, E> conv(const Tracer<Matrix<T>, E>& tr_x, const A& a,
                          const A& axm, const A& axp, const A& aym,
                          const A& ayp) {
  return {std::make_shared<NodeUnary<Matrix<T>, Matrix<T>, E>>(
      tr_x.node(),
      [a, axm, axp, aym, ayp](const Matrix<T>& x) {
        return x.conv(a, axm, axp, aym, ayp);
      },
      [a, axm, axp, aym, ayp](const Matrix<T>&, const Matrix<T>& du) {
        return du.conv(a, axp, axm, ayp, aym);
      },
      "conv5")};
}

template <class T, class A, class E>
Tracer<Matrix<T>, E> conv(const Tracer<Matrix<T>, E>& tr_x,
                          const std::array<A, 9>& a) {
  return {std::make_shared<NodeUnary<Matrix<T>, Matrix<T>, E>>(
      tr_x.node(), [a](const Matrix<T>& x) { return x.conv(a); },
      [a](const Matrix<T>&, const Matrix<T>& du) { return du.conv_adjoint(a); },
      "conv9")};
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
      [](const T&, const T& du) { return du; }, "sum")};
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
      [](const T&, const T& du) { return du; }, "mean")};
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
  out << "Var(";
  const auto stype = GetTypeName<T>();
  // TODO: Revise with constexpr.
  if (stype == "float" || stype == "double" || stype == "int") {
    out << var.value() << ", ";
  }
  out << GetTypeName<T>();
  if (!var.name().empty()) {
    out << ", \"" << var.name() << "\"";
  }
  out << ")";
  return out;
}

template <class E>
std::ostream& operator<<(std::ostream& out, const GenericNode<E>& node) {
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
  template <class E>
  static void Print(std::ostream& out, const GenericNode<E>*) {
    out << "GenericNode()";
  }
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
