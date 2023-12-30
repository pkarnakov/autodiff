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

template <class T>
class Node;
template <class T>
class NodeVar;
template <class T, class X>
class NodeUnary;
template <class T, class X, class Y>
class NodeBinary;

// TODO: Revise without forward declaration of output functions.
//       Consider making the tree traversal polymorphic.
template <class T>
void Print(std::ostream&, const Node<T>*);
template <class T>
void Print(std::ostream&, const NodeVar<T>*);
template <class T, class X>
void Print(std::ostream&, const NodeUnary<T, X>*);
template <class T, class X, class Y>
void Print(std::ostream&, const NodeBinary<T, X, Y>*);

template <class T>
class Node {
 public:
  Node(const std::string& name = "") : name_(name), grad_(T()) {}
  virtual ~Node() = default;
  virtual T value() const = 0;
  virtual T grad() const {
    return grad_;
  }
  virtual void ClearGrad() = 0;
  virtual void UpdateGrad(const T& du) = 0;
  virtual std::string name() const {
    return name_;
  };
  virtual void Print(std::ostream& out) const {
    ::Print(out, this);
  }

 protected:
  std::string name_;
  T grad_;
};

template <class T>
class NodeVar : public Node<T> {
 public:
  NodeVar(const Var<T>& var, std::string name = "")
      : Node<T>(name), var_(var) {}

  T value() const override {
    return var_.value();
  }
  const Var<T>& var() const {
    return var_;
  }
  void ClearGrad() override {
    this->grad_ = T();
  }
  void UpdateGrad(const T& du) override {
    this->grad_ = du;
  }
  void Print(std::ostream& out) const override {
    ::Print(out, this);
  }

  const Var<T>& var_;
};

template <class T, class X>
class NodeUnary : public Node<T> {
 public:
  NodeUnary(std::shared_ptr<Node<X>> x, std::function<T(const X&)> fvalue,
            std::function<X(const X&, const T&)> fgrad, std::string name = "")
      : Node<T>(name), x_(x), fvalue_(fvalue), fgrad_(fgrad) {}

  T value() const override {
    return fvalue_(x_->value());
  }
  void ClearGrad() override {
    this->grad_ = T();
    x_->ClearGrad();
  }
  void UpdateGrad(const T& du) override {
    this->grad_ += du;
    x_->UpdateGrad(fgrad_(x_->value(), du));
  }
  const std::shared_ptr<Node<X>>& x() const {
    return x_;
  }
  void Print(std::ostream& out) const override {
    ::Print(out, this);
  }

  std::shared_ptr<Node<X>> x_;
  std::function<T(const X&)> fvalue_;
  std::function<X(const X&, const T&)> fgrad_;
};

template <class T, class X, class Y>
class NodeBinary : public Node<T> {
 public:
  NodeBinary(std::shared_ptr<Node<X>> x, std::shared_ptr<Node<Y>> y,
             std::function<T(const X&, const Y&)> fvalue,
             std::function<X(const X&, const Y&, const T&)> fgradx,
             std::function<Y(const X&, const Y&, const T&)> fgrady,
             std::string name = "")
      : Node<T>(name),
        x_(x),
        y_(y),
        fvalue_(fvalue),
        fgradx_(fgradx),
        fgrady_(fgrady) {}

  T value() const override {
    return fvalue_(x_->value(), y_->value());
  }
  void ClearGrad() override {
    this->grad_ = T();
    x_->ClearGrad();
    y_->ClearGrad();
  }
  void UpdateGrad(const T& du) override {
    this->grad_ += du;
    x_->UpdateGrad(fgradx_(x_->value(), y_->value(), du));
    y_->UpdateGrad(fgrady_(x_->value(), y_->value(), du));
  }
  const std::shared_ptr<Node<T>>& x() const {
    return x_;
  }
  const std::shared_ptr<Node<T>>& y() const {
    return y_;
  }
  void Print(std::ostream& out) const override {
    ::Print(out, this);
  }

  std::shared_ptr<Node<X>> x_;
  std::shared_ptr<Node<Y>> y_;
  std::function<T(const X&, const Y&)> fvalue_;
  std::function<X(const X&, const Y&, const T&)> fgradx_;
  std::function<Y(const X&, const Y&, const T&)> fgrady_;
};

template <class T>
struct Tracer {
  Tracer(std::shared_ptr<Node<T>> node) : node_(node) {}
  Tracer(const Var<T>& var)
      : node_(std::make_shared<NodeVar<T>>(var, var.name())) {}
  T value() const {
    return node_->value();
  }
  T grad() const {
    return node_->grad();
  }
  void ClearGrad() const {
    node_->ClearGrad();
  }
  void UpdateGrad(const T& du) const {
    node_->UpdateGrad(du);
  }
  std::string name() const {
    return node_->name();
  };
  const std::shared_ptr<Node<T>>& node() const {
    return node_;
  }

 private:
  std::shared_ptr<Node<T>> node_;
};

////////////////////////////////////////
// Unary operations.
////////////////////////////////////////

template <class T>
Tracer<T> ApplyScalarFunction(const Tracer<T>& tr_x,
                              std::function<T(const T&)> fvalue,
                              std::function<T(const T&)> fgrad,
                              const std::string& name = "") {
  return {std::make_shared<NodeUnary<T, T>>(
      tr_x.node(), [fvalue](const T& x) { return fvalue(x); },
      [fgrad](const T& x, const T& du) { return fgrad(x) * du; }, name)};
}

template <class T, class Scal>
Tracer<T> operator+(const Tracer<T>& tr_x, const Scal& y) {
  return {std::make_shared<NodeUnary<T, T>>(
      tr_x.node(), [y](const T& x) { return x + y; },
      [](const T&, const T& du) { return du; }, "+")};
}

template <class T, class Scal>
Tracer<T> operator+(const Scal& x, const Tracer<T>& tr_y) {
  return {std::make_shared<NodeUnary<T, T>>(
      tr_y.node(), [x](const T& y) { return x + y; },
      [](const T&, const T& du) { return du; }, "+")};
}

template <class T, class Scal>
Tracer<T> operator-(const Tracer<T>& tr_x, const Scal& y) {
  return {std::make_shared<NodeUnary<T, T>>(
      tr_x.node(), [y](const T& x) { return x - y; },
      [](const T&, const T& du) { return du; }, "-")};
}

template <class T, class Scal>
Tracer<T> operator-(const Scal& x, const Tracer<T>& tr_y) {
  return {std::make_shared<NodeUnary<T, T>>(
      tr_y.node(), [x](const T& y) { return x - y; },
      [](const T&, const T& du) { return -du; }, "-")};
}

template <class T>
Tracer<T> operator-(const Tracer<T>& tr_x) {
  return {std::make_shared<NodeUnary<T, T>>(
      tr_x.node(), [](const T& x) { return -x; },
      [](const T&, const T& du) { return -du; }, "-")};
}

template <class T, class Scal>
Tracer<T> operator*(const Tracer<T>& tr_x, const Scal& y) {
  return {std::make_shared<NodeUnary<T, T>>(
      tr_x.node(), [y](const T& x) { return x * y; },
      [y](const T&, const T& du) { return du * y; }, "*")};
}

template <class T, class Scal>
Tracer<T> operator*(const Scal& x, const Tracer<T>& tr_y) {
  return {std::make_shared<NodeUnary<T, T>>(
      tr_y.node(), [x](const T& y) { return x * y; },
      [x](const T&, const T& du) { return x * du; }, "*")};
}

template <class T, class Scal>
Tracer<T> operator/(const Tracer<T>& tr_x, const Scal& y) {
  return {std::make_shared<NodeUnary<T, T>>(
      tr_x.node(), [y](const T& x) { return x / y; },
      [y](const T&, const T& du) { return du / y; }, "/")};
}

template <class T>
Tracer<T> transpose(const Tracer<T>& tr_x) {
  return {std::make_shared<NodeUnary<T, T>>(
      tr_x.node(), [](const T& x) { return x.transpose(); },
      [](const T&, const T& du) { return du.transpose(); }, "*")};
}

template <class T>
Tracer<T> sum(const Tracer<Matrix<T>>& tr_x) {
  return {std::make_shared<NodeUnary<T, Matrix<T>>>(
      tr_x.node(), [](const Matrix<T>& x) { return x.sum(); },
      [](const Matrix<T>& x, T du) { return du * Matrix<T>::ones_like(x); },
      "sum")};
}

template <class T>
Tracer<T> sin(const Tracer<T>& tr_x) {
  using std::cos;
  using std::sin;
  return ApplyScalarFunction<T>(
      tr_x, [](const T& x) { return sin(x); },
      [](const T& x) { return cos(x); }, "sin");
}

template <class T>
Tracer<T> cos(const Tracer<T>& tr_x) {
  using std::cos;
  using std::sin;
  return ApplyScalarFunction<T>(
      tr_x, [](const T& x) { return cos(x); },
      [](const T& x) { return -sin(x); }, "cos");
}

template <class T>
Tracer<T> exp(const Tracer<T>& tr_x) {
  using std::exp;
  return ApplyScalarFunction<T>(
      tr_x, [](const T& x) { return exp(x); },
      [](const T& x) { return exp(x); }, "exp");
}

template <class T>
Tracer<T> log(const Tracer<T>& tr_x) {
  using std::log;
  return ApplyScalarFunction<T>(
      tr_x, [](const T& x) { return log(x); },  //
      [](const T& x) { return 1 / x; }, "log");
}

////////////////////////////////////////
// Binary operations.
////////////////////////////////////////

template <class T>
Tracer<T> operator+(Tracer<T> tr_x, Tracer<T> tr_y) {
  return {std::make_shared<NodeBinary<T, T, T>>(
      tr_x.node(), tr_y.node(),  //
      [](const T& x, const T& y) { return x + y; },
      [](const T&, const T&, const T& du) { return du; },
      [](const T&, const T&, const T& du) { return du; }, "+")};
}

template <class T>
Tracer<T> operator-(Tracer<T> tr_x, Tracer<T> tr_y) {
  return {std::make_shared<NodeBinary<T, T, T>>(
      tr_x.node(), tr_y.node(),  //
      [](const T& x, const T& y) { return x - y; },
      [](const T&, const T&, const T& du) { return du; },
      [](const T&, const T&, const T& du) { return -du; }, "-")};
}

template <class T>
Tracer<T> operator*(Tracer<T> tr_x, Tracer<T> tr_y) {
  return {std::make_shared<NodeBinary<T, T, T>>(
      tr_x.node(), tr_y.node(),  //
      [](const T& x, const T& y) { return x * y; },
      [](const T&, const T& y, const T& du) { return y * du; },
      [](const T& x, const T&, const T& du) { return x * du; }, "*")};
}

template <class T>
Tracer<T> operator/(Tracer<T> tr_x, Tracer<T> tr_y) {
  return {std::make_shared<NodeBinary<T, T, T>>(
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
void Print(std::ostream& out, const Node<T>* node) {
  out << "Node(" << node->name() << ", " << GetTypeName<T>() << ")";
}

template <class T>
void Print(std::ostream& out, const NodeVar<T>* node) {
  out << "NodeVar(" << node->var() << ")";
}

template <class T, class X>
void Print(std::ostream& out, const NodeUnary<T, X>* node) {
  out << "NodeUnary(" << node->name() << "[" << node->x()->name() << "]"
      << ", " << GetTypeName<T>() << "[" << GetTypeName<T>() << "]"
      << ")";
}

template <class T, class X, class Y>
void Print(std::ostream& out, const NodeBinary<T, X, Y>* node) {
  out << "NodeBinary(" << node->name() << "[" << node->x()->name() << ","
      << node->y()->name() << "]"
      << ", " << GetTypeName<T>() << "[" << GetTypeName<T>() << ","
      << GetTypeName<T>() << "]"
      << ")";
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

template <class T>
std::ostream& operator<<(std::ostream& out, const Node<T>& node) {
  node.Print(out);
  return out;
}

template <class T>
std::ostream& operator<<(std::ostream& out, const Tracer<T>& tracer) {
  out << "Tracer(";
  tracer.node()->Print(out);
  out << ")";
  return out;
}

template <class T>
void PrintTree(std::ostream& out, const Tracer<T> tracer) {
  PrintTree(out, tracer.node().get());
}

template <class T>
void PrintTree(std::ostream& out, const Node<T>* node, int depth = 0) {
  out << std::string(depth * 4, '.');
  {
    if (auto ptr = dynamic_cast<const NodeVar<T>*>(node)) {
      out << *ptr << '\n';
    }
  }
  {
    if (auto ptr = dynamic_cast<const NodeUnary<T, T>*>(node)) {
      out << *ptr << '\n';
      PrintTree(out, ptr->x().get(), depth + 1);
    }
  }
  {
    if (auto ptr = dynamic_cast<const NodeBinary<T, T, T>*>(node)) {
      out << *ptr << '\n';
      PrintTree(out, ptr->x().get(), depth + 1);
      PrintTree(out, ptr->y().get(), depth + 1);
    }
  }
}

/*
template <class T>
void PrintDot(std::ostream& out, const Node<T>* terminal) {
  std::map<const Node<T>*, std::string> names;
  std::function<void(const Node<T>*, int)> print =
      [&print, &out, &names](const Node<T>* node, int depth) {
        const std::string pad(depth * 4, ' ');
        if (!names.count(node)) {
          const size_t i = names.size();
          names[node] = "n" + std::to_string(i);
          out << pad;
          out << names[node] << " [label=\"" << node->name() << "\"]\n";
        }

        {
          if (auto ptr = dynamic_cast<const NodeVar<T>*>(node)) {
          }
        }
        {
          if (auto ptr = dynamic_cast<const NodeUnary<T>*>(node)) {
            print(ptr->x().get(), depth + 1);
            out << pad;
            out << names[ptr->x().get()] << " -> " << names[node] << '\n';
          }
        }
        {
          if (auto ptr = dynamic_cast<const NodeBinary<T>*>(node)) {
            print(ptr->x().get(), depth + 1);
            print(ptr->y().get(), depth + 1);
            out << pad;
            out << names[ptr->x().get()] << " -> " << names[node] << '\n';
            out << pad;
            out << names[ptr->y().get()] << " -> " << names[node] << '\n';
          }
        }
      };
  out << "digraph {\n";
  out << "    node [shape=circle, margin=0]\n";
  print(terminal, 1);
  out << "}\n";
}

template <class T>
void PrintDot(std::ostream& out, const Tracer<T>& tracer) {
  PrintDot(out, tracer.node().get());
}

template <class T>
void PrintDot(std::string path, const Node<T>* terminal) {
  std::ofstream fout(path);
  PrintDot(fout, terminal);
}

template <class T>
void PrintDot(std::string path, const Tracer<T>& tracer) {
  PrintDot(path, tracer.node().get());
}
*/
