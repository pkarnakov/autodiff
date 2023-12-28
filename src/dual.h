#pragma once

#include <cmath>
#include <iosfwd>

/*
  A dual number (u, ux) is as a class of functions `f(x)` such that
  - f(0) = u
  - f'(0) = ux
  If G() is differentiable at u, then composition Gf(x)=G(f(x)) is such that
  - Gf(0) = G(f(0)) = G(u)
  - Gf'(0) = G'(u) * f'(0) = G'(u) * ux
  This defines certain operations on dual numbers.
*/

template <class T>
class Dual {
 public:
  // Constructor.
  Dual() : u_(T()), ux_(T()) {}
  explicit Dual(const T& u) : u_(u), ux_(T()) {}
  Dual(const T& u, const T& ux) : u_(u), ux_(ux) {}
  Dual(const Dual&) = default;

  // Element access.
  const T& value() const {
    return u_;
  }
  const T& grad() const {
    return ux_;
  }

  // Comparison.
  bool operator==(const Dual& other) const {
    return u_ == other.u_;
  }
  bool operator!=(const Dual& other) const {
    return u_ != other.u_;
  }
  bool operator<(const Dual& other) const {
    return u_ < other.u_;
  }
  bool operator>(const Dual& other) const {
    return u_ > other.u_;
  }
  bool operator<=(const Dual& other) const {
    return u_ <= other.u_;
  }
  bool operator>=(const Dual& other) const {
    return u_ >= other.u_;
  }

  // Assignment.
  Dual& operator=(const Dual&) = default;
  Dual& operator+=(const Dual& other) {
    u_ += other.u_;
    ux_ += other.ux_;
    return *this;
  }
  Dual& operator+=(const T& a) {
    u_ += a;
    return *this;
  }
  Dual& operator-=(const Dual& other) {
    u_ -= other.u_;
    ux_ -= other.ux_;
    return *this;
  }
  Dual& operator-=(const T& a) {
    u_ -= a;
    return *this;
  }
  Dual& operator*=(const Dual& other) {
    *this = *this * other;
    return *this;
  }
  Dual& operator*=(const T& a) {
    u_ *= a;
    ux_ *= a;
    return *this;
  }
  Dual& operator/=(const Dual& other) {
    *this = *this / other;
    return *this;
  }
  Dual& operator/=(const T& a) {
    u_ /= a;
    ux_ /= a;
    return *this;
  }

  // Member functions.
  Dual operator+(const Dual& other) const {
    return Dual{u_ + other.u_, ux_ + other.ux_};
  }
  Dual operator+(const T& a) const {
    return Dual{u_ + a, ux_};
  }
  Dual operator-(const Dual& other) const {
    return Dual{u_ - other.u_, ux_ - other.ux_};
  }
  Dual operator-(const T& a) const {
    return Dual{u_ - a, ux_};
  }
  Dual operator-() const {
    return Dual{-u_, -ux_};
  }
  Dual operator*(const T& a) const {
    return Dual{u_ * a, ux_ * a};
  }
  template <class U>
  Dual operator*(const U& a) const {
    return Dual{u_ * a, ux_ * a};
  }
  Dual operator*(const Dual& other) const {
    return Dual{u_ * other.u_, other.u_ * ux_ + u_ * other.ux_};
  }
  Dual operator/(const Dual& other) const {
    return Dual{
        u_ / other.u_,
        ux_ / other.u_ - (u_ * other.ux_) / (other.u_ * other.u_),
    };
  }
  Dual operator/(const T& a) const {
    return Dual{u_ / a, ux_ / a};
  }

  // Friend functions.
  friend Dual operator+(const T& a, const Dual& dual) {
    return Dual{a + dual.u_, dual.ux_};
  }
  friend Dual operator-(const T& a, const Dual& dual) {
    return Dual{a - dual.u_, -dual.ux_};
  }
  friend Dual operator*(const T& a, const Dual& dual) {
    return Dual{a * dual.u_, a * dual.ux_};
  }
  // XXX: Replacing this with with
  //   friend Dual operator*(const T& a, const Dual& dual) {
  // causes redefinition error.
  // TODO: Revise to avoid ambiguity with member operator*(Dual).
  template <class U>
  friend Dual operator*(const U& a, const Dual& dual) {
    return Dual{a * dual.u_, a * dual.ux_};
  }
  friend Dual operator/(const T& a, const Dual& dual) {
    return Dual{a / dual.u_, -(a * dual.ux_) / (dual.u_ * dual.u_)};
  }
  friend Dual sin(const Dual& dual) {
    using std::cos;
    using std::sin;
    return Dual{sin(dual.u_), cos(dual.u_) * dual.ux_};
  }
  friend Dual cos(const Dual& dual) {
    using std::cos;
    using std::sin;
    return Dual{cos(dual.u_), -sin(dual.u_) * dual.ux_};
  }
  friend Dual exp(const Dual& dual) {
    using std::exp;
    return Dual{exp(dual.u_), exp(dual.u_) * dual.ux_};
  }
  friend Dual log(const Dual& dual) {
    using std::log;
    return Dual{log(dual.u_), dual.ux_ / dual.u_};
  }
  friend Dual pow(const Dual& base, const T& exp) {
    using std::pow;
    return Dual{
        pow(base.u_, exp),
        exp * pow(base.u_, exp - 1) * base.ux_,
    };
  }
  template <class U>
  friend Dual pow(const Dual& base, const U& exp) {
    using std::pow;
    return Dual{
        pow(base.u_, exp),
        exp * pow(base.u_, exp - 1) * base.ux_,
    };
  }
  friend const T& grad(const Dual& dual) {
    return dual.grad();
  }

 private:
  T u_;
  T ux_;
};

template <class T>
T tanh(T x) {
  using std::exp;
  return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

template <class T>
Dual<T> SeedDual(const T& x) {
  return Dual<T>(x, 1);
}

template <class T>
Dual<Dual<T>> SeedDual(const Dual<T>& x) {
  return Dual<Dual<T>>(x, {x.grad(), T()});
}

////////////////////////////////////////
// Output.
////////////////////////////////////////

template <class T>
std::ostream& operator<<(std::ostream& out, const Dual<T>& dual) {
  out << "[" << dual.value() << ", " << dual.grad() << "]";
  return out;
}
