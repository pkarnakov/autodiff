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

template <class T, class D = T>
class Dual {
 public:
  // Constructor.
  Dual() : u_(D()), ux_(D()) {}
  explicit Dual(D u) : u_(u), ux_(D()) {}
  Dual(D u, D ux) : u_(u), ux_(ux) {}
  Dual(const Dual&) = default;

  // Element access.
  explicit operator T() const {
    return u_;
  }
  D value() const {
    return u_;
  }
  D grad() const {
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
  Dual& operator=(const T& a) {
    *this = Dual(a);
    return *this;
  }
  Dual& operator+=(const Dual& other) {
    u_ += other.u_;
    ux_ += other.ux_;
    return *this;
  }
  Dual& operator+=(T a) {
    u_ += a;
    return *this;
  }
  Dual& operator-=(const Dual& other) {
    u_ -= other.u_;
    ux_ -= other.ux_;
    return *this;
  }
  Dual& operator-=(T a) {
    u_ -= a;
    return *this;
  }
  Dual& operator*=(const Dual& other) {
    *this = *this * other;
    return *this;
  }
  Dual& operator*=(T a) {
    u_ *= a;
    ux_ *= a;
    return *this;
  }
  Dual& operator/=(const Dual& other) {
    *this = *this / other;
    return *this;
  }
  Dual& operator/=(T a) {
    u_ /= a;
    ux_ /= a;
    return *this;
  }

  // Member functions.
  Dual operator+(const Dual& other) const {
    return Dual{u_ + other.u_, ux_ + other.ux_};
  }
  Dual operator+(D a) const {
    return Dual{u_ + a, ux_};
  }
  Dual operator-(const Dual& other) const {
    return Dual{u_ - other.u_, ux_ - other.ux_};
  }
  Dual operator-(D a) const {
    return Dual{u_ - a, ux_};
  }
  Dual operator-() const {
    return Dual{-u_, -ux_};
  }
  Dual operator*(T a) const {
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
  Dual operator/(T a) const {
    return Dual{u_ / a, ux_ / a};
  }

  // Friend functions.
  friend Dual operator+(const D& a, const Dual& dual) {
    return Dual{a + dual.u_, dual.ux_};
  }
  friend Dual operator-(const D& a, const Dual& dual) {
    return Dual{a - dual.u_, -dual.ux_};
  }
  friend Dual operator*(T a, const Dual& dual) {
    return Dual{a * dual.u_, a * dual.ux_};
  }
  // XXX: Replacing this with with
  //   friend Dual operator*(const D& a, const Dual& dual) {
  // causes redefinition error.
  // TODO: Revise to avoid ambiguity with member operator*(Dual, Dual).
  template <class U>
  friend Dual operator*(const Dual<T, U>& a, const Dual& dual) {
    return Dual{a * dual.u_, a * dual.ux_};
  }
  friend Dual operator/(T a, const Dual& dual) {
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
  friend Dual pow(const Dual& base, T exp) {
    using std::pow;
    return Dual{
        pow(base.u_, exp),
        exp * pow(base.u_, exp - 1) * base.ux_,
    };
  }
  friend T grad(const Dual& dual) {
    return dual.grad();
  }
  friend std::ostream& operator<<(std::ostream& out, const Dual& dual) {
    out << "[" << dual.u_ << ", " << dual.ux_ << "]";
    return out;
  }

 private:
  D u_;
  D ux_;
};

template <class T>
T tanh(T x) {
  using std::exp;
  return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

template <class T>
Dual<T, T> SeedDual(T x) {
  return Dual<T, T>(x, 1);
}

template <class T, class D>
Dual<T, Dual<T, D>> SeedDual(Dual<T, D> x) {
  return Dual<T, Dual<T, D>>(x, {x.grad(), D()});
}
