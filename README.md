# autodiff

## Clone

```
git clone https://github.com/pkarnakov/autodiff.git
```

## Pages

<https://pkarnakov.github.io/autodiff>

## Requirements

* C++17 compiler
* CMake

## Build

```
make
```

## Applications

### Dual

[`test_dual.cpp`](tests/test_dual.cpp)

```
build/test_dual
```

### Reverse

[`test_reverse.cpp`](tests/test_reverse.cpp)

```
build/test_reverse && make reverse_scal1.pdf
```

<img src="https://pkarnakov.github.io/autodiff/media/reverse_scal1.svg" height="200px">

### Poisson

[`poisson.cpp`](src/poisson.cpp)

```
make run_poisson
```

| graph | inferred |reference |
:---:|:--:|:---:
<img src="https://pkarnakov.github.io/autodiff/media/poisson/poisson.svg" height="200px"> | <img src="https://pkarnakov.github.io/autodiff/media/poisson/u_00010.png" height="200px"> | <img src="https://pkarnakov.github.io/autodiff/media/poisson/uref.png" height="200px">

### Poisson multigrid

| graph | inferred| reference |
:---:|:--:|:---:
<img src="https://pkarnakov.github.io/autodiff/media/poisson_mg/poisson.svg" height="200px"> | <img src="https://pkarnakov.github.io/autodiff/media/poisson_mg/u_00010.png" height="200px"> | <img src="https://pkarnakov.github.io/autodiff/media/poisson_mg/uref.png" height="200px">


