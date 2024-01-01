# autodiff

<https://pkarnakov.github.io/autodiff>

## Dual

[`test_dual.cpp`](src/test_dual.cpp)

```
cd src
make build/test_dual && build/test_dual
```

## Reverse

[`test_reverse.cpp`](src/test_reverse.cpp)

```
cd src
make build/test_reverse && build/test_reverse && make reverse_scal1.pdf
```

<img src="https://pkarnakov.github.io/autodiff/media/reverse_scal1.svg" height="200px">

## Poisson

[`poisson.cpp`](src/poisson.cpp)

```
cd src
make run_poisson
```

| graph | inferred |reference |
:---:|:--:|:---:
<img src="https://pkarnakov.github.io/autodiff/media/poisson/poisson.svg" height="200px"> | <img src="https://pkarnakov.github.io/autodiff/media/poisson/u_00010.png" height="200px"> | <img src="https://pkarnakov.github.io/autodiff/media/poisson/uref.png" height="200px">

## Poisson multigrid

| graph | inferred| reference |
:---:|:--:|:---:
<img src="https://pkarnakov.github.io/autodiff/media/poisson_mg/poisson.svg" height="200px"> | <img src="https://pkarnakov.github.io/autodiff/media/poisson_mg/u_00010.png" height="200px"> | <img src="https://pkarnakov.github.io/autodiff/media/poisson_mg/uref.png" height="200px">
