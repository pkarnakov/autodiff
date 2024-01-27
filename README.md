# autodiff

## Clone

```
git clone https://github.com/pkarnakov/autodiff.git
```

## Requirements

* C++17 compiler
* CMake

## Build

```
make
```

## Interactive demos

These demos run interactively in the web browser through [Emscripten](https://emscripten.org).
They solve problems for partial differential equations using [ODIL](https://github.com/cselab/odil).

| [<img src="https://pkarnakov.github.io/autodiff/media/wasm_poisson.png" width=120>](https://pkarnakov.github.io/autodiff/demos/poisson.html) | [<img src="https://pkarnakov.github.io/autodiff/media/wasm_wave.png" width=120>](https://pkarnakov.github.io/autodiff/demos/wave.html) | [<img src="https://pkarnakov.github.io/autodiff/media/wasm_heat.png" width=120>](https://pkarnakov.github.io/autodiff/demos/heat.html) | [<img src="https://pkarnakov.github.io/autodiff/media/wasm_advection.png" width=120>](https://pkarnakov.github.io/autodiff/demos/advection.html) | [<img src="https://pkarnakov.github.io/autodiff/media/wasm_advection2.png" width=120>](https://pkarnakov.github.io/autodiff/demos/advection2.html) |
|:---:|:---:|:---:|:---:|:---:|
| [Poisson](https://pkarnakov.github.io/autodiff/demos/poisson.html) | [Wave](https://pkarnakov.github.io/autodiff/demos/wave.html) | [Heat](https://pkarnakov.github.io/autodiff/demos/heat.html) | [Advection](https://pkarnakov.github.io/autodiff/demos/advection.html) | [Advection2](https://pkarnakov.github.io/autodiff/demos/advection2.html) |

To build them from [source](wasm/) and run in a local web server do
```
make build_wasm
make serve
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


## Pages

<https://pkarnakov.github.io/autodiff>

