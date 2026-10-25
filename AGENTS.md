# autodiff

Mostly header-only C++17 automatic differentiation (forward dual numbers,
reverse mode on a computational graph, optional OpenCL backend) and
interactive WebAssembly demos solving PDEs with ODIL (optimize the discrete
solution by gradient descent).

## Layout

- `src/`: library and native apps.
  - `dual.h`: forward mode (`Dual<T>`).
  - `reverse.h`: reverse mode (`Var`, graph nodes, `BaseExtra` hooks),
    operations on scalars and `Matrix`.
  - `matrix.h`: dense `Matrix<T>`, 3x3 `conv` / `conv_adjoint`.
  - `optimizer.h`: `optimizer::Adam`.
  - `reverse_cl.h`, `matrix_cl.h`, `opencl.{h,cpp}`, `kernels.cl`: OpenCL
    backend; `kernels.inc` is generated from `kernels.cl` by
    `generate_inc` (do not edit or commit it).
  - `poisson.cpp`, `poisson_cl.cpp`: native Poisson solvers.
  - `profiler.{h,cpp}`.
- `tests/`: `test_*.cpp`, each prints to stdout; `tests/test` diffs the
  output against `tests/ref/<name>.out`.
- `wasm/`: Emscripten demos (`poisson`, `wave`, `heat`, `advection`,
  `advection2`). Each demo is `NAME.cpp` + `NAME.html`; shared code in
  `graphics.{h,cpp}`, `view.{js,css}`, `menu.html` (substituted into pages
  as `@HTML_MENU@` by CMake `configure_file`), `assets/` (colormap, axes,
  with Python generators), `libs/`.
- `pages/`: separate clone of the `gh-pages` branch (`make pages`),
  ignored by the main repo. Published at
  <https://pkarnakov.github.io/autodiff>.
- `plot_field.py`: plots `*.dat` fields written by `poisson`.

## Build and test

- Native: `make` (CMake into `build/`), `make test` (ctest).
  Options: `USE_OPENCL`, `BUILD_POISSON`, `BUILD_TESTS`, `USE_MARCH_NATIVE`.
  Without OpenCL: `cd build && cmake .. -DUSE_OPENCL=0`.
- Reference outputs: a missing `tests/ref/*.out` is created from the
  current output; `make test_update` regenerates all of them. Review the
  diff before committing regenerated references.
- WebAssembly: `make build_wasm` (needs `emcmake`; builds into
  `build_wasm/`), `make serve` (emrun).
- `make run_poisson` runs the native solver and plots results (needs
  Graphviz `dot`, Python with matplotlib).

## Adding a demo

Add `wasm/NAME.cpp` and `wasm/NAME.html` (copy an existing pair), a block
in the `BUILD_WASM` section of `CMakeLists.txt` with the same exported
functions, an entry in `wasm/menu.html`, and a cell in the demo table of
`README.md` (image in `pages/media/wasm_NAME.png`).

## Publishing demos

In `pages/`: `make demos` copies `build_wasm/` output into `pages/demos/`
and records the source revision; `make demos_commit` commits it in the
`gh-pages` clone. Pushing is a separate, explicit step.

## Conventions

- Formatting: `clang-format` (Google style, see `.clang-format`); Python:
  `yapf` (`.style.yapf`).
- Compiler warnings are on (`-Wall -Wextra -pedantic -Wshadow`); keep
  builds warning-free.
- Commit subjects: `area: change`, lowercase, short, e.g.
  `wasm: add heat`, `matrix,reverse: add 3x3 conv`.
