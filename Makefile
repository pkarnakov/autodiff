BUILD = build
MAKEFILE = $(BUILD)/Makefile
CMAKE = cmake

BUILD_WASM = build_wasm
MAKEFILE_WASM = $(BUILD_WASM)/Makefile
CMAKE_WASM = emcmake cmake
FLAGS_WASM = -DUSE_OPENCL=0 -DBUILD_WASM=1 -DBUILD_POISSON=0 -DBUILD_TESTS=0 -DUSE_MARCH_NATIVE=0

default: build

$(MAKEFILE):
	mkdir -p "$(BUILD)"
	cd "$(BUILD)" && $(CMAKE) ..

build: $(MAKEFILE)
	+make -C $(BUILD)

$(MAKEFILE_WASM):
	mkdir -p "$(BUILD_WASM)"
	cd "$(BUILD_WASM)" && $(CMAKE_WASM) .. $(FLAGS_WASM)

build_wasm: $(MAKEFILE_WASM)
	+make -C $(BUILD_WASM)

serve:
	cd "$(BUILD_WASM)" && emrun --serve_after_exit poisson.html

%.pdf: %.gv
	dot "$<" -Tpdf -o "$@"

%.svg: %.gv
	dot "$<" -Tsvg -o "$@"

run_poisson: $(BUILD)/poisson
	cd $(BUILD) && ./poisson
	make $(BUILD)/poisson.pdf
	./plot_field.py $(BUILD)/uref.dat $(BUILD)/u_*.dat

run_poisson_cl: $(BUILD)/poisson_cl
	cd $(BUILD) && ./poisson_cl
	make $(BUILD)/poisson.pdf
	./plot_field.py $(BUILD)/uref.dat $(BUILD)/u_*.dat

test:
	cd $(BUILD) && ctest -j`nproc --all`

test_update:
	rm -vf tests/ref/*.out
	cd $(BUILD) && ctest -j`nproc --all`

clean:
	rm -rf $(BUILD) $(BUILD_WASM)

pages:
	git clone -b gh-pages --single-branch git@github.com:pkarnakov/autodiff.git pages

.SUFFIXES:
.PHONY: default build clean test test_update run_poisson run_poisson_cl build_wasm serve
