BUILD = build
MAKEFILE = $(BUILD)/Makefile
CMAKE = cmake

default: cmake

cmake: $(MAKEFILE)
	+make -C $(BUILD)

$(MAKEFILE):
	mkdir -p "$(BUILD)"
	cd "$(BUILD)" && $(CMAKE) ..

pages:
	git clone -b gh-pages --single-branch git@github.com:pkarnakov/autodiff.git pages

$(BUILD)/poisson: $(MAKEFILE)
	+make -C $(BUILD) $@

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

clean:
	rm -rf $(BUILD)

test:
	cd $(BUILD) && ctest -j`nproc --all`

test_update:
	rm -vf tests/ref/*.out
	cd $(BUILD) && ctest -j`nproc --all`

.SUFFIXES:
.PHONY: all clean test test_update run_poisson run_poisson_cl
