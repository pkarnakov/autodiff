CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic -O3 -Wshadow
BUILD = build

all: \
	$(BUILD)/test_dual \
	$(BUILD)/test_reverse \
	$(BUILD)/test_matrix \
	$(BUILD)/test_optimizer \
	$(BUILD)/poisson \

run_poisson: $(BUILD)/poisson
	$(BUILD)/poisson
	make poisson.pdf
	./plot_field.py uref.dat uref.png
	make plot_u

$(BUILD):
	mkdir "$@"

plot_u:
	for u in u_*.dat ; do echo "$$u" ; ./plot_field.py "$$u" "$${u%.dat}.png" ; done

%.pdf: %.gv
	dot "$<" -Tpdf -o "$@"

%.svg: %.gv
	dot "$<" -Tsvg -o "$@"

$(BUILD)/test_dual: test_dual.cpp macros.h dual.h matrix.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o "$@"

$(BUILD)/test_reverse: test_reverse.cpp macros.h reverse.h matrix.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o "$@"

$(BUILD)/test_matrix: test_matrix.cpp macros.h matrix.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o "$@"

$(BUILD)/test_optimizer: test_optimizer.cpp macros.h optimizer.h matrix.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o "$@"

$(BUILD)/poisson: poisson.cpp macros.h reverse.h matrix.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o "$@"

clean:
	$(RM) -r $(BUILD)

cleandata:
	$(RM) *.pdf *.gv *.dat *.png

.SUFFIXES:
.PHONY: all clean cleandata run_poisson plot_u
