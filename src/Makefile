CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic -O2 -Wshadow
BUILD = build

all: \
	$(BUILD)/test_dual \
	$(BUILD)/test_reverse \
	$(BUILD)/test_matrix \

$(BUILD):
	mkdir "$@"

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

clean:
	$(RM) -r $(BUILD)

.SUFFIXES:
.PHONY: clean all
