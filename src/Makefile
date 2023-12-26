CXXFLAGS=-std=c++14 -Wall -Wextra -pedantic -O3 -Wshadow
CXX=g++

run: main
	./$<

main: main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f main

.PHONY: clean run
