N ?= 4

.PHONY: all test a.out

all: a.out

a.out: main.cpp
	g++ -o a.out main.cpp -O2 -std=c++14 -D N_ENS=$(N)
	ldd a.out

test: a.out
	./a.out ../io/nature/0 ../io/calc 2000 2> test/test_ensemble.csv

