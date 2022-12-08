N ?= 4
CXX := g++

all:
	$(CXX) main.cpp -O2 -o a.out
	ldd a.out

test: test_obs test_calc

test_obs: all
	./a.out ../io/nature/0 ../io/observed/0     2000 > test/test_obs.csv
	sed -i '/UCX/d' test/test_obs.csv

test_calc: all
	./a.out ../io/nature/0 ../io/calc/0     2000 > test/test_cal.csv
	sed -i '/UCX/d' test/test_cal.csv

test_calc_ens: all
	for k in `seq 0 $(N)`; do ./a.out ../io/nature/0 ../io/calc/$$k 2000 > test/test_cal_ens_$$k.csv; sed -i '/UCX/d' test/test_cal_ens_$$k.csv; done

ly: all
	./a.out ../io/nature/0 ../io/calc/0     10000 > test/test_cal.csv
