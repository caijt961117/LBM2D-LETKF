#!/bin/bash -ue


CASE=prf2023
log=log/lyapnov.txt

set -o pipefail; {
echo nature
TEST=LYAPNOV_NATURE make clean resultcleanv
TEST=LYAPNOV_NATURE make -j
TEST=LYAPNOV_NATURE make run

for el in 1 2 3 4 5 6; do
    e="1e-$el"
    echo $e
    TEST=LYAPNOV LY_EPSILON=$e make clean -j
    TEST=LYAPNOV LY_EPSILON=$e make -j
    TEST=LYAPNOV LY_EPSILON=$e make run
    make -C bindiff ly
    mv bindiff/test/test_cal{,_${CASE}_eps$e}.csv
done
} |& tee $log
