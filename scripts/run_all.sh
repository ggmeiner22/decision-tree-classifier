#!/usr/bin/env bash
set -euo pipefail

make

echo "=== testTennis ==="
./dtree testTennis data/tennis-attr.txt data/tennis-train.txt data/tennis-test.txt

echo
echo "=== testIris ==="
./dtree testIris data/iris-attr.txt data/iris-train.txt data/iris-test.txt --holdout 0.2 --seed 1

echo
echo "=== testIrisNoisy ==="
./dtree testIrisNoisy data/iris-attr.txt data/iris-train.txt data/iris-test.txt --holdout 0.2 --seed 1 --out iris_noisy.csv

echo "Done."
