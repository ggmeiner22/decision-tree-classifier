#!/usr/bin/env bash
# ============================================================
# run_all.sh
#
# End-to-end experiment runner for the Decision Tree project.
#
# What this script does:
#   1. Compiles the project using make.
#   2. Runs all required evaluation modes:
#        - testTennis
#        - testIris
#        - testIrisNoisy
#   3. Generates the noisy-Iris accuracy plot using gnuplot.
#
# Purpose:
#   Provides a single reproducible command that builds,
#   evaluates, and visualizes results exactly the same
#   way across different machines.
#
# Notes:
#   - set -euo pipefail ensures the script exits on errors.
#   - Requires gnuplot to be installed.
# ============================================================
set -euo pipefail

make

echo "=== testTennis ==="
./dtree testTennis data/tennis-attr.txt data/tennis-train.txt data/tennis-test.txt

echo
echo "=== testIris ==="
./dtree testIris data/iris-attr.txt data/iris-train.txt data/iris-test.txt --holdout 0.2 --seed 5

echo
echo "=== testIrisNoisy ==="
./dtree testIrisNoisy data/iris-attr.txt data/iris-train.txt data/iris-test.txt --holdout 0.2 --seed 5 --out iris_noisy.csv

echo "Generating plot..."

gnuplot -persist scripts/plot_iris_noisy.gp

echo "Created plot @ iris_noisy.png"

echo "Done."
