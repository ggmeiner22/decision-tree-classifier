CSE 5693 - Decision Tree (C++11) + Rule Post-Pruning
===================================================

This repo contains a from-scratch Decision Tree learner that supports:
- >2 classes
- discrete-valued and continuous-valued attributes
- printing the tree (pre-order traversal, deeper nodes are indented, leaves show class distribution)
- extracting a rule set from the tree
- rule post-pruning (reduced-error pruning using a holdout "prune" set)

No non-standard libraries are used (C++11 only). Intended to run on code01.fit.edu.

Directory layout
----------------
include/   headers
src/       implementation
data/      sample datasets (tennis, iris, bool) as provided
scripts/   helper scripts for running experiments / plotting

Build
-----
make
# produces ./dtree

Run experiments (matches HW2 requirements)
-----------------------------------------
1) testTennis (no pruning; dataset too small)
./dtree testTennis data/tennis-attr.txt data/tennis-train.txt data/tennis-test.txt

Outputs:
- printed tree
- tree accuracy (train/test)
- printed rules
- rule accuracy (train/test)

2) testIris (rule post-pruning enabled)
./dtree testIris data/iris-attr.txt data/iris-train.txt data/iris-test.txt --holdout 0.2 --seed 1

Notes:
- The Iris training set is split into:
  - "train" subset: used to learn the tree
  - "prune" subset: used for reduced-error post-pruning of extracted rules

3) testIrisNoisy (0%..20% label corruption; 2% increment)
./dtree testIrisNoisy data/iris-attr.txt data/iris-train.txt data/iris-test.txt --holdout 0.2 --seed 1 --out iris_noisy.csv

This writes a CSV with columns:
noise_percent, tree_acc_test, rule_acc_test, pruned_rule_acc_test

Plotting
--------
See scripts/plot_iris_noisy.gp for a gnuplot script.
Example:
  gnuplot -persist scripts/plot_iris_noisy.gp

Implementation notes
--------------------
- Split criterion: Information Gain (ID3-style entropy).
- Continuous attribute splits: best threshold chosen by scanning midpoints between sorted unique values.
- Discrete splits: multiway branches by observed attribute value.
- Unseen discrete values at test-time: back off to current node's majority class.

