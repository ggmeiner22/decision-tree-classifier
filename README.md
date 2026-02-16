# decision-tree-classifier

Build
make

testTennis (no pruning)
./dtree testTennis data/tennis-attr.txt data/tennis-train.txt data/tennis-test.txt

testIris (rule post-pruning enabled)
./dtree testIris data/iris-attr.txt data/iris-train.txt data/iris-test.txt --holdout 0.2 --seed 1

testIrisNoisy (outputs CSV)
./dtree testIrisNoisy data/iris-attr.txt data/iris-train.txt data/iris-test.txt --holdout 0.2 --seed 1 --out iris_noisy.csv
gnuplot -persist scripts/plot_iris_noisy.gp
