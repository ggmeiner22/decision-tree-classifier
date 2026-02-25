# decision-tree-classifier
A deterministic C++11 implementation of a decision tree classifier with rule extraction,
post-pruning, and noise robustness experiments on Tennis and Iris datasets.

## Compilation and Execution

Ensure that you have execution permission using the following:   
```
chmod -R u+w .
chmod +x scripts/run_all.sh
chmod +x src/main.cpp
chmod +x src/DecisionTree.cpp
chmod +x src/Dataset.cpp
```

### Run All
```
./scripts/run_all.sh
```
### Run Individually
#### Build
```
make
```
#### testTennis (no pruning)
```
./dtree testTennis data/tennis-attr.txt data/tennis-train.txt data/tennis-test.txt
```
#### testIris (rule post-pruning enabled)
```
./dtree testIris data/iris-attr.txt data/iris-train.txt data/iris-test.txt --holdout 0.2 --seed 1
```
#### testIrisNoisy (outputs CSV)
```
./dtree testIrisNoisy data/iris-attr.txt data/iris-train.txt data/iris-test.txt --holdout 0.2 --seed 1 --out iris_noisy.csv
```
#### Plotting
```
gnuplot -persist scripts/plot_iris_noisy.gp
```
#### Clean object files
```
make clean
```


## File Structure
```
decision-tree-classifier/
│
├── include/
│   ├── Dataset.h        # Dataset structures and loading
│   ├── DecisionTree.h   # Decision tree API + rule extraction
│   ├── Noise.h          # Deterministic label corruption
│   ├── Metrics.h        # Accuracy reporting utilities
│   └── Util.h           # String/file helper functions
│
├── src/
│   ├── main.cpp         # Program entry + experiment modes
│   ├── Dataset.cpp      # Dataset parsing + holdout splitting
│   └── DecisionTree.cpp # Core tree training & evaluation
│
├── data/
│   ├── tennis-*.txt     # Tennis dataset
│   └── iris-*.txt       # Iris dataset
│
├── scripts/
│   ├── run_all.sh       # Runs all experiments
│   └── plot_iris_noisy.gp  # gnuplot script
│
├── Makefile             # Build instructions
├── README.md
└── LICENSE
```

## File Overview
### Dataset.h / Dataset.cpp
Defines the dataset structures and file parsing logic.
- ```AttributeSpec``` describes each feature (name, discrete values, or continuous flag).
- ```DatasetSpec``` stores attribute metadata and class labels.
- ```Example``` represents one training/test instance.
- ```Dataset``` loads data files, converts class labels into indices, and performs deterministic holdout splitting for pruning experiments.
> This module isolates all data handling so the learning code remains clean.

### DecisionTree.h / DecisionTree.cpp
Implements the core decision tree learning algorithm and rule extraction.
Responsibilities include:
- Entropy and information gain calculation
- Attribute selection with deterministic tie-breaking
- Recursive tree construction
- Prediction using tree traversal
- Pretty-printing of the tree structure
- Rule extraction from leaf paths
- Reduced-error rule post-pruning
- Rule-based prediction and evaluation
> This is the main learning engine of the project.

### Noise.h
Provides deterministic label corruption utilities.
- Implements ```corrupt_labels()``` to flip a fixed percentage of training labels.
- Uses a seeded ```mt19937``` and rejection sampling to guarantee identical results across machines.
> Used primarily by the noisy Iris experiment.

### Metrics.h
Defines accuracy reporting utilities.
- ```AccuracyReport``` tracks correct/total predictions and computes accuracy.
- ```fmt_pct()``` formats accuracy values as percentage strings.
> Separates evaluation logic from the model implementation.

### Util.h
Contains reusable helper functions for string and file processing.
Includes:
- whitespace trimming
- tokenization (split_ws)
- file reading (read_lines)
- case-insensitive comparison
- safe string-to-double conversion
> Keeps parsing logic centralized and reusable.

### main.cpp
Program entry point and experiment controller.
Supports three execution modes:
- `testTennis` – builds a tree, prints rules, and reports accuracy (no pruning).
- `testIris` – trains a tree, performs rule post-pruning using a holdout set.
- `testIrisNoisy` – injects label noise, evaluates robustness, and generates CSV output for plotting.
> Also handles command-line parsing, experiment configuration, and formatted output.

### scripts/run_all.sh
Automation script that:
1. Compiles the project using ```make```
2. Runs all experiments
4. Generates the noisy Iris plot using gnuplot
Provides a single reproducible workflow.

### scripts/plot_iris_noisy.gp
Gnuplot script used to visualize accuracy vs noise level.
Plots:
- Tree accuracy
- Rule accuracy (no pruning)
- Rule accuracy after post-pruning
> Outputs ```iris_noisy.png```.

### data/
Contains dataset files used for experiments.
- ```tennis-*``` files – discrete attribute example
- ```iris-*``` files – continuous attribute example

### Makefile
Defines compilation rules for building the project.
- Uses ```-std=c++11```
- Compiles source files into object files
- Links final executable `dtree`
> Ensures consistent builds across machines.



