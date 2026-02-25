#include "Dataset.h"
#include "DecisionTree.h"
#include "Noise.h"
#include "Metrics.h"
#include "Util.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>

// Prints in raw string literal format a debug message for 
// how to use the function to the user if used incorrectly
static void usage() {
    std::cout <<
R"(Usage:
  ./dtree testTennis  <attr> <train> <test>
  ./dtree testIris    <attr> <train> <test> [--holdout 0.2] [--seed 1]
  ./dtree testIrisNoisy <attr> <train> <test> [--seed 1] [--holdout 0.2] [--out iris_noisy.csv]

Notes:
- testTennis: prints the tree, tree accuracy (train/test), rules, rule accuracy (train/test) (no pruning).
- testIris:   prints tree, tree accuracy (train/test), rules after rule post-pruning, rule accuracy (train/test).
- testIrisNoisy: corrupts training labels from 0%..20% in 2% increments. Evaluates on uncorrupted test set
  with and without rule post-pruning, then outputs CSV for plotting.

)";
}

// Compares C strings in the parameters ([--seed 1] or [--holdout 0.2])
static bool arg_eq(const char* a, const char* b) { 
    return std::strcmp(a,b)==0; 
}

// Converts Command Line Interface (CLI) text like "0.2" to a double
static double parse_double(const char* s) {
    return util::to_double(std::string(s));
}

// Converts CLI text like 123 to an unsigned int
static unsigned parse_uint(const char* s) {
    char* end=nullptr;
    unsigned long v = std::strtoul(s, &end, 10);
    // Validates that the whole string was consumed
    if (end==s || *end!='\0') {
        throw std::runtime_error(std::string("Expected integer, got: ")+s);
    }
    return (unsigned)v;
}

// Prints a section divider
static void print_header(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

static void run_testTennis(const std::string& attr, const std::string& trainf, const std::string& testf) {
    auto spec = Dataset::load_spec(attr);  // Read attribute definitions + class labels
    auto train = Dataset::load_data(spec, trainf);  // Load training examples
    auto test  = Dataset::load_data(spec, testf);  // Load test examples

    DecisionTree tree;  // Creates a decision tree
    tree.fit(train);  // Train the tree using the training set

    print_header("Decision Tree");
    tree.print_tree(spec);  // Print the learned decision tree using attribute names

    auto tr_acc = tree.evaluate(train);  // Evaluate tree accuracy on training data
    auto te_acc = tree.evaluate(test);  // Evaluate tree accuracy on test data

    print_header("Tree accuracy");
    // Print correct/total and percentage accuracy
    std::cout << "train: " << tr_acc.correct << "/" << tr_acc.total << " = " << fmt_pct(tr_acc.accuracy()) << "\n";
    std::cout << "test : " << te_acc.correct << "/" << te_acc.total << " = " << fmt_pct(te_acc.accuracy()) << "\n";

    print_header("Rules (no pruning)");
    auto rules = tree.extract_rules(spec);  // Convert each root->leaf path into an IF-THEN rule
    DecisionTree::print_rules(spec, rules);  // Prints the rules

    auto tr_r = tree.evaluate_rules(train, rules, tree.default_class());  // Rule accuracy on train
    auto te_r = tree.evaluate_rules(test, rules, tree.default_class());  // Rule accuracy on test

    print_header("Rule accuracy (no pruning)");
    // Print correct/total and percentage accuracy
    std::cout << "train: " << tr_r.correct << "/" << tr_r.total << " = " << fmt_pct(tr_r.accuracy()) << "\n";
    std::cout << "test : " << te_r.correct << "/" << te_r.total << " = " << fmt_pct(te_r.accuracy()) << "\n";
}

// Runs the "testIris" mode: trains on train-split, prunes rules using holdout-split, and evaluates on test
static void run_testIris(const std::string& attr, const std::string& trainf, const std::string& testf,
                         double holdout, unsigned seed) {
    auto spec = Dataset::load_spec(attr);  // Load iris attribute specification
    auto full_train = Dataset::load_data(spec, trainf);  // Load train file
    auto test  = Dataset::load_data(spec, testf);  // Load test file

    auto split = full_train.split_holdout(holdout, seed);  // Randomly split full_train into (train, prune)
    auto train = split.first;     // Used to train the tree
    auto prune = split.second;    // Used to post-prune extracted rules

    DecisionTree tree;
    tree.fit(train);     // Train tree on training subset only

    print_header("Decision Tree");
    tree.print_tree(spec);   // Print tree structure

    auto tr_acc = tree.evaluate(train);   // Tree accuracy on training subset
    auto te_acc = tree.evaluate(test);   // Tree accuracy on test set

    print_header("Tree accuracy");
    std::cout << "train: " << tr_acc.correct << "/" << tr_acc.total << " = " << fmt_pct(tr_acc.accuracy()) << "\n";
    std::cout << "test : " << te_acc.correct << "/" << te_acc.total << " = " << fmt_pct(te_acc.accuracy()) << "\n";

    print_header("Rules (pre-pruning)");
    auto rules = tree.extract_rules(spec);  // Rules extracted directly from the tree
    DecisionTree::print_rules(spec, rules);  // Print unpruned rules

    auto pruned_rules = tree.post_prune_rules(prune, rules, tree.default_class());
    print_header("Rules (post-pruning)");
    DecisionTree::print_rules(spec, pruned_rules);   // Print pruned/simplified rules

    auto tr_r = tree.evaluate_rules(train, pruned_rules, tree.default_class());  // Pruned rule accuracy on train
    auto te_r = tree.evaluate_rules(test, pruned_rules, tree.default_class());  // Pruned rule accuracy on test

    print_header("Rule accuracy (post-pruning)");
    std::cout << "train: " << tr_r.correct << "/" << tr_r.total << " = " << fmt_pct(tr_r.accuracy()) << "\n";
    std::cout << "test : " << te_r.correct << "/" << te_r.total << " = " << fmt_pct(te_r.accuracy()) << "\n";
}


// Runs the "testIrisNoisy" mode: add label noise to training data (0..20%),
// train tree + rules, prune rules using holdout set, and then evaluate on CLEAN test set.
static void run_testIrisNoisy(const std::string& attr, const std::string& trainf, const std::string& testf,
                              double holdout, unsigned seed, const std::string& out_csv) {
    auto spec = Dataset::load_spec(attr);   // Load spec
    auto clean_train = Dataset::load_data(spec, trainf);   // Load clean training data (will be copied each loop)
    auto test  = Dataset::load_data(spec, testf);   // Load CLEAN test data (never corrupted)

    // Open CSV output file and fail if cannot open
    std::ofstream out(out_csv.c_str());
    if (!out) {
        throw std::runtime_error("Failed to open output CSV: " + out_csv);
    }

    // Write CSV header row so plotting scripts know column meanings
    out << "noise_percent,tree_acc_test,rule_acc_test,pruned_rule_acc_test\n";

    // Sweep noise percent from 0 to 20 in increments of 2
    for (int p = 0; p <= 20; p += 2) {
        Dataset noisy = clean_train;  // Copy clean training set
        corrupt_labels(noisy, (double)p, seed);  // Corrupt p% of TRAINING labels (noise injection)

        auto split = noisy.split_holdout(holdout, seed + 999u);  // Split noisy training set into train/prune
        auto train = split.first;   // Used to fit tree
        auto prune = split.second;   // Used to prune rules

        DecisionTree tree;
        tree.fit(train);  // Fit decision tree on noisy training subset

        auto tree_te = tree.evaluate(test);  // Evaluate tree accuracy on CLEAN test set

        auto rules = tree.extract_rules(spec);  // Extract rules from trained tree
        auto rule_te = tree.evaluate_rules(test, rules, tree.default_class());  // Evaluate rules on CLEAN test set

        auto pruned = tree.post_prune_rules(prune, rules, tree.default_class());  // Post-prune rules using prune set
        auto pruned_te = tree.evaluate_rules(test, pruned, tree.default_class());  // Evaluate pruned rules on clean test

        // Write numeric accuracies (0..1) to CSV for plotting
        out << p << ","
            << tree_te.accuracy() << ","
            << rule_te.accuracy() << ","
            << pruned_te.accuracy() << "\n";

        // Print aligned one-line summary for this noise level
        std::cout << std::left
          << std::setw(10) << ("noise=" + std::to_string(p) + "%")  // Left column: noise label
          << " | "
          << "test_acc(clean_test): "   // Clarify accuracy is on clean test set
          << "tree=" << std::setw(7) << fmt_pct(tree_te.accuracy())  // Tree accuracy (trained on noisy data)
          << " | rules_no_prune=" << std::setw(7) << fmt_pct(rule_te.accuracy())  // Rules extracted from that tree (no pruning)
          << " | rules_post_prune=" << std::setw(7) << fmt_pct(pruned_te.accuracy())  // Rules after post-pruning
          << "\n";
    }
    std::cout << "Wrote: " << out_csv << "\n";  // Confirm CSV file path written
}

int main(int argc, char** argv) {
    try {
        // Must have at least the mode argument
        if (argc < 2) {
            usage(); 
            return 1; 
        }
        std::string mode = argv[1];  // Mode selects which experiment to run

        // Use testTennis data set
        if (mode == "testTennis") {
            if (argc != 5) { 
                usage();
                return 1; 
            }
            run_testTennis(argv[2], argv[3], argv[4]);  // Run experiment
            return 0;
        }

        // Use TestIris data set
        if (mode == "testIris") {
            if (argc < 5) { 
                usage(); 
                return 1; 
            }

            /*
            holdout = portion of the training data intentionally not trained on
            Saved for the following:
                1. Validation
                2. Pruning
                3. Unbiased evaluation during training
            */
            double holdout = 0.2;  // Default holdout fraction (portion of the training data intentionally not trained on)
            unsigned seed = 5;  // Default RNG seed

            // Parse added flags
            for (int i = 5; i < argc; i++) {
                if (arg_eq(argv[i], "--holdout") && i + 1 < argc) { 
                    holdout = parse_double(argv[++i]); 
                } else if (arg_eq(argv[i], "--seed") && i + 1 < argc) { 
                    seed = parse_uint(argv[++i]); 
                } else { 
                    throw std::runtime_error(std::string("Unknown arg: ") + argv[i]); 
                }
            }
            run_testIris(argv[2], argv[3], argv[4], holdout, seed);  // Run iris experiment
            return 0;
        }

        // Use TestIrisNoisy data set
        if (mode == "testIrisNoisy") {
            if (argc < 5) { 
                usage(); 
                return 1; 
            }
            double holdout = 0.2;
            unsigned seed = 5;
            std::string out_csv = "iris_noisy.csv";
            for (int i = 5; i < argc; i++) {
                if (arg_eq(argv[i], "--holdout") && i + 1 < argc) { 
                    holdout = parse_double(argv[++i]); 
                } else if (arg_eq(argv[i], "--seed") && i + 1 < argc) { 
                    seed = parse_uint(argv[++i]); 
                } else if (arg_eq(argv[i], "--out") && i + 1 < argc) {  // Output CSV override
                    out_csv = argv[++i]; 
                } else { 
                    throw std::runtime_error(std::string("Unknown arg: ") + argv[i]); 
                }
            }
            run_testIrisNoisy(argv[2], argv[3], argv[4], holdout, seed, out_csv);  // Run iris noisy experiment
            return 0;
        }

        usage();  // If mode is unknown, show usage
        return 1;  // indicates improper usage
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;  // indicates runtime error
    }
}
