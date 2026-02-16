#include "Dataset.h"
#include "DecisionTree.h"
#include "Noise.h"
#include "Metrics.h"
#include "Util.h"
#include <iostream>
#include <fstream>
#include <cstring>

static void usage() {
    std::cout <<
R"(Usage:
  ./dtree testTennis  <attr> <train> <test>
  ./dtree testIris    <attr> <train> <test> [--holdout 0.2] [--seed 1]
  ./dtree testIrisNoisy <attr> <train> <test> [--seed 1] [--holdout 0.2] [--out iris_noisy.csv]

Notes:
- testTennis: prints the tree, tree accuracy (train/test), rules, rule accuracy (train/test) (no pruning).
- testIris:   prints tree, tree accuracy (train/test), rules after rule post-pruning, rule accuracy (train/test).
- testIrisNoisy: corrupts training labels from 0%..20% in 2% increments; evaluates on uncorrupted test set
  with and without rule post-pruning; outputs CSV for plotting.

)";
}

static bool arg_eq(const char* a, const char* b) { return std::strcmp(a,b)==0; }

static double parse_double(const char* s) {
    return util::to_double(std::string(s));
}
static unsigned parse_uint(const char* s) {
    char* end=nullptr;
    unsigned long v = std::strtoul(s, &end, 10);
    if (end==s || *end!='\0') throw std::runtime_error(std::string("Expected integer, got: ")+s);
    return (unsigned)v;
}

static void print_header(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

static void run_testTennis(const std::string& attr, const std::string& trainf, const std::string& testf) {
    auto spec = Dataset::load_spec(attr);
    auto train = Dataset::load_data(spec, trainf);
    auto test  = Dataset::load_data(spec, testf);

    DecisionTree tree;
    tree.fit(train);

    print_header("Decision Tree");
    tree.print_tree(spec);

    auto tr_acc = tree.evaluate(train);
    auto te_acc = tree.evaluate(test);

    print_header("Tree accuracy");
    std::cout << "train: " << tr_acc.correct << "/" << tr_acc.total << " = " << fmt_pct(tr_acc.accuracy()) << "\n";
    std::cout << "test : " << te_acc.correct << "/" << te_acc.total << " = " << fmt_pct(te_acc.accuracy()) << "\n";

    print_header("Rules (no pruning)");
    auto rules = tree.extract_rules(spec);
    DecisionTree::print_rules(spec, rules);

    auto tr_r = tree.evaluate_rules(train, rules, tree.default_class());
    auto te_r = tree.evaluate_rules(test, rules, tree.default_class());

    print_header("Rule accuracy (no pruning)");
    std::cout << "train: " << tr_r.correct << "/" << tr_r.total << " = " << fmt_pct(tr_r.accuracy()) << "\n";
    std::cout << "test : " << te_r.correct << "/" << te_r.total << " = " << fmt_pct(te_r.accuracy()) << "\n";
}

static void run_testIris(const std::string& attr, const std::string& trainf, const std::string& testf,
                         double holdout, unsigned seed) {
    auto spec = Dataset::load_spec(attr);
    auto full_train = Dataset::load_data(spec, trainf);
    auto test  = Dataset::load_data(spec, testf);

    auto split = full_train.split_holdout(holdout, seed);
    auto train = split.first;
    auto prune = split.second;

    DecisionTree tree;
    tree.fit(train);

    print_header("Decision Tree");
    tree.print_tree(spec);

    auto tr_acc = tree.evaluate(train);
    auto te_acc = tree.evaluate(test);

    print_header("Tree accuracy");
    std::cout << "train: " << tr_acc.correct << "/" << tr_acc.total << " = " << fmt_pct(tr_acc.accuracy()) << "\n";
    std::cout << "test : " << te_acc.correct << "/" << te_acc.total << " = " << fmt_pct(te_acc.accuracy()) << "\n";

    print_header("Rules (pre-pruning)");
    auto rules = tree.extract_rules(spec);
    DecisionTree::print_rules(spec, rules);

    auto pruned_rules = tree.post_prune_rules(prune, rules, tree.default_class());
    print_header("Rules (post-pruning)");
    DecisionTree::print_rules(spec, pruned_rules);

    auto tr_r = tree.evaluate_rules(train, pruned_rules, tree.default_class());
    auto te_r = tree.evaluate_rules(test, pruned_rules, tree.default_class());

    print_header("Rule accuracy (post-pruning)");
    std::cout << "train: " << tr_r.correct << "/" << tr_r.total << " = " << fmt_pct(tr_r.accuracy()) << "\n";
    std::cout << "test : " << te_r.correct << "/" << te_r.total << " = " << fmt_pct(te_r.accuracy()) << "\n";
}

static void run_testIrisNoisy(const std::string& attr, const std::string& trainf, const std::string& testf,
                              double holdout, unsigned seed, const std::string& out_csv) {
    auto spec = Dataset::load_spec(attr);
    auto clean_train = Dataset::load_data(spec, trainf);
    auto test  = Dataset::load_data(spec, testf);

    std::ofstream out(out_csv.c_str());
    if (!out) throw std::runtime_error("Failed to open output CSV: " + out_csv);

    // header
    out << "noise_percent,tree_acc_test,rule_acc_test,pruned_rule_acc_test\n";

    for (int p = 0; p <= 20; p += 2) {
        Dataset noisy = clean_train;
        corrupt_labels(noisy, (double)p, seed + (unsigned)p);

        auto split = noisy.split_holdout(holdout, seed + 999u + (unsigned)p);
        auto train = split.first;
        auto prune = split.second;

        DecisionTree tree;
        tree.fit(train);

        auto tree_te = tree.evaluate(test);

        auto rules = tree.extract_rules(spec);
        auto rule_te = tree.evaluate_rules(test, rules, tree.default_class());

        auto pruned = tree.post_prune_rules(prune, rules, tree.default_class());
        auto pruned_te = tree.evaluate_rules(test, pruned, tree.default_class());

        out << p << ","
            << tree_te.accuracy() << ","
            << rule_te.accuracy() << ","
            << pruned_te.accuracy() << "\n";

        std::cout << "noise " << p << "%  "
                  << "tree=" << fmt_pct(tree_te.accuracy()) << "  "
                  << "rules=" << fmt_pct(rule_te.accuracy()) << "  "
                  << "pruned=" << fmt_pct(pruned_te.accuracy()) << "\n";
    }

    std::cout << "Wrote: " << out_csv << "\n";
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) { usage(); return 1; }
        std::string mode = argv[1];

        if (mode == "testTennis") {
            if (argc != 5) { usage(); return 1; }
            run_testTennis(argv[2], argv[3], argv[4]);
            return 0;
        }

        if (mode == "testIris") {
            if (argc < 5) { usage(); return 1; }
            double holdout = 0.2;
            unsigned seed = 1;
            for (int i=5;i<argc;i++) {
                if (arg_eq(argv[i], "--holdout") && i+1<argc) { holdout = parse_double(argv[++i]); }
                else if (arg_eq(argv[i], "--seed") && i+1<argc) { seed = parse_uint(argv[++i]); }
                else { throw std::runtime_error(std::string("Unknown arg: ") + argv[i]); }
            }
            run_testIris(argv[2], argv[3], argv[4], holdout, seed);
            return 0;
        }

        if (mode == "testIrisNoisy") {
            if (argc < 5) { usage(); return 1; }
            double holdout = 0.2;
            unsigned seed = 1;
            std::string out_csv = "iris_noisy.csv";
            for (int i=5;i<argc;i++) {
                if (arg_eq(argv[i], "--holdout") && i+1<argc) { holdout = parse_double(argv[++i]); }
                else if (arg_eq(argv[i], "--seed") && i+1<argc) { seed = parse_uint(argv[++i]); }
                else if (arg_eq(argv[i], "--out") && i+1<argc) { out_csv = argv[++i]; }
                else { throw std::runtime_error(std::string("Unknown arg: ") + argv[i]); }
            }
            run_testIrisNoisy(argv[2], argv[3], argv[4], holdout, seed, out_csv);
            return 0;
        }

        usage();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
