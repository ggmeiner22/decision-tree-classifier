#pragma once
#include "Dataset.h"
#include "Metrics.h"
#include <memory>
#include <unordered_map>
#include <set>
// Evaluation utilities for reporting classification accuracy
// and formatting results for display.



// Decision tree classifier supporting discrete and continuous attributes.
// Provides training, prediction, rule extraction, rule evaluation, and
// reduced-error post-pruning functionality.
struct TreeNode {
    bool is_leaf = false;

    // leaf
    int predicted_class = -1;
    std::vector<int> class_counts;

    // split
    int attr_index = -1;
    bool is_continuous_split = false;
    double threshold = 0.0; // for continuous
    // children: for discrete -> map value->child
    std::unordered_map<std::string, std::unique_ptr<TreeNode>> child_by_value;
    // for continuous -> left/right
    std::unique_ptr<TreeNode> left;  // <= threshold
    std::unique_ptr<TreeNode> right; // > threshold
};

// Configuration parameters controlling tree growth behavior,
// including minimum samples required to split and maximum tree depth.
struct TreeParams {
    int min_samples_split = 2;
    int max_depth = 1000; // effectively unlimited
};


// Implements a decision tree learning algorithm with:
// - entropy-based attribute selection
// - support for discrete and continuous attributes
// - rule extraction and reduced-error pruning.
class DecisionTree {
public:
    // Initializes the decision tree with optional training parameters.
    explicit DecisionTree(TreeParams p = TreeParams()) : params_(p) {}

    // Trains the decision tree using the provided dataset by recursively
    // building nodes based on information gain.
    void fit(const Dataset& train);

    // Predicts the class label for a single example by traversing the tree.
    int predict_one(const DatasetSpec& spec, const Example& ex) const;

    // Evaluates tree performance on a dataset and returns accuracy statistics.
    AccuracyReport evaluate(const Dataset& ds) const;

    // Prints a human-readable representation of the decision tree structure.
    void print_tree(const DatasetSpec& spec) const;

    // Represents a single condition within a rule, either discrete equality
    // or continuous threshold comparison.
    struct Condition {
        int attr_index = -1;
        bool is_cont = false;
        // discrete:
        std::string eq_value;
        // continuous:
        double threshold = 0.0;
        bool leq = true; // if cont: <= thresh else >
    };

    // Represents an IF-THEN rule extracted from a root-to-leaf path in the tree.
    struct Rule {
        std::vector<Condition> conds;
        int predicted_class = -1;
        std::vector<int> class_counts;
    };

    // Converts the trained decision tree into a list of rules.
    std::vector<Rule> extract_rules(const DatasetSpec& spec) const;

    // Predicts using an ordered list of rules; returns default class if none match.
    int predict_one_rules(const DatasetSpec& spec, const Example& ex,
                          const std::vector<Rule>& rules, int default_class) const;

    // Computes prediction accuracy using rule-based classification.                     
    AccuracyReport evaluate_rules(const Dataset& ds, const std::vector<Rule>& rules, int default_class) const;

    // Applies reduced-error pruning by removing rule conditions that do not
    // decrease accuracy on a pruning dataset.
    std::vector<Rule> post_prune_rules(const Dataset& prune_set,
                                       const std::vector<Rule>& rules,
                                       int default_class) const;

    // Prints rules in readable IF-THEN format with class distributions.
    static void print_rules(const DatasetSpec& spec, const std::vector<Rule>& rules);

    // Returns the default fallback class used when prediction is ambiguous.
    int default_class() const { 
        return default_class_; 
    }

private:
    TreeParams params_;
    std::unique_ptr<TreeNode> root_;
    int default_class_ = -1;

    // Recursively constructs the decision tree by selecting the best split
    // and generating child nodes.
    std::unique_ptr<TreeNode> build(const Dataset& ds, const std::vector<int>& rows,
                                    const std::vector<int>& avail_attrs, int depth);

    // Computes entropy given class count statistics.
    double entropy_counts(const std::vector<int>& counts) const;

    // Returns the index of the majority class from class counts.
    int argmax_counts(const std::vector<int>& counts) const;

    // Stores information about the best attribute split found during training,
    // including gain, partitions, and thresholds.
    struct BestSplit {
        int attr = -1;
        bool is_cont = false;
        double threshold = 0.0;
        double gain = -1e9;
        // for discrete, partitions by value -> row indices
        std::unordered_map<std::string, std::vector<int>> parts_disc;
        // for continuous, left/right row indices
        std::vector<int> left_rows, right_rows;
    };

    // Determines the optimal attribute split based on information gain.
    BestSplit choose_best_split(const Dataset& ds, const std::vector<int>& rows,
                                const std::vector<int>& avail_attrs) const;

    // Computes class label frequencies for a subset of dataset rows.                           
    std::vector<int> class_counts_for(const Dataset& ds, const std::vector<int>& rows) const;

    // Recursive helper for printing tree structure with indentation formatting.
    void print_node(const DatasetSpec& spec, const TreeNode* node,
                              const std::string& indent, bool is_root) const;

    // Recursively traverses the tree to convert paths into rules.
    void extract_rules_rec(const DatasetSpec& spec, const TreeNode* node,
                           std::vector<Condition>& path, std::vector<Rule>& out) const;

    // Recursively traverses the tree to convert paths into rules.
    bool rule_matches(const DatasetSpec& spec, const Example& ex, const Rule& r) const;
};
