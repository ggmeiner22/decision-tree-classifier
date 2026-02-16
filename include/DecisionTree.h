#pragma once
#include "Dataset.h"
#include "Metrics.h"
#include <memory>
#include <unordered_map>
#include <set>

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

struct TreeParams {
    int min_samples_split = 2;
    int max_depth = 1000; // effectively unlimited
};

class DecisionTree {
public:
    explicit DecisionTree(TreeParams p = TreeParams()) : params_(p) {}

    void fit(const Dataset& train);
    int predict_one(const DatasetSpec& spec, const Example& ex) const;
    AccuracyReport evaluate(const Dataset& ds) const;

    // pretty printing: pre-order, deeper indented, leaves show class distribution
    void print_tree(const DatasetSpec& spec) const;

    // rule extraction
    struct Condition {
        int attr_index = -1;
        bool is_cont = false;
        // discrete:
        std::string eq_value;
        // continuous:
        double threshold = 0.0;
        bool leq = true; // if cont: <= thresh else >
    };
    struct Rule {
        std::vector<Condition> conds;
        int predicted_class = -1;
        std::vector<int> class_counts;
    };

    std::vector<Rule> extract_rules(const DatasetSpec& spec) const;

    // apply rules (first-match). If none matches, use default_class.
    int predict_one_rules(const DatasetSpec& spec, const Example& ex,
                          const std::vector<Rule>& rules, int default_class) const;
    AccuracyReport evaluate_rules(const Dataset& ds, const std::vector<Rule>& rules, int default_class) const;

    // rule post-pruning (reduced error pruning on prune_set)
    std::vector<Rule> post_prune_rules(const Dataset& prune_set,
                                       const std::vector<Rule>& rules,
                                       int default_class) const;

    static void print_rules(const DatasetSpec& spec, const std::vector<Rule>& rules);

    int default_class() const { return default_class_; }

private:
    TreeParams params_;
    std::unique_ptr<TreeNode> root_;
    int default_class_ = -1;

    std::unique_ptr<TreeNode> build(const Dataset& ds, const std::vector<int>& rows,
                                    const std::vector<int>& avail_attrs, int depth);

    // splitting helpers
    double entropy_counts(const std::vector<int>& counts) const;
    int argmax_counts(const std::vector<int>& counts) const;

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

    BestSplit choose_best_split(const Dataset& ds, const std::vector<int>& rows,
                                const std::vector<int>& avail_attrs) const;

    std::vector<int> class_counts_for(const Dataset& ds, const std::vector<int>& rows) const;

    void print_node(const DatasetSpec& spec, const TreeNode* node,
                              const std::string& indent, bool is_root) const;

    void extract_rules_rec(const DatasetSpec& spec, const TreeNode* node,
                           std::vector<Condition>& path, std::vector<Rule>& out) const;

    bool rule_matches(const DatasetSpec& spec, const Example& ex, const Rule& r) const;
};
