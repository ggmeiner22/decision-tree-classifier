#include "DecisionTree.h"
#include "Util.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <map>

// Small tolerance for floating-point comparisons
static const double EPS = 1e-12;

double DecisionTree::entropy_counts(const std::vector<int>& counts) const {
    double sum = 0.0;  // Small tolerance for floating-point comparisons

    // Sum class counts
    for (int c : counts) {
        sum += c;
    }

    // No samples means entropy defined as 0
    if (sum <= 0.0) {
        return 0.0;
    }

    double H = 0.0;  // Entropy accumulator
    for (int c : counts) {
        if (c <= 0) {  // Skip zero counts (log(0) undefined)
            continue;
        }
        double p = (double)c / sum;  // Probability of this class at the node
        H -= p * std::log(p) / std::log(2.0);  // H = -Σ p log2(p)
    }
    return H;  // Returns entropy in bits
}

int DecisionTree::argmax_counts(const std::vector<int>& counts) const {
    int best_i = 0;  // Index of most frequent class so far
    int best_v;  // Best count value so far
    if (counts.empty()) {
        best_v = 0;
    } else {
        best_v = counts[0];
    }
    // Scan remaining classes
    for (size_t i = 1; i < counts.size(); ++i) {
        if (counts[i] > best_v) {   // Update if this class count is larger
            best_v = counts[i]; 
            best_i = (int)i; 
        }
    }
    return best_i;   // Return best class index
}

std::vector<int> DecisionTree::class_counts_for(const Dataset& ds, const std::vector<int>& rows) const {
    std::vector<int> counts(ds.spec.class_labels.size(), 0);  // Initialize K class counts to 0
    
    // Increment count for each row’s class
    for (int rid : rows) {
        counts[ds.rows[rid].y] += 1;  
    }
    return counts;  // Increment count for each row’s class
}

DecisionTree::BestSplit DecisionTree::choose_best_split(const Dataset& ds, const std::vector<int>& rows,
                                                        const std::vector<int>& avail_attrs) const {
    BestSplit best;  // Tracks the best split found so far
    const auto parent_counts = class_counts_for(ds, rows);  // Class distribution at current node 
    const double parent_H = entropy_counts(parent_counts);  //Parent node entropy
    const double parent_n = (double)rows.size();   // Parent node sample count

    for (int aidx : avail_attrs) {  // Try splitting on each available attribute
        const auto& attr = ds.spec.attrs[aidx];  // gets attribute data

        if (!attr.is_continuous) {
            // multiway split by discrete value
            std::unordered_map<std::string, std::vector<int>> parts;  // value is a list of row indices
            parts.reserve(attr.values.size() + 2);
            for (int rid : rows) {    // Partition rows by attribute value
                const std::string& v = ds.rows[rid].x[aidx];  // Attribute value for this row
                parts[v].push_back(rid);    // Add row to that value’s bucket
            }
            // information gain
            double child_H = 0.0;   // Weighted average entropy after split
            for (auto& kv : parts) {   // for each branch
                const auto& part_rows = kv.second;   // rows in this branch
                const auto cc = class_counts_for(ds, part_rows);   // class distributon in branch
                const double w = (double)part_rows.size() / parent_n;   // calculates branch weight
                child_H += w * entropy_counts(cc);   // adds weighted entropy
            }

            const double gain = parent_H - child_H;   // IG = H(parent) - Σ w * H(child)
            const int branches = (int)parts.size();    // Number of children created by this split
            int best_branches;  // # of branches in current best
            if (best.is_cont) {
                best_branches = 2;
            } else {
                best_branches = (int)best.parts_disc.size();
            }
            
            /*
            Choose this attribute if:
                1) it has strictly higher information gain, OR
                2) same gain (within EPS) but fewer branches (simpler split), OR
                3) still tied: choose smaller attribute index for deterministic behavior
            */
            if (gain > best.gain + EPS ||
                (std::fabs(gain - best.gain) <= EPS && branches < best_branches) || 
                (std::fabs(gain - best.gain) <= EPS && branches == best_branches && aidx < best.attr)) { 
                best.gain = gain;  // best gain
                best.attr = aidx;  // best attribute
                best.is_cont = false;  // Mark as discrete split
                best.parts_disc = std::move(parts);  // Store partitions
                best.left_rows.clear();   // Clear binary partitions
                best.right_rows.clear();
            }
        } else {
            // continuous: choose threshold that maximizes gain (binary split)
            std::vector<std::pair<double,int>> vals; // Pairs: (numeric value (x), row id (rid))
            vals.reserve(rows.size());  // allocate storage

            // Collect numeric values for this attribute
            for (int rid : rows) {
                const std::string& s = ds.rows[rid].x[aidx];
                const double x = util::to_double(s);  // convert string to double
                vals.push_back({x, rid});   // store value and row ID
            }

            // Sort values to be able to test thresholds between consecutive distinct values
            std::sort(vals.begin(), vals.end(),
                      [](const std::pair<double,int>& p1, const std::pair<double,int>& p2)
                      { return p1.first < p2.first; }
                    );
            // Can't split if fewer than 2 rows
            if (vals.size() < 2) {
                continue;
            }

            // precompute prefix class counts for fast left/right counts at each cut position
            const int K = (int)ds.spec.class_labels.size();
            std::vector<std::vector<int>> prefix(vals.size()+1, std::vector<int>(K,0));  // # of classes
            for (size_t i = 0; i < vals.size(); ++i) {
                prefix[i+1] = prefix[i];  // get previous prefix counts
                prefix[i+1][ ds.rows[vals[i].second].y ] += 1;  // add class of the i-th sorted row
            }
            const auto total = prefix.back();   // total class count of all rows

            double best_gain_a = -1e9;   // Best gain for this continuous attribute so far
            double best_thr = 0.0;   // Threshold that produced best gain
            size_t best_cut = 0;   // Index where left side ends (number of points on left)

            // Try thresholds between each adjacent pair of distinct values
            for (size_t i = 0; i + 1 < vals.size(); ++i) {
                const double x1 = vals[i].first;
                const double x2 = vals[i+1].first;
                if (std::fabs(x2 - x1) < EPS) {  // no midpoint
                    continue;   // skip itentical values
                }
                const double thr = 0.5 * (x1 + x2);  // Candudate threshold midpoint

                std::vector<int> left_counts = prefix[i+1];  // Left side class counts (first i+1 items)
                std::vector<int> right_counts(K,0);    // Right side class counts = total - left
                for (int k = 0; k < K; ++k) {
                    right_counts[k] = total[k] - left_counts[k];
                } 

                const double nL = (double)(i+1);  // left sample count
                const double nR = (double)(vals.size()-(i+1));  // right sample count

                // Weighted entropy after split
                const double child_H = 
                    (nL/parent_n) * entropy_counts(left_counts) + 
                    (nR/parent_n) * entropy_counts(right_counts);
                const double gain = parent_H - child_H;  // Information gain for this threshold

                // Keep best threshold for this attribute
                if (gain > best_gain_a + EPS) {
                    best_gain_a = gain;
                    best_thr = thr;
                    best_cut = i+1;
                }
            }

            const int branches = 2;  // Keep best threshold for THIS attribute

            // Best so far branch count
            int best_branches;
            if (best.is_cont) {
                 best_branches = 2 ;
             } else {
                best_branches = (int)best.parts_disc.size();
             }

             /*
             Update global best split if:
                1) higher gain, OR
                2) same gain but fewer branches, OR
                3) still tied: smaller attribute index
             */
            if (best_gain_a > best.gain + EPS ||
                (std::fabs(best_gain_a - best.gain) <= EPS && branches < best_branches) ||
                (std::fabs(best_gain_a - best.gain) <= EPS && branches == best_branches && aidx < best.attr)) {
                best.gain = best_gain_a;  // best gain overall
                best.attr = aidx;   // best attribute index
                best.is_cont = true;   // split is continuous
                best.threshold = best_thr;   // threashold value
                best.parts_disc.clear();   // clear discrete partions
                best.left_rows.clear();
                best.right_rows.clear();

                // Build left/right row sets from best_cut
                for (size_t i = 0; i < vals.size(); ++i) {
                    if (i < best_cut) {
                        best.left_rows.push_back(vals[i].second);
                    } else {
                        best.right_rows.push_back(vals[i].second);
                    }
                }
            }
        }
    }
    return best;  // Return best split found (may have gain<=0)
}

std::unique_ptr<TreeNode> DecisionTree::build(const Dataset& ds, const std::vector<int>& rows,
                                              const std::vector<int>& avail_attrs, int depth) {
    auto node = std::unique_ptr<TreeNode>(new TreeNode());  // Allocate a new node
    node->class_counts = class_counts_for(ds, rows);   // Store class distribution at this node
    node->predicted_class = argmax_counts(node->class_counts);  // Best class prediction for this node

    // Base Cases
    const int majority = node->predicted_class;  // best class index
    const int maj_count = node->class_counts[majority];  // best class count
    if ((int)rows.size() < params_.min_samples_split ||  // Too few samples to split
        depth >= params_.max_depth ||                    // Reached max recursion depth
        avail_attrs.empty() ||                           // No attributes left to split on
        maj_count == (int)rows.size()) {                 // Node is pure (all same class)
        node->is_leaf = true;                            // Mark as leaf node
        return node;
    }

    // Choose best attribute split by info gain
    BestSplit split = choose_best_split(ds, rows, avail_attrs);
    if (split.attr < 0 || split.gain <= EPS) {  // If no useful split found
        node->is_leaf = true;
        return node;
    }

    node->attr_index = split.attr;                              // Store attribute used for splitting
    node->is_continuous_split = split.is_cont;                  // Store split type (continuous vs discrete)
    node->threshold = split.threshold;                          // Store threshold if continuous

    // Build list of attributes allowed for children:
    // 1.  allow reuse of continuous attributes
    // 2.  remove discrete attribute after splitting to avoid cycles
    std::vector<int> next_avail;
    next_avail.reserve(avail_attrs.size());
    for (int a : avail_attrs) {
        if (a == split.attr && !ds.spec.attrs[a].is_continuous) {  // Remove discrete split attribute
            continue;
        }
        next_avail.push_back(a);
    }

    if (!split.is_cont) {
        for (auto& kv : split.parts_disc) {  // For each discrete value branch
            const std::string& val = kv.first;  // Branch value name
            auto& part_rows = kv.second;       // Rows in this branch
            node->child_by_value[val] = build(ds, part_rows, next_avail, depth+1);  // Recurse build child
        }
        node->is_leaf = false;    // Node is internal
    } else {
        if (split.left_rows.empty() || split.right_rows.empty()) {  // Invalid split
            node->is_leaf = true;
            return node;
        }
        node->left = build(ds, split.left_rows, next_avail, depth+1);  // Left child: x <= threshold
        node->right = build(ds, split.right_rows, next_avail, depth+1);  // Right child: x > threshold
        node->is_leaf = false;  // Node is internal
    }
    return node;  // Return subtree rooted here
}

void DecisionTree::fit(const Dataset& train) {
    // compute default class from training distribution
    std::vector<int> all_rows(train.rows.size());  // Indices of all training rows
    for (size_t i = 0; i < train.rows.size(); ++i) {
        all_rows[i] = (int)i;
    }
    auto counts = class_counts_for(train, all_rows);  // Training class distribution
    default_class_ = argmax_counts(counts);   // Default prediction = best class

    std::vector<int> avail_attrs;  // List of all attributes initially available
    avail_attrs.reserve(train.spec.attrs.size());
    for (size_t i=0;i<train.spec.attrs.size();++i) avail_attrs.push_back((int)i);

    root_ = build(train, all_rows, avail_attrs, 0);  // Build tree starting at depth 0
}

int DecisionTree::predict_one(const DatasetSpec& spec, const Example& ex) const {
    (void)spec;
    const TreeNode* node = root_.get();  // Start traversal at root
    while (node && !node->is_leaf) {     // Continue until leaf reached
        const int a = node->attr_index;  // Attribute index used at this node
        if (!node->is_continuous_split) {  // Discrete decision
            const std::string& v = ex.x[a];  // Example's attribute value
            auto it = node->child_by_value.find(v);  // Find matching branch
            if (it == node->child_by_value.end()) return node->predicted_class; // unseen value fallback
            node = it->second.get();
        } else {
            const double x = util::to_double(ex.x[a]);
            node = (x <= node->threshold)     // Go left if <= threshold
                     ? node->left.get()
                     : node->right.get();
        }
    }
    if (!node) {
        return default_class_;
    }
    return node->predicted_class;
}


AccuracyReport DecisionTree::evaluate(const Dataset& ds) const {
    AccuracyReport r;   // Holds {correct, total}
    r.total = (int)ds.rows.size();   // Total number of examples evaluated
    for (auto& ex : ds.rows) {      // for each example in the dataset
        const int yp = predict_one(ds.spec, ex);   // Predict using tree traversal
        if (yp == ex.y) {         // Count correct predictions
            r.correct += 1;
        }
    }
    return r;   // Return accuracy report
}

// Convert class count vector like [3,1] into a printable string "(3,1)"
static std::string counts_str(const std::vector<int>& cc) {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < cc.size(); ++i) {
        oss << cc[i];
        if (i + 1 < cc.size()) {
            oss << ",";  // Comma between counts
        }
    }
    oss << ")";
    return oss.str();
}

void DecisionTree::print_tree(const DatasetSpec& spec) const {
    if (!root_) {   // If no tree has been trained
        std::cout << "(empty tree)\n";
        return;
    }

    // Print root “node label”
    const TreeNode* r = root_.get();
    if (r->is_leaf) {  // If tree is just a single leaf
        std::cout << "[LEAF] predict " << spec.class_labels[r->predicted_class]
                  << " " << counts_str(r->class_counts) << "\n";
        return;
    }

    const auto& a = spec.attrs[r->attr_index];   // Attribute used at the root split
    std::cout << "[ROOT] split on " << a.name;    // Print root split attribute
    if (r->is_continuous_split) {
        std::cout << " (continuous)";  // if continuous indicate so
    }
    std::cout << "\n";

    // Print children with connectors
    print_node(spec, r, "", true);
}

// Recursively prints a subtree in a readable “ASCII tree” format.
// `indent` carries the current indentation + vertical connector bars.
// `node` is expected to be non-leaf when called from print_tree().
void DecisionTree::print_node(const DatasetSpec& spec, const TreeNode* node,
                              const std::string& indent, bool /*is_root*/) const {
    const int aidx = node->attr_index;      // node is guaranteed non-leaf when called from print_tree()
    const auto& attr = spec.attrs[aidx];    // Attribute data

    if (!node->is_continuous_split) {
        // Build a sorted list of branch values
        std::vector<std::string> keys;
        keys.reserve(node->child_by_value.size());  // each possible branch value present
        for (auto& kv : node->child_by_value) {
            keys.push_back(kv.first);
        }
        std::sort(keys.begin(), keys.end());  // Sort for deterministic output

        for (size_t i = 0; i < keys.size(); ++i) {
            const bool last = (i + 1 == keys.size());  // True if this is the final printed child
            const std::string branch = last ? "└── " : "├── ";

            // if last child -> no vertical bar continues, else keep the vertical bar
            const std::string nextIndent = indent + (last ? "    " : "│   ");

            const std::string& val = keys[i];
            const TreeNode* child = node->child_by_value.at(val).get();  // Child subtree for this value

            // Print edge condition first
            std::cout << indent << branch << attr.name << " = " << val;

            // Then print what the child is
            if (child->is_leaf) {
                std::cout << "  =>  [LEAF] predict "
                          << spec.class_labels[child->predicted_class] << " "
                          << counts_str(child->class_counts) << "\n";
            } else {
                // Internal node: print what attribute the child splits on, then recurse
                const auto& childAttr = spec.attrs[child->attr_index];
                std::cout << "  ->  split on " << childAttr.name;
                if (child->is_continuous_split) {  // Tag if continuous
                    std::cout << " (continuous)";
                }
                std::cout << "\n";
                print_node(spec, child, nextIndent, false);  // Recurse into the child subtree
            }
        }
    } else {
        // continuous: two branches
        // Package the two branches in an array to print them uniformly
        struct Branch { 
            bool leq; 
            const TreeNode* child; 
        };
        Branch bs[2] = {
            { true,  node->left.get()  },   // Left branch: value <= threshold
            { false, node->right.get() }    // Right branch: value > threshold
        };

        for (int i = 0; i < 2; ++i) {
            const bool last = (i == 1);  // Second of two branches is printed last
            const std::string branch = last ? "└── " : "├── ";
            // if last child -> no vertical bar continues, else keep the vertical bar
            const std::string nextIndent = indent + (last ? "    " : "│   ");

            // Build the condition text for this continuous branch (<= or > threshold)
            std::ostringstream cond;
            cond << attr.name << (bs[i].leq ? " <= " : " > ") << node->threshold;

            const TreeNode* child = bs[i].child;  // Child subtree for this branch
            std::cout << indent << branch << cond.str();

            // Leaf: print predicted class + class counts
            if (child->is_leaf) {
                std::cout << "  =>  [LEAF] predict "
                          << spec.class_labels[child->predicted_class] << " "
                          << counts_str(child->class_counts) << "\n";
            } else {
                // Internal node: print next split attribute, then recurse
                const auto& childAttr = spec.attrs[child->attr_index];
                std::cout << "  ->  split on " << childAttr.name;
                if (child->is_continuous_split) {
                    std::cout << " (continuous)";
                }
                std::cout << "\n";
                print_node(spec, child, nextIndent, false);  // Recurse down
            }
        }
    }
}

// DFS traversal that converts each root-to-leaf path into a single rule.
// `path` = current list of conditions from root to this node.
// `out` collects completed rules.
void DecisionTree::extract_rules_rec(const DatasetSpec& spec, const TreeNode* node,
                                     std::vector<Condition>& path, std::vector<Rule>& out) const {
    // Base case: reached a leaf, so finalize one rule.
    if (node->is_leaf) {
        Rule r;
        r.conds = path;  // Copy the accumulated conditions
        r.predicted_class = node->predicted_class;  // Leaf prediction is the rule’s output class
        r.class_counts = node->class_counts;
        out.push_back(r);
        return;
    }
    const int a = node->attr_index;  // Attribute tested at this internal node
    if (!node->is_continuous_split) {
        // DISCRETE: iterate children in sorted order so rule list is deterministic
        std::vector<std::string> keys;
        keys.reserve(node->child_by_value.size());
        for (auto& kv : node->child_by_value) {
            keys.push_back(kv.first);
        }
        std::sort(keys.begin(), keys.end());
        for (auto& v : keys) {
            Condition c;
            c.attr_index = a;  // Which attribute index this condition tests
            c.is_cont = false;
            c.eq_value = v;  // Required value for a match
            path.push_back(c);  // Extend current path
            extract_rules_rec(spec, node->child_by_value.at(v).get(), path, out);  //recurse
            path.pop_back();  // Backtrack
        }
    } else {
        // CONTINUOUS: produce exactly two conditions (<= threshold) and (> threshold)

        // Left branch: attr <= threshold
        {
            Condition c;
            c.attr_index = a;
            c.is_cont = true;
            c.threshold = node->threshold;  // Threshold to compare against
            c.leq = true;                   // True means <= threshold
            path.push_back(c);
            extract_rules_rec(spec, node->left.get(), path, out);
            path.pop_back();
        }
        // Right branch: attr > threshold
        {
            Condition c;
            c.attr_index = a;
            c.is_cont = true;
            c.threshold = node->threshold;
            c.leq = false;              // False means > threshold
            path.push_back(c);
            extract_rules_rec(spec, node->right.get(), path, out);
            path.pop_back();
        }
    }
}

// Public wrapper: starts DFS at root and returns all extracted rules
std::vector<DecisionTree::Rule> DecisionTree::extract_rules(const DatasetSpec& spec) const {
    std::vector<Rule> rules;  // Output rule list
    std::vector<Condition> path;  // Current path conditions during traversal
    if (root_) {
        extract_rules_rec(spec, root_.get(), path, rules);
    }
    return rules;
}

// Returns true if the example satisfies ALL conditions in the rule
bool DecisionTree::rule_matches(const DatasetSpec& spec, const Example& ex, const Rule& r) const {
    (void)spec;
    for (const auto& c : r.conds) {  // Check each condition
        if (!c.is_cont) {  // Discrete: must match exact value
            if (ex.x[c.attr_index] != c.eq_value) {
                return false;
            }
        } else {  // Continuous: compare numeric value against threshold
            const double x = util::to_double(ex.x[c.attr_index]);
            if (c.leq) { 
                if (!(x <= c.threshold + EPS)) {  // Condition is "x <= threshold"
                    return false; 
                }
            } else {
                if (!(x >  c.threshold + EPS)) {   // Condition is "x > threshold"
                    return false; 
                }
            }
        }
    }
    return true;
}

// Predict using ordered rule list: return the FIRST rule that matches
int DecisionTree::predict_one_rules(const DatasetSpec& spec, const Example& ex,
                                    const std::vector<Rule>& rules, int default_class) const {
    for (const auto& r : rules) {
        if (rule_matches(spec, ex, r)) {  // First match wins
            return r.predicted_class;
        }
    }
    return default_class;  // Fallback if nothing matches
}

// Computes accuracy of a rule set over a dataset
AccuracyReport DecisionTree::evaluate_rules(const Dataset& ds, const std::vector<Rule>& rules, int default_class) const {
    AccuracyReport r;
    r.total = (int)ds.rows.size();  // Total examples
    for (auto& ex : ds.rows) {
        const int yp = predict_one_rules(ds.spec, ex, rules, default_class);  // Rule-based prediction
        if (yp == ex.y) {
            r.correct += 1;  // Count correct predictions
        }
    }
    return r;
}


// Reduced-error pruning:
// For each rule, try removing conditions if it DOES NOT reduce accuracy on the prune_set.
// This simplifies rules and can improve generalization (especially with noisy labels).
std::vector<DecisionTree::Rule> DecisionTree::post_prune_rules(const Dataset& prune_set,
                                                               const std::vector<Rule>& rules,
                                                               int default_class) const {
    std::vector<Rule> pruned = rules;

    // compute prune_set accuracy for a given rule list
    auto acc_of = [&](const std::vector<Rule>& rr)->double {
        return evaluate_rules(prune_set, rr, default_class).accuracy();
    };

    double base_acc = acc_of(pruned);  // Current best pruning-set accuracy

    for (size_t ri = 0; ri < pruned.size(); ++ri) {  // Process each rule in order
        bool improved_or_equal = true;  // Keep pruning this rule while we can keep accuracy >= baseline
        while (improved_or_equal && !pruned[ri].conds.empty()) {
            improved_or_equal = false;  // Assume no acceptable removal until we find one
            // Try removing each condition once; keep the best change (if not worse).
            double best_acc = base_acc;  // Best accuracy found for any one-condition removal
            int best_remove = -1;  // Index of condition to remove (if any)

            for (size_t ci = 0; ci < pruned[ri].conds.size(); ++ci) {
                auto trial = pruned;  // Copy entire rule list
                trial[ri].conds.erase(trial[ri].conds.begin() + (long)ci);  // Remove condition ci from rule ri
                double a = acc_of(trial);  // Evaluate new rule list on prune_set

                // Accept if accuracy is not worse (within EPS); keep the best such removal
                if (a + EPS >= best_acc) {
                    best_acc = a;
                    best_remove = (int)ci;
                }
            }
            if (best_remove >= 0) {
                // Apply the best removal permanently
                pruned[ri].conds.erase(pruned[ri].conds.begin() + best_remove);
                base_acc = best_acc;  // Update baseline accuracy after pruning
                improved_or_equal = true;  // Continue attempting to prune more conditions
            }
        }
    }
    return pruned;  // Return simplified rule set
}

// Prints each rule in a readable IF-condition format with predicted class and leaf counts
void DecisionTree::print_rules(const DatasetSpec& spec, const std::vector<Rule>& rules) {
    for (const auto& r : rules) {
        // No conditions means rule always matches
        if (r.conds.empty()) {
            std::cout << "(TRUE)";
        } else {
            // Print each condition joined by " ^ "
            for (size_t i = 0; i < r.conds.size(); ++i) {
                const auto& c = r.conds[i];  // Current condition
                const auto& an = spec.attrs[c.attr_index].name;  // Attribute name
                if (!c.is_cont) {
                    std::cout << an << " = " << c.eq_value;  // Discrete condition
                } else {
                    std::cout << an << (c.leq ? " <= " : " > ") << c.threshold;  // Continuous condition
                }
                if (i + 1 < r.conds.size()) std::cout << " ^ ";  // AND separator
            }
        }
        // Print the predicted class label and the class counts from the original leaf
        std::cout << " => " << spec.class_labels[r.predicted_class] << " (";
        for (size_t k = 0; k < r.class_counts.size(); ++k) {
            std::cout << r.class_counts[k];
            if (k + 1 < r.class_counts.size()) {
                std::cout << ",";
            }
        }
        std::cout << ")\n";
    }
}
