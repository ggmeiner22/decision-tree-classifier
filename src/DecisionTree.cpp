#include "DecisionTree.h"
#include "Util.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <map>

static const double EPS = 1e-12;

double DecisionTree::entropy_counts(const std::vector<int>& counts) const {
    double sum = 0.0;
    for (int c : counts) sum += c;
    if (sum <= 0.0) return 0.0;

    double H = 0.0;
    for (int c : counts) {
        if (c <= 0) continue;
        double p = (double)c / sum;
        H -= p * std::log(p) / std::log(2.0);
    }
    return H;
}

int DecisionTree::argmax_counts(const std::vector<int>& counts) const {
    int best_i = 0;
    int best_v = counts.empty() ? 0 : counts[0];
    for (size_t i=1;i<counts.size();++i) {
        if (counts[i] > best_v) { best_v = counts[i]; best_i = (int)i; }
    }
    return best_i;
}

std::vector<int> DecisionTree::class_counts_for(const Dataset& ds, const std::vector<int>& rows) const {
    std::vector<int> counts(ds.spec.class_labels.size(), 0);
    for (int rid : rows) counts[ds.rows[rid].y] += 1;
    return counts;
}

DecisionTree::BestSplit DecisionTree::choose_best_split(const Dataset& ds, const std::vector<int>& rows,
                                                        const std::vector<int>& avail_attrs) const {
    BestSplit best;
    const auto parent_counts = class_counts_for(ds, rows);
    const double parent_H = entropy_counts(parent_counts);
    const double parent_n = (double)rows.size();

    for (int aidx : avail_attrs) {
        const auto& attr = ds.spec.attrs[aidx];

        if (!attr.is_continuous) {
            // multiway split by discrete value
            std::unordered_map<std::string, std::vector<int>> parts;
            parts.reserve(attr.values.size() + 2);
            for (int rid : rows) {
                const std::string& v = ds.rows[rid].x[aidx];
                parts[v].push_back(rid);
            }
            // information gain
            double child_H = 0.0;
            for (auto& kv : parts) {
                const auto& part_rows = kv.second;
                const auto cc = class_counts_for(ds, part_rows);
                const double w = (double)part_rows.size() / parent_n;
                child_H += w * entropy_counts(cc);
            }

            const double gain = parent_H - child_H;
            const int branches = (int)parts.size(); // if I want simplest attribute on tie
            const int best_branches = best.is_cont ? 2 : (int)best.parts_disc.size(); // if I want simplest attribute on tie
            if (gain > best.gain + EPS ||
                (std::fabs(gain - best.gain) <= EPS && branches < best_branches) || // if I want simplest attribute on tie
                (std::fabs(gain - best.gain) <= EPS && branches == best_branches && aidx < best.attr)) { // if I want simplest attribute on tie
                best.gain = gain;
                best.attr = aidx;
                best.is_cont = false;
                best.parts_disc = std::move(parts);
                best.left_rows.clear();
                best.right_rows.clear();
            }
        } else {
            // continuous: choose threshold that maximizes gain (binary split)
            std::vector<std::pair<double,int>> vals; // (x, rid)
            vals.reserve(rows.size());
            for (int rid : rows) {
                const std::string& s = ds.rows[rid].x[aidx];
                const double x = util::to_double(s);
                vals.push_back({x, rid});
            }
            std::sort(vals.begin(), vals.end(),
                      [](const std::pair<double,int>& p1, const std::pair<double,int>& p2){ return p1.first < p2.first; });
            if (vals.size() < 2) continue;

            // precompute prefix class counts
            const int K = (int)ds.spec.class_labels.size();
            std::vector<std::vector<int>> prefix(vals.size()+1, std::vector<int>(K,0));
            for (size_t i=0;i<vals.size();++i) {
                prefix[i+1] = prefix[i];
                prefix[i+1][ ds.rows[vals[i].second].y ] += 1;
            }
            const auto total = prefix.back();

            double best_gain_a = -1e9;
            double best_thr = 0.0;
            size_t best_cut = 0;

            for (size_t i=0;i+1<vals.size();++i) {
                const double x1 = vals[i].first;
                const double x2 = vals[i+1].first;
                if (std::fabs(x2 - x1) < EPS) continue; // no midpoint
                const double thr = 0.5*(x1+x2);

                std::vector<int> left_counts = prefix[i+1];
                std::vector<int> right_counts(K,0);
                for (int k=0;k<K;++k) right_counts[k] = total[k] - left_counts[k];

                const double nL = (double)(i+1);
                const double nR = (double)(vals.size()-(i+1));
                const double child_H = (nL/parent_n)*entropy_counts(left_counts) + (nR/parent_n)*entropy_counts(right_counts);
                const double gain = parent_H - child_H;

                if (gain > best_gain_a + EPS) {
                    best_gain_a = gain;
                    best_thr = thr;
                    best_cut = i+1;
                }
            }

            const int branches = 2;
            const int best_branches = best.is_cont ? 2 : (int)best.parts_disc.size();

            if (best_gain_a > best.gain + EPS ||
                (std::fabs(best_gain_a - best.gain) <= EPS && branches < best_branches) ||
                (std::fabs(best_gain_a - best.gain) <= EPS && branches == best_branches && aidx < best.attr)) {
                best.gain = best_gain_a;
                best.attr = aidx;
                best.is_cont = true;
                best.threshold = best_thr;
                best.parts_disc.clear();
                best.left_rows.clear();
                best.right_rows.clear();
                for (size_t i=0;i<vals.size();++i) {
                    if (i < best_cut) best.left_rows.push_back(vals[i].second);
                    else best.right_rows.push_back(vals[i].second);
                }
            }
        }
    }
    return best;
}

std::unique_ptr<TreeNode> DecisionTree::build(const Dataset& ds, const std::vector<int>& rows,
                                              const std::vector<int>& avail_attrs, int depth) {
    auto node = std::unique_ptr<TreeNode>(new TreeNode());
    node->class_counts = class_counts_for(ds, rows);
    node->predicted_class = argmax_counts(node->class_counts);

    // stopping criteria
    const int majority = node->predicted_class;
    const int maj_count = node->class_counts[majority];
    if ((int)rows.size() < params_.min_samples_split ||
        depth >= params_.max_depth ||
        avail_attrs.empty() ||
        maj_count == (int)rows.size()) {
        node->is_leaf = true;
        return node;
    }

    BestSplit split = choose_best_split(ds, rows, avail_attrs);
    if (split.attr < 0 || split.gain <= EPS) {
        node->is_leaf = true;
        return node;
    }

    node->attr_index = split.attr;
    node->is_continuous_split = split.is_cont;
    node->threshold = split.threshold;

    // next avail attrs: for discrete, you can reuse attrs if you want (C4.5 style), but assignment doesn't require.
    // We'll allow reusing continuous attrs as well; for discrete, remove to avoid cycles.
    std::vector<int> next_avail;
    next_avail.reserve(avail_attrs.size());
    for (int a : avail_attrs) {
        if (a == split.attr && !ds.spec.attrs[a].is_continuous) continue;
        next_avail.push_back(a);
    }

    if (!split.is_cont) {
        for (auto& kv : split.parts_disc) {
            const std::string& val = kv.first;
            auto& part_rows = kv.second;
            node->child_by_value[val] = build(ds, part_rows, next_avail, depth+1);
        }
        node->is_leaf = false;
    } else {
        if (split.left_rows.empty() || split.right_rows.empty()) {
            node->is_leaf = true;
            return node;
        }
        node->left = build(ds, split.left_rows, next_avail, depth+1);
        node->right = build(ds, split.right_rows, next_avail, depth+1);
        node->is_leaf = false;
    }
    return node;
}

void DecisionTree::fit(const Dataset& train) {
    // compute default class from training distribution
    std::vector<int> all_rows(train.rows.size());
    for (size_t i=0;i<train.rows.size();++i) all_rows[i] = (int)i;
    auto counts = class_counts_for(train, all_rows);
    default_class_ = argmax_counts(counts);

    std::vector<int> avail_attrs;
    avail_attrs.reserve(train.spec.attrs.size());
    for (size_t i=0;i<train.spec.attrs.size();++i) avail_attrs.push_back((int)i);

    root_ = build(train, all_rows, avail_attrs, 0);
}

int DecisionTree::predict_one(const DatasetSpec& spec, const Example& ex) const {
    (void)spec;
    const TreeNode* node = root_.get();
    while (node && !node->is_leaf) {
        const int a = node->attr_index;
        if (!node->is_continuous_split) {
            const std::string& v = ex.x[a];
            auto it = node->child_by_value.find(v);
            if (it == node->child_by_value.end()) return node->predicted_class; // unseen value fallback
            node = it->second.get();
        } else {
            const double x = util::to_double(ex.x[a]);
            node = (x <= node->threshold) ? node->left.get() : node->right.get();
        }
    }
    if (!node) return default_class_;
    return node->predicted_class;
}

AccuracyReport DecisionTree::evaluate(const Dataset& ds) const {
    AccuracyReport r;
    r.total = (int)ds.rows.size();
    for (auto& ex : ds.rows) {
        const int yp = predict_one(ds.spec, ex);
        if (yp == ex.y) r.correct += 1;
    }
    return r;
}

// Replace BOTH print_tree() and print_node() with these

static std::string counts_str(const std::vector<int>& cc) {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < cc.size(); ++i) {
        oss << cc[i];
        if (i + 1 < cc.size()) oss << ",";
    }
    oss << ")";
    return oss.str();
}

void DecisionTree::print_tree(const DatasetSpec& spec) const {
    if (!root_) {
        std::cout << "(empty tree)\n";
        return;
    }

    // Print root “node label”
    const TreeNode* r = root_.get();
    if (r->is_leaf) {
        std::cout << "[LEAF] predict " << spec.class_labels[r->predicted_class]
                  << " " << counts_str(r->class_counts) << "\n";
        return;
    }

    const auto& a = spec.attrs[r->attr_index];
    std::cout << "[ROOT] split on " << a.name;
    if (r->is_continuous_split) std::cout << " (continuous)";
    std::cout << "\n";

    // Print children with connectors
    print_node(spec, r, "", true);
}

void DecisionTree::print_node(const DatasetSpec& spec, const TreeNode* node,
                              const std::string& indent, bool /*is_root*/) const {
    // node is guaranteed non-leaf when called from print_tree()

    const int aidx = node->attr_index;
    const auto& attr = spec.attrs[aidx];

    if (!node->is_continuous_split) {
        // stable order for printing
        std::vector<std::string> keys;
        keys.reserve(node->child_by_value.size());
        for (auto& kv : node->child_by_value) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());

        for (size_t i = 0; i < keys.size(); ++i) {
            const bool last = (i + 1 == keys.size());
            const std::string branch = last ? "└── " : "├── ";
            const std::string nextIndent = indent + (last ? "    " : "│   ");

            const std::string& val = keys[i];
            const TreeNode* child = node->child_by_value.at(val).get();

            // Print edge condition first
            std::cout << indent << branch << attr.name << " = " << val;

            // Then print what the child is
            if (child->is_leaf) {
                std::cout << "  =>  [LEAF] predict "
                          << spec.class_labels[child->predicted_class] << " "
                          << counts_str(child->class_counts) << "\n";
            } else {
                const auto& childAttr = spec.attrs[child->attr_index];
                std::cout << "  ->  split on " << childAttr.name;
                if (child->is_continuous_split) std::cout << " (continuous)";
                std::cout << "\n";
                print_node(spec, child, nextIndent, false);
            }
        }
    } else {
        // continuous: two branches
        struct Branch { bool leq; const TreeNode* child; };
        Branch bs[2] = {
            { true,  node->left.get()  },
            { false, node->right.get() }
        };

        for (int i = 0; i < 2; ++i) {
            const bool last = (i == 1);
            const std::string branch = last ? "└── " : "├── ";
            const std::string nextIndent = indent + (last ? "    " : "│   ");

            std::ostringstream cond;
            cond << attr.name << (bs[i].leq ? " <= " : " > ") << node->threshold;

            const TreeNode* child = bs[i].child;
            std::cout << indent << branch << cond.str();

            if (child->is_leaf) {
                std::cout << "  =>  [LEAF] predict "
                          << spec.class_labels[child->predicted_class] << " "
                          << counts_str(child->class_counts) << "\n";
            } else {
                const auto& childAttr = spec.attrs[child->attr_index];
                std::cout << "  ->  split on " << childAttr.name;
                if (child->is_continuous_split) std::cout << " (continuous)";
                std::cout << "\n";
                print_node(spec, child, nextIndent, false);
            }
        }
    }
}

void DecisionTree::extract_rules_rec(const DatasetSpec& spec, const TreeNode* node,
                                     std::vector<Condition>& path, std::vector<Rule>& out) const {
    if (node->is_leaf) {
        Rule r;
        r.conds = path;
        r.predicted_class = node->predicted_class;
        r.class_counts = node->class_counts;
        out.push_back(r);
        return;
    }
    const int a = node->attr_index;
    if (!node->is_continuous_split) {
        // stable order
        std::vector<std::string> keys;
        keys.reserve(node->child_by_value.size());
        for (auto& kv : node->child_by_value) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        for (auto& v : keys) {
            Condition c;
            c.attr_index = a;
            c.is_cont = false;
            c.eq_value = v;
            path.push_back(c);
            extract_rules_rec(spec, node->child_by_value.at(v).get(), path, out);
            path.pop_back();
        }
    } else {
        // left (<=)
        {
            Condition c;
            c.attr_index = a;
            c.is_cont = true;
            c.threshold = node->threshold;
            c.leq = true;
            path.push_back(c);
            extract_rules_rec(spec, node->left.get(), path, out);
            path.pop_back();
        }
        // right (>)
        {
            Condition c;
            c.attr_index = a;
            c.is_cont = true;
            c.threshold = node->threshold;
            c.leq = false;
            path.push_back(c);
            extract_rules_rec(spec, node->right.get(), path, out);
            path.pop_back();
        }
    }
}

std::vector<DecisionTree::Rule> DecisionTree::extract_rules(const DatasetSpec& spec) const {
    std::vector<Rule> rules;
    std::vector<Condition> path;
    if (root_) extract_rules_rec(spec, root_.get(), path, rules);
    return rules;
}

bool DecisionTree::rule_matches(const DatasetSpec& spec, const Example& ex, const Rule& r) const {
    (void)spec;
    for (const auto& c : r.conds) {
        if (!c.is_cont) {
            if (ex.x[c.attr_index] != c.eq_value) return false;
        } else {
            const double x = util::to_double(ex.x[c.attr_index]);
            if (c.leq) { if (!(x <= c.threshold + EPS)) return false; }
            else       { if (!(x >  c.threshold + EPS)) return false; }
        }
    }
    return true;
}

int DecisionTree::predict_one_rules(const DatasetSpec& spec, const Example& ex,
                                    const std::vector<Rule>& rules, int default_class) const {
    for (const auto& r : rules) {
        if (rule_matches(spec, ex, r)) return r.predicted_class;
    }
    return default_class;
}

AccuracyReport DecisionTree::evaluate_rules(const Dataset& ds, const std::vector<Rule>& rules, int default_class) const {
    AccuracyReport r;
    r.total = (int)ds.rows.size();
    for (auto& ex : ds.rows) {
        const int yp = predict_one_rules(ds.spec, ex, rules, default_class);
        if (yp == ex.y) r.correct += 1;
    }
    return r;
}

std::vector<DecisionTree::Rule> DecisionTree::post_prune_rules(const Dataset& prune_set,
                                                               const std::vector<Rule>& rules,
                                                               int default_class) const {
    // Reduced-error pruning: for each rule, attempt to remove conditions that don't reduce accuracy on prune_set.
    // Order: rules are applied in sequence; we preserve order.
    std::vector<Rule> pruned = rules;

    auto acc_of = [&](const std::vector<Rule>& rr)->double {
        return evaluate_rules(prune_set, rr, default_class).accuracy();
    };

    double base_acc = acc_of(pruned);

    for (size_t ri=0; ri<pruned.size(); ++ri) {
        bool improved_or_equal = true;
        while (improved_or_equal && !pruned[ri].conds.empty()) {
            improved_or_equal = false;
            // Try removing each condition once; keep the best change (if not worse).
            double best_acc = base_acc;
            int best_remove = -1;

            for (size_t ci=0; ci<pruned[ri].conds.size(); ++ci) {
                auto trial = pruned;
                trial[ri].conds.erase(trial[ri].conds.begin() + (long)ci);
                double a = acc_of(trial);
                if (a + EPS >= best_acc) {
                    best_acc = a;
                    best_remove = (int)ci;
                }
            }
            if (best_remove >= 0) {
                pruned[ri].conds.erase(pruned[ri].conds.begin() + best_remove);
                base_acc = best_acc;
                improved_or_equal = true;
            }
        }
    }
    return pruned;
}

void DecisionTree::print_rules(const DatasetSpec& spec, const std::vector<Rule>& rules) {
    for (const auto& r : rules) {
        if (r.conds.empty()) {
            std::cout << "(TRUE)";
        } else {
            for (size_t i=0;i<r.conds.size();++i) {
                const auto& c = r.conds[i];
                const auto& an = spec.attrs[c.attr_index].name;
                if (!c.is_cont) {
                    std::cout << an << " = " << c.eq_value;
                } else {
                    std::cout << an << (c.leq ? " <= " : " > ") << c.threshold;
                }
                if (i+1<r.conds.size()) std::cout << " ^ ";
            }
        }
        std::cout << " => " << spec.class_labels[r.predicted_class] << " (";
        for (size_t k=0;k<r.class_counts.size();++k) {
            std::cout << r.class_counts[k];
            if (k+1<r.class_counts.size()) std::cout << ",";
        }
        std::cout << ")\n";
    }
}
