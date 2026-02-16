#pragma once
#include <string>
#include <vector>
#include <unordered_map>

struct AttributeSpec {
    std::string name;
    bool is_continuous = false;
    std::vector<std::string> values; // for discrete
};

struct DatasetSpec {
    std::vector<AttributeSpec> attrs;
    std::string class_name;
    std::vector<std::string> class_labels;

    int class_index(const std::string& y) const;
};

struct Example {
    // raw tokens for discrete; for continuous, store string but parsed on demand
    std::vector<std::string> x;
    int y = -1; // class index
};

struct Dataset {
    DatasetSpec spec;
    std::vector<Example> rows;

    static DatasetSpec load_spec(const std::string& attr_path);
    static Dataset load_data(const DatasetSpec& spec, const std::string& data_path);

    // Utility: split rows into train/prune (holdout fraction)
    std::pair<Dataset, Dataset> split_holdout(double holdout_frac, unsigned seed) const;

    size_t n_attrs() const { return spec.attrs.size(); }
};
