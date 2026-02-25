#pragma once
#include <string>
#include <vector>
#include <unordered_map>
// Dataset definitions and parsing utilities.
// Provides structures for attribute specifications, examples,
// and dataset loading/splitting used by the decision tree.



// Describes a single attribute in the dataset specification.
// Stores attribute name, whether it is continuous, and possible
// discrete values if applicable.
struct AttributeSpec {
    std::string name;
    bool is_continuous = false;
    std::vector<std::string> values; // for discrete
};

// Defines the schema of a dataset, including attribute metadata
// and class label information used during training and evaluation.
struct DatasetSpec {
    std::vector<AttributeSpec> attrs;
    std::string class_name;
    std::vector<std::string> class_labels;

    /*
    Looks up a class label string (like "Yes" or "Iris-setosa") in class_labels and returns its numeric index.
    */
    // Returns the integer index corresponding to a class label string.
    // Used when converting dataset text labels into numeric form.
    int class_index(const std::string& y) const;
};

// Represents one data instance consisting of feature values (as strings)
// and a numeric class label index.
struct Example {
    // raw tokens for discrete; for continuous, store string but parsed on demand
    std::vector<std::string> x;
    int y = -1; // class index
};

// Container for a full dataset, including specification metadata
// and all example rows. Provides utilities for loading and splitting data.
struct Dataset {
    DatasetSpec spec;
    std::vector<Example> rows;

    /*
    Parses the attribute/spec file (e.g., iris-attr.txt, tennis-attr.txt) and builds a DatasetSpec containing:
        1. the list of attributes (spec.attrs)
        2. whether each attribute is continuous or discrete
        3. allowed values for discrete attributes
        4. the class name and class label list
    */
    static DatasetSpec load_spec(const std::string& attr_path);

    /*
    Loads a dataset file (train/test) into a Dataset:
        1. each row becomes an Example
        2. stores feature values as strings in ex.x
        3. converts class label string into integer ex.y using spec.class_index()
    */
    static Dataset load_data(const DatasetSpec& spec, const std::string& data_path);

    /*
    Randomly (but deterministically) splits the current dataset into:
        1. a = training subset
        2. b = holdout/pruning subset
    It uses a seeded deterministic Fisher–Yates shuffle to ensure the exact same split happens across machines.
    The holdout set is used for reduced-error rule post-pruning
    */
    std::pair<Dataset, Dataset> split_holdout(double holdout_frac, unsigned seed) const;

    // Returns the number of attributes defined in the dataset specification.
    size_t n_attrs() const { 
        return spec.attrs.size();
    }
};
