#include "Dataset.h"
#include "Util.h"
#include <stdexcept>
#include <random>


int DatasetSpec::class_index(const std::string& y) const {
    // Search for class label y in the class_labels vector and return its index
    for (size_t i = 0; i < class_labels.size(); ++i) {
        // If labels match exactly, return the index as int
        if (class_labels[i] == y) {
            return static_cast<int>(i);
        }
    }
    // Return -1 if label not found
    return -1;
}


DatasetSpec Dataset::load_spec(const std::string& attr_path) {
    DatasetSpec spec;  // Output spec object to fill
    auto lines = util::read_lines(attr_path);  // Read entire attr file into memory

    std::vector<std::vector<std::string>> toks;  // Tokenized non-empty lines
    for (auto& line : lines) {  // Loop through raw lines
        auto t = util::split_ws(line);  // Split line by whitespace into tokens
        if (t.empty()) {  // Skip blank or whitespace only lines
            continue;
        }
        toks.push_back(t);  // Save token list
    }
    // Need at least 1 attribute line + 1 class line
    if (toks.size() < 2) {
        throw std::runtime_error("Attr file seems too short: " + attr_path);
    }

    // Convention: last non-empty line defines the class labels.
    auto class_line = toks.back();  // Grab last tokenized line
    toks.pop_back();  // Remove it so the remaining lines are attributes

    // Must have "ClassName" + at least 1 label
    if (class_line.size() < 2) {
        throw std::runtime_error("Class line must have at least 2 tokens in: " + attr_path);
    }
    spec.class_name = class_line[0];  // First token is the class attribute name
    // Remaining tokens are class labels
    for (size_t i = 1; i < class_line.size(); ++i) {
        spec.class_labels.push_back(class_line[i]);
    }

    // Parse each remaining line as an attribute specification
    // Remaining lines define attributes.
    for (auto& t : toks) {
        if (t.size() < 2) {  // Skip malformed/short lines
            continue;
        }
        AttributeSpec a;  // Build an attribute spec
        a.name = t[0];  // First token is the attribute name
        if (util::ieq(t[1], "continuous")) {  // Case-insensitive check for "continuous"
            a.is_continuous = true;  // Mark as continuous numeric attribute
        } else {
            a.is_continuous = false;  // Otherwise it's discrete/categorical
            // Tokens after name are allowed values
            for (size_t i = 1; i < t.size(); ++i) {
                a.values.push_back(t[i]);
            }
        }
        spec.attrs.push_back(a);  // Add attribute definition to spec
    }
    if (spec.attrs.empty()) {
        throw std::runtime_error("No attributes parsed from: " + attr_path);
    }
    return spec;  // Return fully-parsed spec
}


Dataset Dataset::load_data(const DatasetSpec& spec, const std::string& data_path) {
    Dataset ds;  // Dataset object to fill
    ds.spec = spec;  // Copy spec so dataset knows attribute names/types/classes

    auto lines = util::read_lines(data_path);  // Read entire data file into memory
    for (auto& line_raw : lines) {
        auto line = util::trim(line_raw); // Trim leading/trailing whitespace
        if (line.empty()) {   // skip blank/empty lines
            continue;
        }
        auto t = util::split_ws(line);  // Split into tokens (attributes + class)

        // Validate expected token count: #attributes + 1 class label at end
        if (t.size() != spec.attrs.size() + 1) {
            throw std::runtime_error("Row has wrong #tokens in " + data_path +
                                     " expected " + std::to_string(spec.attrs.size()+1) +
                                     " got " + std::to_string(t.size()) + " line: " + line);
        }

        // One training/test instance
        Example ex;  
        // Store attribute values (all but last token) as strings
        ex.x.assign(t.begin(), t.begin() + static_cast<long>(spec.attrs.size()));
        const std::string ylab = t.back();  // Last token is the class label
        const int yi = spec.class_index(ylab);  // Convert class label string to its numeric index
        if (yi < 0) {
            throw std::runtime_error("Unknown class label '" + ylab + "' in " + data_path);
        }
        ex.y = yi;  // Save numeric class index
        ds.rows.push_back(ex);  // Add example to dataset
    }
    if (ds.rows.empty()) {
        throw std::runtime_error("No data loaded from: " + data_path);
    }
    return ds;  // Return the dataset
}

std::pair<Dataset, Dataset> Dataset::split_holdout(double holdout_frac, unsigned seed) const {
     // Validate holdout fraction is a proper probability in (0,1)
    if (holdout_frac <= 0.0 || holdout_frac >= 1.0) {
        throw std::runtime_error("holdout_frac must be in (0,1)");
    }
    Dataset a; a.spec = spec;   // 'a' will be the training subset (keeps same spec)
    Dataset b; b.spec = spec;   // 'b' will be the holdout/prune subset (keeps same spec)

    std::vector<size_t> idx(rows.size());
    // Fill index vector
    for (size_t i = 0; i < rows.size(); ++i) {
        idx[i] = i;
    }

    // Seeded RNG for reproducible shuffling
    std::mt19937 rng(seed);

    /*
    Shuffle indices before splitting so that train and holdout sets
    contain a representative mixture of classes. Without shuffling,
    ordered datasets could produce biased splits (e.g., only one class
    in the holdout set). A deterministic Fisher–Yates shuffle is used
    to ensure identical results across machines.
    */
    for (size_t i = idx.size(); i > 1; --i) {
        const uint64_t n = i;  // Current range size
        const uint64_t limit = (uint64_t(0x100000000ULL) / n) * n;  // Largest multiple of n below 2^32
        while (true) {
            uint32_t x = rng();  // Draw 32-bit random value from mt19937
            if (x < limit) {  // Reject values that would bias modulo
                size_t j = (size_t)(x % n);  // Map to [0, n-1] uniformly
                std::swap(idx[i - 1], idx[j]);  // Swap into final position (Fisher–Yates step)
                break;  // move to next i
            }
        }
    }

    // Compute number of holdout examples to allocate
    // Holdout is used for reduced-error post-pruning
    const size_t n_holdout = static_cast<size_t>(rows.size() * holdout_frac);
    // First n_holdout indices go to holdout set b
    // remainder go to training set a
    for (size_t k = 0; k < idx.size(); ++k) {
        if (k < n_holdout) {
            b.rows.push_back(rows[idx[k]]);
        } else {
            a.rows.push_back(rows[idx[k]]);
        }
    }

    // Ensure neither split is empty
    // (Possible with tiny datasets or extreme fractions)
    if (a.rows.empty() || b.rows.empty()) {
        // Fall back to a fixed pattern split (1/5th to holdout, rest to training)
        b.rows.clear(); 
        a.rows.clear();
        for (size_t k = 0; k < idx.size(); ++k) {
            if (k % 5 == 0) {
                b.rows.push_back(rows[idx[k]]);
            } else {
                a.rows.push_back(rows[idx[k]]);
            }
        }
    }
    return {a,b};  // Return pair: (train_subset, holdout_subset)
}
