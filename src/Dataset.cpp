#include "Dataset.h"
#include "Util.h"
#include <stdexcept>
#include <random>

int DatasetSpec::class_index(const std::string& y) const {
    for (size_t i=0;i<class_labels.size();++i) {
        if (class_labels[i] == y) return static_cast<int>(i);
    }
    return -1;
}

DatasetSpec Dataset::load_spec(const std::string& attr_path) {
    DatasetSpec spec;
    auto lines = util::read_lines(attr_path);

    std::vector<std::vector<std::string>> toks;
    for (auto& line : lines) {
        auto t = util::split_ws(line);
        if (t.empty()) continue;
        toks.push_back(t);
    }
    if (toks.size() < 2) {
        throw std::runtime_error("Attr file seems too short: " + attr_path);
    }

    // Convention (based on provided files): last non-empty line defines the class labels.
    // Examples:
    //   PlayTennis Yes No
    //   Iris Iris-setosa Iris-versicolor Iris-virginica
    //   Class Yes No
    auto class_line = toks.back();
    toks.pop_back();

    if (class_line.size() < 2) {
        throw std::runtime_error("Class line must have at least 2 tokens in: " + attr_path);
    }
    spec.class_name = class_line[0];
    for (size_t i=1;i<class_line.size();++i) spec.class_labels.push_back(class_line[i]);

    // Remaining lines define attributes.
    for (auto& t : toks) {
        if (t.size() < 2) continue;
        AttributeSpec a;
        a.name = t[0];
        if (util::ieq(t[1], "continuous")) {
            a.is_continuous = true;
        } else {
            a.is_continuous = false;
            for (size_t i=1;i<t.size();++i) a.values.push_back(t[i]);
        }
        spec.attrs.push_back(a);
    }
    if (spec.attrs.empty()) throw std::runtime_error("No attributes parsed from: " + attr_path);
    return spec;
}

Dataset Dataset::load_data(const DatasetSpec& spec, const std::string& data_path) {
    Dataset ds;
    ds.spec = spec;

    auto lines = util::read_lines(data_path);
    for (auto& line_raw : lines) {
        auto line = util::trim(line_raw);
        if (line.empty()) continue;
        auto t = util::split_ws(line);
        if (t.size() != spec.attrs.size() + 1) {
            throw std::runtime_error("Row has wrong #tokens in " + data_path +
                                     " expected " + std::to_string(spec.attrs.size()+1) +
                                     " got " + std::to_string(t.size()) + " line: " + line);
        }
        Example ex;
        ex.x.assign(t.begin(), t.begin() + static_cast<long>(spec.attrs.size()));
        const std::string ylab = t.back();
        const int yi = spec.class_index(ylab);
        if (yi < 0) throw std::runtime_error("Unknown class label '" + ylab + "' in " + data_path);
        ex.y = yi;
        ds.rows.push_back(ex);
    }
    if (ds.rows.empty()) throw std::runtime_error("No data loaded from: " + data_path);
    return ds;
}

std::pair<Dataset, Dataset> Dataset::split_holdout(double holdout_frac, unsigned seed) const {
    if (holdout_frac <= 0.0 || holdout_frac >= 1.0) {
        throw std::runtime_error("holdout_frac must be in (0,1)");
    }
    Dataset a; a.spec = spec;
    Dataset b; b.spec = spec;

    std::vector<size_t> idx(rows.size());
    for (size_t i=0;i<rows.size();++i) idx[i]=i;

    std::mt19937 rng(seed);

    // deterministic Fisher–Yates shuffle
    for (size_t i = idx.size(); i > 1; --i) {
        // generate j in [0, i-1] deterministically (no std::uniform_int_distribution)
        const uint64_t n = i;
        const uint64_t limit = (uint64_t(0x100000000ULL) / n) * n;
        while (true) {
            uint32_t x = rng();
            if (x < limit) {
                size_t j = (size_t)(x % n);
                std::swap(idx[i - 1], idx[j]);
                break;
            }
        }
    }

    const size_t n_holdout = static_cast<size_t>(rows.size() * holdout_frac);
    for (size_t k=0;k<idx.size();++k) {
        if (k < n_holdout) b.rows.push_back(rows[idx[k]]);
        else a.rows.push_back(rows[idx[k]]);
    }
    if (a.rows.empty() || b.rows.empty()) {
        // fall back: ensure at least one row each
        b.rows.clear(); a.rows.clear();
        for (size_t k=0;k<idx.size();++k) {
            if (k % 5 == 0) b.rows.push_back(rows[idx[k]]);
            else a.rows.push_back(rows[idx[k]]);
        }
    }
    return {a,b};
}
