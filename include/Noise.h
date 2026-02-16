#pragma once
#include "Dataset.h"
#include <random>

inline void corrupt_labels(Dataset& ds, double percent, unsigned seed) {
    // percent in [0,100]
    if (percent <= 0.0) return;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 100.0);
    std::uniform_int_distribution<int> pick(0, (int)ds.spec.class_labels.size()-1);

    for (auto& ex : ds.rows) {
        if (u(rng) < percent) {
            int newy = ex.y;
            // change to a different class label
            while (newy == ex.y) newy = pick(rng);
            ex.y = newy;
        }
    }
}
