#pragma once
#include "Dataset.h"
#include <random>
#include <vector>
#include <cmath>
// Noise utilities for controlled label corruption experiments.
// Provides deterministic random helpers to ensure identical results across machines.



// Generates a deterministic uniform random integer in the range [0, n-1]
// using mt19937 output and rejection sampling to avoid modulo bias.
// Designed so randomness is identical across different machines.
static inline size_t uniform_index(std::mt19937& rng, size_t n) {
    if (n == 0) return 0;
    const uint64_t limit = (uint64_t(0x100000000ULL) / n) * n; // 2^32
    while (true) {
        uint32_t x = rng();               // mt19937 gives 32-bit output
        if (x < limit) return x % n;      // unbiased
    }
}

// Corrupts a specified percentage of class labels in a dataset.
// Exactly round(percent * N) examples are selected via a deterministic shuffle,
// and each chosen label is changed to a different valid class.
// Used to simulate noisy training data for robustness experiments.
inline void corrupt_labels(Dataset& ds, double percent, unsigned seed) {
    if (percent <= 0.0) {
        return;
    }

    const size_t N = ds.rows.size();
    const size_t K = ds.spec.class_labels.size();
    if (N == 0 || K < 2) {
        return;
    }

    std::mt19937 rng(seed);

    // how many to flip (exact count, deterministic)
    size_t k = (size_t)std::llround((percent / 100.0) * (double)N);
    if (k > N) {
        k = N;
    }

    // deterministic shuffle of indices (Fisher–Yates using uniform_index)
    std::vector<size_t> idx(N);
    for (size_t i = 0; i < N; ++i) {
        idx[i] = i;
    }
    for (size_t i = N; i > 1; --i) {
        size_t j = uniform_index(rng, i);
        std::swap(idx[i - 1], idx[j]);
    }

    // flip first k
    for (size_t t = 0; t < k; ++t) {
        Example& ex = ds.rows[idx[t]];
        // pick a new class in [0, K-2], then "skip over" the old label
        size_t r = uniform_index(rng, K - 1);
        int newy = (int)r;
        if (newy >= ex.y) newy += 1;
        ex.y = newy;
    }
}
