#pragma once
#include <vector>
#include <string>
// Evaluation utilities for reporting classification accuracy
// and formatting results for display.


// Stores classification accuracy statistics.
// Tracks number of correct predictions and total examples,
// and provides a helper function to compute accuracy as a ratio.
struct AccuracyReport {
    int correct = 0;
    int total = 0;

    // Returns prediction accuracy as correct / total.
    // Safely returns 0.0 if total is zero to avoid division by zero.
    double accuracy() const { 
        return total ? (double)correct / (double)total : 0.0; 
    }
};

// Formats an accuracy value as a percentage string with two decimal places.
// Example: 0.94 -> "94.00%". Used for clean console output.
inline std::string fmt_pct(double a) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f%%", a*100.0);
    return std::string(buf);
}
