#pragma once
#include <vector>
#include <string>

struct AccuracyReport {
    int correct = 0;
    int total = 0;
    double accuracy() const { return total ? (double)correct / (double)total : 0.0; }
};

inline std::string fmt_pct(double a) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f%%", a*100.0);
    return std::string(buf);
}
