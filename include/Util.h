#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace util {

// Removes leading and trailing whitespace from a string.
// Used when reading dataset lines to normalize formatting.
inline std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) {
        b++;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(s[e-1]))) {
        e--;
    }
    return s.substr(b, e-b);
}

// Removes leading and trailing whitespace from a string.
// Used when reading dataset lines to normalize formatting.
inline std::vector<std::string> split_ws(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) {
        out.push_back(tok);
    }
    return out;
}

// Reads an entire text file into a vector of strings (one per line).
// Throws an exception if the file cannot be opened.
inline std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream in(path.c_str());
    if (!in) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    return lines;
}

// Performs a case-insensitive comparison between two strings.
// Used when checking keywords such as "continuous" in attribute specs.
inline bool ieq(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i])))
            return false;
    }
    return true;
}

// Converts a string to a double with strict validation.
// Throws an exception if the string is not a valid numeric value.
inline double to_double(const std::string& s) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') {
        throw std::runtime_error("Expected numeric value, got: " + s);
    }
    return v;
}

}
