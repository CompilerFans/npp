#include "benchmark_base.h"
#include <cstring>

namespace npp_benchmark {
    void runBenchmarks(const std::string& filter, const std::string& output, bool jsonOutput, bool verbose);
}

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n\n"
              << "Options:\n"
              << "  -h, --help            Show this help message\n"
              << "  -f, --filter PATTERN  Run only benchmarks matching PATTERN\n"
              << "  -o, --output FILE     Save results to CSV file\n"
              << "  -l, --list            List available benchmarks\n"
              << "  --json                Output results in JSON format (to stdout or to -o *.json)\n"
              << "  --verbose             Print per-benchmark progress\n"
              << "\n"
              << "Examples:\n"
              << "  " << progName << " -f Add              # Run Add benchmarks with built-in timing settings\n"
              << "  " << progName << " -o results.csv     # Save results to CSV\n";
}

void listBenchmarks() {
    std::cout << "Available benchmarks:\n";
    for (const auto& bench : npp_benchmark::BenchmarkRegistry::instance().benchmarks()) {
        std::cout << "  " << bench->name() << "\n";
    }
}

int main(int argc, char* argv[]) {
    std::string filter;
    std::string output;
    bool jsonOutput = false;
    bool verbose = false;
    bool listOnly = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--list") == 0) {
            listOnly = true;
        }
        else if ((strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--filter") == 0) && i + 1 < argc) {
            filter = argv[++i];
        }
        else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
            output = argv[++i];
        }
        else if (strcmp(argv[i], "--json") == 0) {
            jsonOutput = true;
        }
        else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        }
        else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (listOnly) {
        listBenchmarks();
        return 0;
    }

    if (!jsonOutput) {
        std::cout << "NPP Performance Benchmark\n";
        std::cout << "Warmup iterations: " << npp_benchmark::kBenchmarkWarmupDefault << "\n";
        std::cout << "Timed batch calls: " << npp_benchmark::kBenchmarkMeasureDefault << "\n";
        if (!filter.empty()) {
            std::cout << "Filter: " << filter << "\n";
        }
        std::cout << "\n";
    }

    npp_benchmark::runBenchmarks(filter, output, jsonOutput, verbose);

    return 0;
}
