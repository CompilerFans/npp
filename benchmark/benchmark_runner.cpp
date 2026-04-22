#include "benchmark_base.h"
#include <fstream>
#include <sstream>
#include <ctime>

namespace npp_benchmark {
namespace {

bool endsWith(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

std::string jsonEscape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (char c : input) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

void writeJson(std::ostream& os, const DeviceInfo& device, const std::vector<BenchmarkResult>& results,
               const std::string& filter) {
    int passed = 0;
    int failed = 0;
    for (const auto& r : results) {
        if (r.success)
            passed++;
        else
            failed++;
    }

    os << "{";
    os << "\"device\":{";
    os << "\"name\":\"" << jsonEscape(device.name) << "\",";
    os << "\"compute_capability\":\"" << device.major << "." << device.minor << "\",";
    os << "\"global_memory_bytes\":" << device.globalMemBytes << ",";
    os << "\"sm_count\":" << device.smCount << ",";
    os << "\"clock_rate_khz\":" << device.clockRateKHz << ",";
    os << "\"memory_clock_rate_khz\":" << device.memoryClockRateKHz << ",";
    os << "\"memory_bus_width_bits\":" << device.memoryBusWidthBits;
    os << "},";

    os << "\"config\":{";
    os << "\"warmup_iterations\":" << kBenchmarkWarmupDefault << ",";
    os << "\"measure_iterations\":" << kBenchmarkMeasureDefault << ",";
    os << "\"filter\":\"" << jsonEscape(filter) << "\"";
    os << "},";

    os << "\"results\":[";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        os << "{";
        os << "\"function_name\":\"" << jsonEscape(r.functionName) << "\",";
        os << "\"size\":\"" << jsonEscape(r.size) << "\",";
        os << "\"variant_tags\":\"" << jsonEscape(r.variantTags) << "\",";
        os << "\"data_type\":\"" << jsonEscape(r.dataType) << "\",";
        os << "\"channels\":" << r.channels << ",";
        os << "\"avg_time_ms\":" << r.avgTimeMs << ",";
        os << "\"throughput_gbps\":" << r.throughputGBps << ",";
        os << "\"success\":" << (r.success ? "true" : "false") << ",";
        os << "\"error\":\"" << jsonEscape(r.errorMessage) << "\"";
        os << "}";
        if (i + 1 < results.size()) os << ",";
    }
    os << "],";

    os << "\"summary\":{";
    os << "\"total\":" << results.size() << ",";
    os << "\"passed\":" << passed << ",";
    os << "\"failed\":" << failed;
    os << "}";
    os << "}\n";
}

} // namespace

class BenchmarkRunner {
public:
    void setFilter(const std::string& filter) { filter_ = filter; }
    void setOutputFile(const std::string& file) { outputFile_ = file; }
    void setJsonOutput(bool enabled) { jsonOutput_ = enabled; }
    void setVerbose(bool enabled) { verbose_ = enabled; }

    void run() {
        const DeviceInfo device = getDeviceInfo();
        if (!jsonOutput_) {
            printDeviceInfo();
            std::cout << std::endl;
        }

        std::vector<BenchmarkResult> results;

        if (filter_.empty()) {
            results = BenchmarkRegistry::instance().runAll(kBenchmarkWarmupDefault, kBenchmarkMeasureDefault, verbose_);
        } else {
            results = BenchmarkRegistry::instance().runFiltered(filter_, kBenchmarkWarmupDefault, kBenchmarkMeasureDefault, verbose_);
        }

        if (jsonOutput_) {
            if (!outputFile_.empty() && endsWith(outputFile_, ".json")) {
                std::ofstream file(outputFile_);
                if (!file.is_open()) {
                    std::cerr << "Failed to open output file: " << outputFile_ << std::endl;
                    return;
                }
                writeJson(file, device, results, filter_);
                file.close();
            } else {
                writeJson(std::cout, device, results, filter_);
            }

            if (!outputFile_.empty() && endsWith(outputFile_, ".csv")) {
                saveResultsCSV(results, outputFile_);
            }
            return;
        }

        printResults(results);

        if (!outputFile_.empty()) {
            saveResultsCSV(results, outputFile_);
        }

        printSummary(results);
    }

private:
    std::string filter_;
    std::string outputFile_;
    bool jsonOutput_ = false;
    bool verbose_ = false;

    void printResults(const std::vector<BenchmarkResult>& results) {
        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Benchmark Results" << std::endl;
        std::cout << "========================================" << std::endl;

        std::cout << std::left
                  << std::setw(40) << "Function"
                  << std::setw(18) << "Size"
                  << std::setw(16) << "Variant"
                  << std::setw(8) << "Type"
                  << std::setw(4) << "Ch"
                  << std::setw(15) << "Avg Time"
                  << std::setw(15) << "Throughput"
                  << std::endl;
        std::cout << std::string(116, '-') << std::endl;

        for (const auto& r : results) {
            r.print();
        }
    }

    void saveResultsCSV(const std::vector<BenchmarkResult>& results, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file: " << filename << std::endl;
            return;
        }

        file << "function_name,size,variant_tags,data_type,channels,avg_time_ms,throughput_gbps,success,error\n";

        for (const auto& r : results) {
            file << r.functionName << ","
                 << r.size << ","
                 << "\"" << r.variantTags << "\","
                 << r.dataType << ","
                 << r.channels << ","
                 << r.avgTimeMs << ","
                 << r.throughputGBps << ","
                 << (r.success ? "true" : "false") << ","
                 << "\"" << r.errorMessage << "\"\n";
        }

        file.close();
        std::cout << "\nResults saved to: " << filename << std::endl;
    }

    void printSummary(const std::vector<BenchmarkResult>& results) {
        int total = results.size();
        int passed = 0;
        int failed = 0;

        for (const auto& r : results) {
            if (r.success) passed++;
            else failed++;
        }

        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Summary" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total benchmarks: " << total << std::endl;
        std::cout << "Passed: " << passed << std::endl;
        std::cout << "Failed: " << failed << std::endl;
    }
};

// Global runner instance
static BenchmarkRunner g_runner;

void runBenchmarks(const std::string& filter, const std::string& output, bool jsonOutput, bool verbose) {
    g_runner.setFilter(filter);
    g_runner.setOutputFile(output);
    g_runner.setJsonOutput(jsonOutput);
    g_runner.setVerbose(verbose);
    g_runner.run();
}

} // namespace npp_benchmark
