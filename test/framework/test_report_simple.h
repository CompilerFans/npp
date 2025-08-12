#ifndef TEST_REPORT_SIMPLE_H
#define TEST_REPORT_SIMPLE_H

#include "npp_test_framework.h"
#include <fstream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <vector>

/**
 * @brief Simple test report generator (no JSON dependency)
 */
class TestReport {
private:
    struct TestData {
        std::string function;
        std::string parameters;
        bool passed;
        double max_error;
        double cpu_time_ms;
        double gpu_time_ms;
        double nvidia_time_ms;
        std::string error_message;
    };
    
    std::string report_name_;
    std::string output_dir_;
    std::ofstream html_file_;
    std::ofstream csv_file_;
    std::vector<TestData> test_data_;
    
public:
    TestReport(const std::string& report_name, const std::string& output_dir = "test_reports")
        : report_name_(report_name), output_dir_(output_dir) {
        
        // Create output directory
        std::string mkdir_cmd = "mkdir -p " + output_dir_;
        system(mkdir_cmd.c_str());
    }
    
    ~TestReport() {
        if (html_file_.is_open()) {
            html_file_.close();
        }
        if (csv_file_.is_open()) {
            csv_file_.close();
        }
    }
    
    void startReport() {
        // Start HTML report
        std::string html_filename = output_dir_ + "/" + report_name_ + ".html";
        html_file_.open(html_filename);
        
        if (html_file_.is_open()) {
            writeHTMLHeader();
        }
        
        // Start CSV report
        std::string csv_filename = output_dir_ + "/" + report_name_ + ".csv";
        csv_file_.open(csv_filename);
        if (csv_file_.is_open()) {
            csv_file_ << "Function,Parameters,Status,MaxError,CPUTime_ms,GPUTime_ms,NVIDIATime_ms,GPUSpeedup,NVIDIASpeedup,ErrorMessage\n";
        }
    }
    
    void addTestResult(const std::string& function_name, 
                      const NPPTest::TestParameters& params,
                      const NPPTest::TestResult& result) {
        
        TestData data;
        data.function = function_name;
        data.parameters = params.toString();
        data.passed = result.passed;
        data.max_error = result.max_error;
        data.cpu_time_ms = result.cpu_time_ms;
        data.gpu_time_ms = result.gpu_time_ms;
        data.nvidia_time_ms = result.nvidia_time_ms;
        data.error_message = result.error_message;
        
        test_data_.push_back(data);
        
        // Add to HTML
        if (html_file_.is_open()) {
            writeTestResultHTML(function_name, params, result);
        }
        
        // Add to CSV
        if (csv_file_.is_open()) {
            writeTestResultCSV(function_name, params, result);
        }
    }
    
    void finalizeReport() {
        // Write HTML footer
        if (html_file_.is_open()) {
            writeHTMLFooter();
        }
        
        if (csv_file_.is_open()) {
            csv_file_.close();
        }
        
        printSummary();
    }
    
private:
    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
    
    void writeHTMLHeader() {
        html_file_ << R"(
<!DOCTYPE html>
<html>
<head>
    <title>OpenNPP Test Report - )" << report_name_ << R"(</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .summary { background: #f8f8f8; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .error { background: #fff2f2; color: #d32f2f; }
    </style>
</head>
<body>
    <div class="header">
        <h1>OpenNPP Test Report</h1>
        <p><strong>Report:</strong> )" << report_name_ << R"(</p>
        <p><strong>Generated:</strong> )" << getCurrentTimestamp() << R"(</p>
    </div>
    <div class="summary">
        <h2>Test Summary</h2>
        <div id="summary-content">
            <p>Loading...</p>
        </div>
    </div>
    <table id="results-table">
        <thead>
            <tr>
                <th>Function</th>
                <th>Parameters</th>
                <th>Status</th>
                <th>Max Error</th>
                <th>CPU Time (ms)</th>
                <th>GPU Time (ms)</th>
                <th>NVIDIA Time (ms)</th>
                <th>GPU Speedup</th>
                <th>NVIDIA Speedup</th>
                <th>Error Message</th>
            </tr>
        </thead>
        <tbody>
        )";
    }
    
    void writeTestResultHTML(const std::string& function_name,
                           const NPPTest::TestParameters& params,
                           const NPPTest::TestResult& result) {
        html_file_ << "<tr>";
        html_file_ << "<td>" << function_name << "</td>";
        html_file_ << "<td>" << params.toString() << "</td>";
        
        if (result.passed) {
            html_file_ << "<td class=\"pass\">PASS</td>";
        } else {
            html_file_ << "<td class=\"fail\">FAIL</td>";
        }
        
        html_file_ << "<td>" << std::fixed << std::setprecision(4) << result.max_error << "</td>";
        html_file_ << "<td>" << std::fixed << std::setprecision(2) << result.cpu_time_ms << "</td>";
        html_file_ << "<td>" << std::fixed << std::setprecision(2) << result.gpu_time_ms << "</td>";
        html_file_ << "<td>" << std::fixed << std::setprecision(2) << result.nvidia_time_ms << "</td>";
        
        double gpu_speedup = (result.cpu_time_ms > 0) ? result.cpu_time_ms / result.gpu_time_ms : 0.0;
        double nvidia_speedup = (result.nvidia_time_ms > 0) ? result.gpu_time_ms / result.nvidia_time_ms : 0.0;
        
        html_file_ << "<td>" << std::fixed << std::setprecision(2) << gpu_speedup << "x</td>";
        html_file_ << "<td>" << std::fixed << std::setprecision(2) << nvidia_speedup << "x</td>";
        
        if (!result.error_message.empty()) {
            html_file_ << "<td class=\"error\">" << result.error_message << "</td>";
        } else {
            html_file_ << "<td>-</td>";
        }
        
        html_file_ << "</tr>\n";
    }
    
    void writeTestResultCSV(const std::string& function_name,
                          const NPPTest::TestParameters& params,
                          const NPPTest::TestResult& result) {
        csv_file_ << function_name << ",";
        csv_file_ << params.toString() << ",";
        csv_file_ << (result.passed ? "PASS" : "FAIL") << ",";
        csv_file_ << result.max_error << ",";
        csv_file_ << result.cpu_time_ms << ",";
        csv_file_ << result.gpu_time_ms << ",";
        csv_file_ << result.nvidia_time_ms << ",";
        
        double gpu_speedup = (result.cpu_time_ms > 0) ? result.cpu_time_ms / result.gpu_time_ms : 0.0;
        double nvidia_speedup = (result.nvidia_time_ms > 0) ? result.gpu_time_ms / result.nvidia_time_ms : 0.0;
        
        csv_file_ << gpu_speedup << ",";
        csv_file_ << nvidia_speedup << ",";
        csv_file_ << result.error_message << "\n";
    }
    
    void writeHTMLFooter() {
        html_file_ << R"(
        </tbody>
    </table>
    <script>
        // Generate summary
        const rows = document.querySelectorAll('#results-table tbody tr');
        let total = 0, passed = 0;
        let total_cpu_time = 0, total_gpu_time = 0, total_nvidia_time = 0;
        
        rows.forEach(row => {
            total++;
            const status = row.cells[2].textContent;
            if (status === 'PASS') passed++;
            
            total_cpu_time += parseFloat(row.cells[4].textContent);
            total_gpu_time += parseFloat(row.cells[5].textContent);
            total_nvidia_time += parseFloat(row.cells[6].textContent);
        });
        
        document.getElementById('summary-content').innerHTML = `
            <p><strong>Total Tests:</strong> ${total}</p>
            <p><strong>Passed:</strong> ${passed}</p>
            <p><strong>Failed:</strong> ${total - passed}</p>
            <p><strong>Success Rate:</strong> ${(100 * passed / total).toFixed(2)}%</p>
            <p><strong>Total CPU Time:</strong> ${total_cpu_time.toFixed(2)} ms</p>
            <p><strong>Total GPU Time:</strong> ${total_gpu_time.toFixed(2)} ms</p>
            <p><strong>Total NVIDIA Time:</strong> ${total_nvidia_time.toFixed(2)} ms</p>
        `;
    </script>
</body>
</html>
        )";
    }
    
    void printSummary() {
        int total = test_data_.size();
        int passed = 0;
        
        for (const auto& test : test_data_) {
            if (test.passed) passed++;
        }
        
        printf("\n=== Test Report Generated ===\n");
        printf("HTML Report: %s/%s.html\n", output_dir_.c_str(), report_name_.c_str());
        printf("CSV Report: %s/%s.csv\n", output_dir_.c_str(), report_name_.c_str());
        printf("Total tests: %d\n", total);
        printf("Passed: %d\n", passed);
        printf("Failed: %d\n", total - passed);
        printf("Success rate: %.2f%%\n", 100.0 * passed / total);
    }
};

#endif // TEST_REPORT_SIMPLE_H