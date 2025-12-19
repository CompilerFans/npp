/**
 * @file verify_results.cpp
 * @brief MPP vs NVIDIA NPP 像素级对比验证工具
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <ctime>
#include <set>
#include <map>
#include <cstdint>

// 终端颜色
#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_BOLD    "\033[1m"

// 全局报告文件
std::ofstream g_report;

// 统计信息
int g_total_tests = 0;
int g_perfect_match = 0;
int g_high_quality = 0;
int g_acceptable = 0;
int g_label_total = 0;
int g_label_perfect = 0;
int g_label_semantic_match = 0;  // 语义等价（新增）
int g_label_count_match = 0;     // 仅标签数一致
int g_watershed_total = 0;
int g_watershed_perfect = 0;
int g_watershed_semantic_match = 0;  // 语义等价（新增）

//=============================================================================
// 文件读取函数
//=============================================================================

bool readPGM(const char* filename, std::vector<unsigned char>& data, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    std::string magic;
    file >> magic;
    if (magic != "P5") return false;

    char c;
    file.get(c);
    while (file.peek() == '#') {
        std::string comment;
        std::getline(file, comment);
    }

    file >> width >> height;
    int maxval;
    file >> maxval;
    file.ignore();

    data.resize(width * height);
    file.read(reinterpret_cast<char*>(data.data()), width * height);
    return file.good() || file.eof();
}

bool readRaw32u(const char* filename, std::vector<uint32_t>& data, int& width, int& height) {
    std::string fname(filename);
    size_t pos = fname.rfind('_');
    if (pos == std::string::npos) return false;

    size_t pos2 = fname.rfind('_', pos - 1);
    if (pos2 == std::string::npos) return false;

    std::string sizeStr = fname.substr(pos2 + 1, pos - pos2 - 1);
    size_t xpos = sizeStr.find('x');
    if (xpos == std::string::npos) return false;

    width = std::stoi(sizeStr.substr(0, xpos));
    height = std::stoi(sizeStr.substr(xpos + 1));

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    size_t numPixels = (size_t)width * height;
    data.resize(numPixels);
    file.read(reinterpret_cast<char*>(data.data()), numPixels * sizeof(uint32_t));
    return file.good() || file.eof();
}

bool readRaw8u(const char* filename, std::vector<unsigned char>& data, int& width, int& height) {
    std::string fname(filename);
    size_t pos = fname.rfind('_');
    if (pos == std::string::npos) return false;

    size_t pos2 = fname.rfind('_', pos - 1);
    if (pos2 == std::string::npos) return false;

    std::string sizeStr = fname.substr(pos2 + 1, pos - pos2 - 1);
    size_t xpos = sizeStr.find('x');
    if (xpos == std::string::npos) return false;

    width = std::stoi(sizeStr.substr(0, xpos));
    height = std::stoi(sizeStr.substr(xpos + 1));

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    size_t numPixels = (size_t)width * height;
    data.resize(numPixels);
    file.read(reinterpret_cast<char*>(data.data()), numPixels);
    return file.good() || file.eof();
}

//=============================================================================
// 对比结果结构
//=============================================================================

struct CompareResult {
    std::string name;
    int total_pixels = 0;
    int diff_pixels = 0;
    int max_diff = 0;
    double diff_percent = 0.0;
    double psnr = 0.0;
    std::string status;
    int ref_unique_labels = 0;
    int mpp_unique_labels = 0;
    // 语义等价性验证结果
    bool semantically_equivalent = false;
    int connectivity_errors = 0;  // 连通性错误数
};

//=============================================================================
// 语义等价性验证函数
//=============================================================================

/**
 * 验证两个标签图是否语义等价
 *
 * 原理：如果两个标签图表示相同的连通区域划分，那么：
 * 1. 在 ref 中属于同一标签的任意两个像素，在 mpp 中也必须属于同一标签
 * 2. 在 ref 中属于不同标签的任意两个像素，在 mpp 中也必须属于不同标签
 *
 * 实现方法：
 * - 建立 ref_label -> mpp_label 的映射
 * - 如果同一个 ref_label 映射到多个不同的 mpp_label，说明连通性被破坏
 * - 如果不同的 ref_label 映射到同一个 mpp_label，说明区域被错误合并
 */
bool checkSemanticEquivalence(const std::vector<uint32_t>& ref_data,
                               const std::vector<uint32_t>& mpp_data,
                               int& connectivity_errors) {
    connectivity_errors = 0;

    if (ref_data.size() != mpp_data.size()) return false;

    // ref_label -> mpp_label 的映射
    std::map<uint32_t, uint32_t> ref_to_mpp;
    // mpp_label -> ref_label 的映射（用于检测错误合并）
    std::map<uint32_t, uint32_t> mpp_to_ref;

    for (size_t i = 0; i < ref_data.size(); i++) {
        uint32_t ref_label = ref_data[i];
        uint32_t mpp_label = mpp_data[i];

        // 检查 ref_label 是否已经映射到某个 mpp_label
        auto it = ref_to_mpp.find(ref_label);
        if (it == ref_to_mpp.end()) {
            // 第一次遇到这个 ref_label，建立映射
            ref_to_mpp[ref_label] = mpp_label;
        } else if (it->second != mpp_label) {
            // 同一个 ref_label 映射到了不同的 mpp_label
            // 说明 ref 中的一个连通区域在 mpp 中被分割了
            connectivity_errors++;
        }

        // 检查 mpp_label 是否已经映射到某个 ref_label
        auto it2 = mpp_to_ref.find(mpp_label);
        if (it2 == mpp_to_ref.end()) {
            mpp_to_ref[mpp_label] = ref_label;
        } else if (it2->second != ref_label) {
            // 同一个 mpp_label 映射到了不同的 ref_label
            // 说明 ref 中的多个连通区域在 mpp 中被错误合并了
            connectivity_errors++;
        }
    }

    return connectivity_errors == 0;
}

/**
 * 验证 8u 标签图的语义等价性（用于 Watershed 的 Segments 输出）
 */
bool checkSemanticEquivalence8u(const std::vector<unsigned char>& ref_data,
                                 const std::vector<unsigned char>& mpp_data,
                                 int& connectivity_errors) {
    connectivity_errors = 0;

    if (ref_data.size() != mpp_data.size()) return false;

    std::map<uint8_t, uint8_t> ref_to_mpp;
    std::map<uint8_t, uint8_t> mpp_to_ref;

    for (size_t i = 0; i < ref_data.size(); i++) {
        uint8_t ref_label = ref_data[i];
        uint8_t mpp_label = mpp_data[i];

        auto it = ref_to_mpp.find(ref_label);
        if (it == ref_to_mpp.end()) {
            ref_to_mpp[ref_label] = mpp_label;
        } else if (it->second != mpp_label) {
            connectivity_errors++;
        }

        auto it2 = mpp_to_ref.find(mpp_label);
        if (it2 == mpp_to_ref.end()) {
            mpp_to_ref[mpp_label] = ref_label;
        } else if (it2->second != ref_label) {
            connectivity_errors++;
        }
    }

    return connectivity_errors == 0;
}

//=============================================================================
// 对比函数
//=============================================================================

CompareResult compareImagesPGM(const char* ref_file, const char* mpp_file, const char* name) {
    CompareResult result;
    result.name = name;

    std::vector<unsigned char> ref_data, mpp_data;
    int ref_w, ref_h, mpp_w, mpp_h;

    if (!readPGM(ref_file, ref_data, ref_w, ref_h)) {
        result.status = "REF_MISSING";
        return result;
    }

    if (!readPGM(mpp_file, mpp_data, mpp_w, mpp_h)) {
        result.status = "MPP_MISSING";
        return result;
    }

    if (ref_w != mpp_w || ref_h != mpp_h) {
        result.status = "SIZE_MISMATCH";
        return result;
    }

    result.total_pixels = ref_w * ref_h;
    double mse = 0.0;

    for (int i = 0; i < result.total_pixels; i++) {
        int diff = std::abs(static_cast<int>(ref_data[i]) - static_cast<int>(mpp_data[i]));
        if (diff > 0) {
            result.diff_pixels++;
            result.max_diff = std::max(result.max_diff, diff);
        }
        mse += diff * diff;
    }

    mse /= result.total_pixels;
    result.diff_percent = 100.0 * result.diff_pixels / result.total_pixels;
    result.psnr = (mse > 0) ? 20.0 * std::log10(255.0 / std::sqrt(mse)) : 100.0;

    if (result.diff_pixels == 0) {
        result.status = "PERFECT";
    } else if (result.psnr >= 40.0) {
        result.status = "HIGH_QUALITY";
    } else if (result.psnr >= 30.0) {
        result.status = "ACCEPTABLE";
    } else {
        result.status = "DIFFERENT";
    }

    return result;
}

CompareResult compareLabels32u(const char* ref_file, const char* mpp_file, const char* name) {
    CompareResult result;
    result.name = name;

    std::vector<uint32_t> ref_data, mpp_data;
    int ref_w, ref_h, mpp_w, mpp_h;

    if (!readRaw32u(ref_file, ref_data, ref_w, ref_h)) {
        result.status = "REF_MISSING";
        return result;
    }

    if (!readRaw32u(mpp_file, mpp_data, mpp_w, mpp_h)) {
        result.status = "MPP_MISSING";
        return result;
    }

    if (ref_w != mpp_w || ref_h != mpp_h) {
        result.status = "SIZE_MISMATCH";
        return result;
    }

    result.total_pixels = ref_w * ref_h;

    // 统计唯一标签数和像素差异
    std::set<uint32_t> ref_labels, mpp_labels;
    for (size_t i = 0; i < ref_data.size(); i++) {
        ref_labels.insert(ref_data[i]);
        mpp_labels.insert(mpp_data[i]);
        if (ref_data[i] != mpp_data[i]) {
            result.diff_pixels++;
        }
    }

    result.ref_unique_labels = ref_labels.size();
    result.mpp_unique_labels = mpp_labels.size();
    result.diff_percent = 100.0 * result.diff_pixels / result.total_pixels;

    // 语义等价性验证（关键改进！）
    result.semantically_equivalent = checkSemanticEquivalence(ref_data, mpp_data, result.connectivity_errors);

    // 状态判定（使用语义等价性而不是简单的标签数比较）
    if (result.diff_pixels == 0) {
        result.status = "PERFECT";
    } else if (result.semantically_equivalent) {
        // 像素值不同，但语义等价（连通区域划分相同）
        result.status = "SEMANTIC_MATCH";
    } else if (result.ref_unique_labels == result.mpp_unique_labels) {
        // 标签数相同，但连通性有错误
        result.status = "LABEL_COUNT_MATCH";
    } else {
        // 标签数不同，连通性也有错误
        result.status = "LABEL_DIFF";
    }

    return result;
}

/**
 * 对比 8u RAW 图像（像素级对比）
 * 用于 Watershed 的 SegmentBoundaries 输出（边界图像，确定性）
 */
CompareResult compareRaw8u(const char* ref_file, const char* mpp_file, const char* name) {
    CompareResult result;
    result.name = name;

    std::vector<unsigned char> ref_data, mpp_data;
    int ref_w, ref_h, mpp_w, mpp_h;

    if (!readRaw8u(ref_file, ref_data, ref_w, ref_h)) {
        result.status = "REF_MISSING";
        return result;
    }

    if (!readRaw8u(mpp_file, mpp_data, mpp_w, mpp_h)) {
        result.status = "MPP_MISSING";
        return result;
    }

    if (ref_w != mpp_w || ref_h != mpp_h) {
        result.status = "SIZE_MISMATCH";
        return result;
    }

    result.total_pixels = ref_w * ref_h;
    double mse = 0.0;

    for (int i = 0; i < result.total_pixels; i++) {
        int diff = std::abs(static_cast<int>(ref_data[i]) - static_cast<int>(mpp_data[i]));
        if (diff > 0) {
            result.diff_pixels++;
            result.max_diff = std::max(result.max_diff, diff);
        }
        mse += diff * diff;
    }

    mse /= result.total_pixels;
    result.diff_percent = 100.0 * result.diff_pixels / result.total_pixels;
    result.psnr = (mse > 0) ? 20.0 * std::log10(255.0 / std::sqrt(mse)) : 100.0;

    if (result.diff_pixels == 0) {
        result.status = "PERFECT";
    } else if (result.psnr >= 40.0) {
        result.status = "HIGH_QUALITY";
    } else if (result.psnr >= 30.0) {
        result.status = "ACCEPTABLE";
    } else {
        result.status = "DIFFERENT";
    }

    return result;
}

/**
 * 对比 8u 标签图像（使用语义等价性验证）
 * 用于 Watershed 的 Segments 输出
 */
CompareResult compareLabels8u(const char* ref_file, const char* mpp_file, const char* name) {
    CompareResult result;
    result.name = name;

    std::vector<unsigned char> ref_data, mpp_data;
    int ref_w, ref_h, mpp_w, mpp_h;

    if (!readRaw8u(ref_file, ref_data, ref_w, ref_h)) {
        result.status = "REF_MISSING";
        return result;
    }

    if (!readRaw8u(mpp_file, mpp_data, mpp_w, mpp_h)) {
        result.status = "MPP_MISSING";
        return result;
    }

    if (ref_w != mpp_w || ref_h != mpp_h) {
        result.status = "SIZE_MISMATCH";
        return result;
    }

    result.total_pixels = ref_w * ref_h;

    // 统计唯一标签数和像素差异
    std::set<uint8_t> ref_labels, mpp_labels;
    for (size_t i = 0; i < ref_data.size(); i++) {
        ref_labels.insert(ref_data[i]);
        mpp_labels.insert(mpp_data[i]);
        if (ref_data[i] != mpp_data[i]) {
            result.diff_pixels++;
        }
    }

    result.ref_unique_labels = ref_labels.size();
    result.mpp_unique_labels = mpp_labels.size();
    result.diff_percent = 100.0 * result.diff_pixels / result.total_pixels;

    // 语义等价性验证
    result.semantically_equivalent = checkSemanticEquivalence8u(ref_data, mpp_data, result.connectivity_errors);

    // 状态判定
    if (result.diff_pixels == 0) {
        result.status = "PERFECT";
    } else if (result.semantically_equivalent) {
        result.status = "SEMANTIC_MATCH";
    } else if (result.ref_unique_labels == result.mpp_unique_labels) {
        result.status = "LABEL_COUNT_MATCH";
    } else {
        result.status = "LABEL_DIFF";
    }

    return result;
}

//=============================================================================
// 输出函数
//=============================================================================

void printTableHeader(const char* title) {
    std::cout << "\n" << COLOR_BOLD << COLOR_BLUE << title << COLOR_RESET << "\n";
    std::cout << "+----------------------------------+----------+----------+----------------------+\n";
    std::cout << "| " << COLOR_BOLD << "Name" << COLOR_RESET
              << "                             |   " << COLOR_BOLD << "Diff" << COLOR_RESET
              << "   |   " << COLOR_BOLD << "PSNR" << COLOR_RESET
              << "   | " << COLOR_BOLD << "Status" << COLOR_RESET
              << "               |\n";
    std::cout << "+----------------------------------+----------+----------+----------------------+\n";
}

void printTableHeaderLabel(const char* title) {
    std::cout << "\n" << COLOR_BOLD << COLOR_BLUE << title << COLOR_RESET << "\n";
    // 使用英文标题避免中文字符宽度问题
    std::cout << "+----------------------------------------------------------------+---------------+---------+----------------------+\n";
    std::cout << "| " << COLOR_BOLD << "Name" << COLOR_RESET
              << "                                                           | " << COLOR_BOLD << "Labels" << COLOR_RESET
              << "        | " << COLOR_BOLD << "Equiv" << COLOR_RESET
              << "   | " << COLOR_BOLD << "Status" << COLOR_RESET
              << "               |\n";
    std::cout << "|                                                                | REF/MPP       |         |                      |\n";
    std::cout << "+----------------------------------------------------------------+---------------+---------+----------------------+\n";
}

void printTableFooter() {
    std::cout << "+----------------------------------+----------+----------+----------------------+\n";
}

void printTableFooterLabel() {
    std::cout << "+----------------------------------------------------------------+---------------+---------+----------------------+\n";
}

std::string getStatusIcon(const std::string& status) {
    if (status == "PERFECT") return COLOR_GREEN "PASS" COLOR_RESET " Perfect         ";
    if (status == "HIGH_QUALITY") return COLOR_GREEN "OK  " COLOR_RESET " High Quality    ";
    if (status == "ACCEPTABLE") return COLOR_YELLOW "WARN" COLOR_RESET " Acceptable      ";
    if (status == "DIFFERENT") return COLOR_RED "DIFF" COLOR_RESET " Different       ";
    if (status == "SEMANTIC_MATCH") return COLOR_GREEN "PASS" COLOR_RESET " Semantic Eq     ";
    if (status == "LABEL_COUNT_MATCH") return COLOR_YELLOW "WARN" COLOR_RESET " Count Match     ";
    if (status == "LABEL_DIFF") return COLOR_RED "FAIL" COLOR_RESET " Connect Err     ";
    if (status == "LABEL_MATCH") return COLOR_GREEN "OK  " COLOR_RESET " Label Match     ";
    if (status == "REF_MISSING") return COLOR_RED "SKIP" COLOR_RESET " No Reference    ";
    if (status == "MPP_MISSING") return COLOR_RED "FAIL" COLOR_RESET " MPP Missing     ";
    return "                     ";
}

void printResultPGM(const CompareResult& r) {
    std::cout << "| " << std::left << std::setw(32) << r.name << " |";

    if (r.total_pixels == 0) {
        std::cout << "   N/A    |   N/A    | " << getStatusIcon(r.status) << "|\n";
        return;
    }

    std::cout << "  " << std::right << std::setw(5) << std::fixed << std::setprecision(2) << r.diff_percent << "%  |";
    std::cout << "  " << std::setw(5) << std::setprecision(1) << r.psnr << "dB |";
    std::cout << " " << getStatusIcon(r.status) << "|\n";
}

void printResultLabel(const CompareResult& r) {
    std::cout << "| " << std::left << std::setw(62) << r.name << " |";

    if (r.total_pixels == 0) {
        std::cout << "      N/A      |  N/A   | " << getStatusIcon(r.status) << "|\n";
        return;
    }

    std::cout << " " << std::right << std::setw(6) << r.ref_unique_labels << "/"
              << std::left << std::setw(6) << r.mpp_unique_labels << " |";

    if (r.semantically_equivalent) {
        std::cout << "   " << COLOR_GREEN << "Yes" << COLOR_RESET << "   | ";
    } else {
        std::cout << "   " << COLOR_RED << "No " << COLOR_RESET << "   | ";
    }

    std::cout << getStatusIcon(r.status) << "|\n";
}

void writeReportRowPGM(const CompareResult& r, const char* sample, const char* file) {
    std::string icon = (r.status == "PERFECT") ? "[PASS]" :
                       (r.status == "HIGH_QUALITY") ? "[OK]" :
                       (r.status == "ACCEPTABLE") ? "[WARN]" : "[DIFF]";

    g_report << "| " << sample << " | " << file << " | " << icon << " | ";

    if (r.total_pixels == 0) {
        g_report << "N/A | N/A |\n";
    } else {
        g_report << std::fixed << std::setprecision(2) << r.diff_percent << "% | "
                 << std::setprecision(1) << r.psnr << "dB |\n";
    }
}

void writeReportRowLabel(const CompareResult& r, const char* sample, const char* file) {
    std::string icon;
    if (r.status == "PERFECT" || r.status == "SEMANTIC_MATCH") {
        icon = "[PASS]";
    } else if (r.status == "LABEL_COUNT_MATCH" || r.status == "LABEL_MATCH") {
        icon = "[WARN]";
    } else {
        icon = "[FAIL]";
    }

    g_report << "| " << sample << " | " << file << " | ";

    if (r.total_pixels == 0) {
        g_report << "N/A | N/A | " << icon << " |\n";
    } else {
        // 显示标签数量对比和语义等价状态
        g_report << r.ref_unique_labels << "/" << r.mpp_unique_labels << " | "
                 << (r.semantically_equivalent ? "是" : "否") << " | "
                 << icon << " |\n";
    }
}

//=============================================================================
// 主函数
//=============================================================================

int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << COLOR_BOLD COLOR_CYAN;
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              MPP vs NVIDIA NPP 像素级对比验证                                ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
    std::cout << COLOR_RESET;

    // 打开报告文件
    g_report.open("VERIFICATION_REPORT.md");

    time_t now = time(nullptr);
    char timestr[64];
    strftime(timestr, sizeof(timestr), "%Y-%m-%d %H:%M:%S", localtime(&now));

    g_report << "# MPP vs NVIDIA NPP 验证报告\n\n";
    g_report << "**生成时间**: " << timestr << "\n\n";
    g_report << "---\n\n";

    // 添加验证方法说明
    g_report << "## 验证方法说明\n\n";
    g_report << "### 比较原理\n\n";
    g_report << "本工具将 MPP 实现的输出与 NVIDIA NPP 官方库的输出进行逐像素对比：\n\n";
    g_report << "1. **读取文件**: 分别读取 MPP 输出和 NVIDIA NPP 参考输出\n";
    g_report << "2. **逐像素比较**: 对每个像素位置，计算 `|MPP[i] - NVIDIA[i]|`\n";
    g_report << "3. **统计差异**: 统计不相等的像素数量\n\n";
    g_report << "### 指标定义\n\n";
    g_report << "| 指标 | 计算公式 | 说明 |\n";
    g_report << "|------|----------|------|\n";
    g_report << "| 差异像素数 | `count(MPP[i] != REF[i])` | 值不相等的像素总数 |\n";
    g_report << "| 差异比例 | `差异像素数 / 总像素数 × 100%` | 不匹配像素的百分比 |\n";
    g_report << "| PSNR | `20 × log10(255 / √MSE)` | 峰值信噪比，越高越好 |\n";
    g_report << "| MSE | `Σ(MPP[i] - REF[i])² / N` | 均方误差 |\n\n";
    g_report << "### 状态说明\n\n";
    g_report << "| 状态 | 条件 | 含义 |\n";
    g_report << "|------|------|------|\n";
    g_report << "| [PASS] | 差异比例 = 0% | 完全匹配 |\n";
    g_report << "| [OK] | PSNR ≥ 40dB 或 标签数一致 | 高质量匹配 |\n";
    g_report << "| [WARN] | 30dB ≤ PSNR < 40dB | 可接受的差异 |\n";
    g_report << "| [DIFF] | PSNR < 30dB | 存在明显差异 |\n\n";
    g_report << "---\n\n";

    //=========================================================================
    // 1. 基础图像处理对比
    //=========================================================================
    struct TestPGM { const char* ref; const char* mpp; const char* name; const char* sample; const char* file; };
    TestPGM basic_tests[] = {
        {"reference_nvidia_npp/teapot512_boxFilter.pgm", "mpp_results/teapot512_boxFilter.pgm",
         "boxFilterNPP", "boxFilterNPP", "teapot512_boxFilter.pgm"},
        {"reference_nvidia_npp/teapot512_cannyEdgeDetection.pgm", "mpp_results/teapot512_cannyEdgeDetection.pgm",
         "cannyEdgeDetectorNPP", "cannyEdgeDetectorNPP", "teapot512_cannyEdgeDetection.pgm"},
        {"reference_nvidia_npp/teapot512_histEqualization.pgm", "mpp_results/teapot512_histEqualization.pgm",
         "histEqualizationNPP", "histEqualizationNPP", "teapot512_histEqualization.pgm"},
        {"reference_nvidia_npp/teapot512_boxFilterFII.pgm", "mpp_results/teapot512_boxFilterFII.pgm",
         "freeImageInteropNPP", "freeImageInteropNPP", "teapot512_boxFilterFII.pgm"}
    };

    printTableHeader("基础图像处理");
    g_report << "## 基础图像处理\n\n";
    g_report << "| 示例程序 | 输出文件 | 状态 | 差异比例 | PSNR |\n";
    g_report << "|----------|----------|------|----------|------|\n";

    for (const auto& t : basic_tests) {
        auto r = compareImagesPGM(t.ref, t.mpp, t.name);
        printResultPGM(r);
        writeReportRowPGM(r, t.sample, t.file);

        if (r.total_pixels > 0) {
            g_total_tests++;
            if (r.status == "PERFECT") g_perfect_match++;
            else if (r.status == "HIGH_QUALITY") g_high_quality++;
            else if (r.status == "ACCEPTABLE") g_acceptable++;
        }
    }
    printTableFooter();
    g_report << "\n";

    //=========================================================================
    // 2. FilterBorderControl 对比
    //=========================================================================
    TestPGM filter_tests[] = {
        {"reference_nvidia_npp/FilterBorderControl/teapot512_gradientVectorPrewittBorderY_Horizontal.pgm",
         "mpp_results/FilterBorderControl/teapot512_gradientVectorPrewittBorderY_Horizontal.pgm",
         "Prewitt Y Horizontal", "FilterBorderControlNPP", "PrewittBorderY_Horizontal.pgm"},
        {"reference_nvidia_npp/FilterBorderControl/teapot512.pgm_gradientVectorPrewittBorderX_Vertical.pgm",
         "mpp_results/FilterBorderControl/teapot512.pgm_gradientVectorPrewittBorderX_Vertical.pgm",
         "Prewitt X Vertical", "FilterBorderControlNPP", "PrewittBorderX_Vertical.pgm"}
    };

    printTableHeader("FilterBorderControl");
    g_report << "## FilterBorderControl\n\n";
    g_report << "| 示例程序 | 输出文件 | 状态 | 差异比例 | PSNR |\n";
    g_report << "|----------|----------|------|----------|------|\n";

    for (const auto& t : filter_tests) {
        auto r = compareImagesPGM(t.ref, t.mpp, t.name);
        printResultPGM(r);
        writeReportRowPGM(r, t.sample, t.file);

        if (r.total_pixels > 0) {
            g_total_tests++;
            if (r.status == "PERFECT") g_perfect_match++;
            else if (r.status == "HIGH_QUALITY") g_high_quality++;
            else if (r.status == "ACCEPTABLE") g_acceptable++;
        }
    }
    printTableFooter();
    g_report << "\n";

    //=========================================================================
    // 3. BatchedLabelMarkers 对比
    //=========================================================================
    struct LabelImage { const char* image; const char* size; };
    LabelImage images[] = {
        {"teapot", "512x512"},
        {"CT_skull", "512x512"},
        {"PCB_METAL", "509x335"},
        {"PCB", "1280x720"},
        {"PCB2", "1024x683"},
    };
    const char* types[] = {"LabelMarkersUF", "LabelMarkersUFBatch",
                           "CompressedMarkerLabelsUF", "CompressedMarkerLabelsUFBatch"};

    printTableHeaderLabel("BatchedLabelMarkers (32u标签图像)");
    g_report << "## BatchedLabelMarkers\n\n";
    g_report << "| 示例程序 | 测试文件 | 标签数(REF/MPP) | 语义等价 | 状态 |\n";
    g_report << "|----------|----------|-----------------|----------|------|\n";

    for (const auto& img : images) {
        for (const auto& type : types) {
            char ref[256], mpp[256], name[128];
            snprintf(ref, sizeof(ref), "reference_nvidia_npp/batchedLabelMarkers/%s_%s_8Way_%s_32u.raw",
                     img.image, type, img.size);
            snprintf(mpp, sizeof(mpp), "mpp_results/batchedLabelMarkers/%s_%s_8Way_%s_32u.raw",
                     img.image, type, img.size);
            snprintf(name, sizeof(name), "%s_%s", img.image, type);

            auto r = compareLabels32u(ref, mpp, name);
            printResultLabel(r);
            writeReportRowLabel(r, "batchedLabelMarkersNPP", name);

            if (r.total_pixels > 0) {
                g_label_total++;
                if (r.status == "PERFECT") g_label_perfect++;
                else if (r.status == "SEMANTIC_MATCH") g_label_semantic_match++;
                else if (r.status == "LABEL_COUNT_MATCH") g_label_count_match++;
            }
        }
    }
    printTableFooterLabel();
    g_report << "\n";

    //=========================================================================
    // 4. Watershed 对比
    //=========================================================================
    // verify_type: 0=8u标签(语义等价), 1=32u标签(语义等价)
    struct WatershedTest { const char* image; const char* type; int verify_type; };
    WatershedTest watershed_tests[] = {
        {"teapot", "Segments_8Way_512x512_8u.raw", 0},                            // 分割标签 - 语义等价
        {"teapot", "CompressedSegmentLabels_8Way_512x512_32u.raw", 1},            // 压缩标签 - 语义等价
        {"teapot", "SegmentBoundaries_8Way_512x512_8u.raw", 0},                   // 边界图像 - 语义等价（依赖分割结果）
        {"teapot", "SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw", 0},   // 带边界分割 - 语义等价
        {"CT_skull", "Segments_8Way_512x512_8u.raw", 0},
        {"CT_skull", "CompressedSegmentLabels_8Way_512x512_32u.raw", 1},
        {"CT_skull", "SegmentBoundaries_8Way_512x512_8u.raw", 0},                 // 边界图像 - 语义等价
        {"CT_skull", "SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw", 0},
        {"Rocks", "Segments_8Way_512x512_8u.raw", 0},
        {"Rocks", "CompressedSegmentLabels_8Way_512x512_32u.raw", 1},
        {"Rocks", "SegmentBoundaries_8Way_512x512_8u.raw", 0},                    // 边界图像 - 语义等价
        {"Rocks", "SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw", 0},
    };

    printTableHeaderLabel("Watershed");
    g_report << "## Watershed\n\n";
    g_report << "| 示例程序 | 测试文件 | 标签数(REF/MPP) | 语义等价 | 状态 |\n";
    g_report << "|----------|----------|-----------------|----------|------|\n";

    for (const auto& t : watershed_tests) {
        char ref[256], mpp[256], name[128];
        snprintf(ref, sizeof(ref), "reference_nvidia_npp/watershed/%s_%s", t.image, t.type);
        snprintf(mpp, sizeof(mpp), "mpp_results/watershed/%s_%s", t.image, t.type);
        snprintf(name, sizeof(name), "%s_%s", t.image, t.type);

        CompareResult r;
        if (t.verify_type == 1) {
            // 32u 标签图 - 语义等价性验证
            r = compareLabels32u(ref, mpp, name);
        } else {
            // 8u 标签图 - 语义等价性验证（包括 Segments, SegmentBoundaries, SegmentsWithContrastingBoundaries）
            r = compareLabels8u(ref, mpp, name);
        }
        printResultLabel(r);
        writeReportRowLabel(r, "watershedSegmentationNPP", name);

        if (r.total_pixels > 0) {
            g_watershed_total++;
            if (r.status == "PERFECT") g_watershed_perfect++;
            else if (r.status == "SEMANTIC_MATCH") g_watershed_semantic_match++;
        }
    }
    printTableFooterLabel();
    g_report << "\n";

    //=========================================================================
    // 总结
    //=========================================================================
    std::cout << "\n";
    std::cout << COLOR_BOLD COLOR_CYAN;
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                              验证结果总结                                    ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
    std::cout << COLOR_RESET;

    std::cout << "\n  " << COLOR_BOLD << "基础图像处理 (确定性算法):" << COLOR_RESET << "\n";
    std::cout << "    - 完全匹配: " << COLOR_GREEN << g_perfect_match << "/" << g_total_tests << COLOR_RESET << "\n";
    std::cout << "    - 高质量:   " << g_high_quality << "/" << g_total_tests << "\n";
    int basic_fail = g_total_tests - g_perfect_match - g_high_quality - g_acceptable;
    if (basic_fail > 0) {
        std::cout << "    - " << COLOR_RED << "需要修复: " << basic_fail << COLOR_RESET << "\n";
    }

    std::cout << "\n  " << COLOR_BOLD << "BatchedLabelMarkers (语义等价性验证):" << COLOR_RESET << "\n";
    int label_pass = g_label_perfect + g_label_semantic_match;
    std::cout << "    - 验证通过: " << COLOR_GREEN << label_pass << "/" << g_label_total << COLOR_RESET
              << " (完全匹配:" << g_label_perfect << " + 语义等价:" << g_label_semantic_match << ")\n";
    if (g_label_count_match > 0) {
        std::cout << "    - " << COLOR_YELLOW << "标签数一致但连通性有误: " << g_label_count_match << COLOR_RESET << "\n";
    }
    int label_fail = g_label_total - label_pass - g_label_count_match;
    if (label_fail > 0) {
        std::cout << "    - " << COLOR_RED << "连通性错误: " << label_fail << COLOR_RESET << "\n";
    }

    std::cout << "\n  " << COLOR_BOLD << "Watershed (语义等价性验证):" << COLOR_RESET << "\n";
    int watershed_pass = g_watershed_perfect + g_watershed_semantic_match;
    std::cout << "    - 验证通过: " << COLOR_GREEN << watershed_pass << "/" << g_watershed_total << COLOR_RESET
              << " (完全匹配:" << g_watershed_perfect << " + 语义等价:" << g_watershed_semantic_match << ")\n";

    // 写入报告总结
    g_report << "---\n\n";
    g_report << "## 验证结果总结\n\n";

    // 总体统计（使用语义等价性）
    int total_all = g_total_tests + g_label_total + g_watershed_total;
    int label_pass_count = g_label_perfect + g_label_semantic_match;
    int watershed_pass_count = g_watershed_perfect + g_watershed_semantic_match;
    int pass_all = g_perfect_match + label_pass_count + watershed_pass_count;

    g_report << "### 总体统计\n\n";
    g_report << "| 类别 | 测试数 | 通过 | 通过率 | 说明 |\n";
    g_report << "|------|--------|------|--------|------|\n";
    g_report << "| 基础图像处理 | " << g_total_tests << " | " << g_perfect_match << " | "
             << std::fixed << std::setprecision(1)
             << (g_total_tests > 0 ? 100.0 * g_perfect_match / g_total_tests : 0) << "% | 像素级精确匹配 |\n";
    g_report << "| BatchedLabelMarkers | " << g_label_total << " | " << label_pass_count << " | "
             << std::fixed << std::setprecision(1)
             << (g_label_total > 0 ? 100.0 * label_pass_count / g_label_total : 0) << "% | 语义等价性验证 |\n";
    g_report << "| Watershed | " << g_watershed_total << " | " << watershed_pass_count << " | "
             << std::fixed << std::setprecision(1)
             << (g_watershed_total > 0 ? 100.0 * watershed_pass_count / g_watershed_total : 0) << "% | 语义等价性验证 |\n";
    g_report << "| **总计** | " << total_all << " | " << pass_all << " | "
             << std::fixed << std::setprecision(1)
             << (total_all > 0 ? 100.0 * pass_all / total_all : 0) << "% | |\n\n";

    // 基础图像处理详细
    g_report << "### 基础图像处理详细\n\n";
    g_report << "这些是**确定性算法**，相同输入必须产生相同输出。\n\n";
    g_report << "| 状态 | 数量 | 说明 |\n";
    g_report << "|------|------|------|\n";
    g_report << "| [PASS] | " << g_perfect_match << " | 完全匹配 (差异比例 = 0%) |\n";
    g_report << "| [OK] | " << g_high_quality << " | 高质量 (PSNR >= 40dB) |\n";
    g_report << "| [WARN] | " << g_acceptable << " | 可接受 (30dB <= PSNR < 40dB) |\n";
    g_report << "| [FAIL] | " << (g_total_tests - g_perfect_match - g_high_quality - g_acceptable)
             << " | **需要修复** (PSNR < 30dB) |\n\n";

    // BatchedLabelMarkers详细
    g_report << "### BatchedLabelMarkers 详细\n\n";
    g_report << "这些是**非确定性算法**，使用语义等价性验证。\n\n";
    g_report << "| 状态 | 数量 | 说明 |\n";
    g_report << "|------|------|------|\n";
    g_report << "| [PASS] 完全匹配 | " << g_label_perfect << " | 所有像素值完全相同 |\n";
    g_report << "| [PASS] 语义等价 | " << g_label_semantic_match << " | 像素值不同但连通区域划分相同 |\n";
    g_report << "| [WARN] 标签数一致 | " << g_label_count_match << " | 标签数相同但连通性有误 |\n";
    g_report << "| [FAIL] 连通性错误 | " << (g_label_total - g_label_perfect - g_label_semantic_match - g_label_count_match)
             << " | **需要修复** |\n\n";

    // 说明
    g_report << "### 验证方法说明\n\n";
    g_report << "#### 语义等价性验证原理\n\n";
    g_report << "对于标签图像，我们不比较具体的标签值，而是验证**连通区域划分是否相同**：\n\n";
    g_report << "1. 在 NVIDIA 结果中属于同一标签的像素，在 MPP 结果中也必须属于同一标签\n";
    g_report << "2. 在 NVIDIA 结果中属于不同标签的像素，在 MPP 结果中也必须属于不同标签\n\n";
    g_report << "如果满足以上条件，即使标签值完全不同，也认为是**语义等价**的正确实现。\n\n";
    g_report << "#### 为什么标签值可以不同？\n\n";
    g_report << "Union-Find 算法的标签分配取决于 GPU 线程执行顺序，这是非确定性的。\n";
    g_report << "不同实现、甚至同一实现的不同运行，都可能产生不同的标签值。\n";

    g_report.close();

    std::cout << "\n  " << COLOR_CYAN << "详细报告: VERIFICATION_REPORT.md" << COLOR_RESET << "\n\n";

    return 0;
}
