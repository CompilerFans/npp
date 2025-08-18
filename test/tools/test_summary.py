#!/usr/bin/env python3
"""
OpenNPP测试结果汇总脚本
解析GTest输出并生成友好的测试报告
"""

import sys
import re
import json
from datetime import datetime
from pathlib import Path
import subprocess
import argparse

class TestSummary:
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.disabled_tests = 0
        self.test_suites = {}
        self.execution_time = 0
        
    def parse_gtest_output(self, output):
        """解析GTest输出"""
        lines = output.split('\n')
        
        for line in lines:
            # 匹配测试套件开始
            suite_match = re.match(r'\[----------\] (\d+) tests? from (\w+)', line)
            if suite_match:
                count = int(suite_match.group(1))
                suite_name = suite_match.group(2)
                if suite_name not in self.test_suites:
                    self.test_suites[suite_name] = {
                        'total': count,
                        'passed': 0,
                        'failed': 0,
                        'tests': []
                    }
            
            # 匹配测试结果
            test_match = re.match(r'\[ (OK|FAILED) \] (\w+)\.(\w+) \((\d+) ms\)', line)
            if test_match:
                status = test_match.group(1)
                suite = test_match.group(2)
                test = test_match.group(3)
                time = int(test_match.group(4))
                
                if suite in self.test_suites:
                    self.test_suites[suite]['tests'].append({
                        'name': test,
                        'status': status,
                        'time': time
                    })
                    if status == 'OK':
                        self.test_suites[suite]['passed'] += 1
                    else:
                        self.test_suites[suite]['failed'] += 1
            
            # 匹配总结信息
            summary_match = re.match(r'\[==========\] (\d+) tests? from (\d+) test (?:suite|case)', line)
            if summary_match:
                self.total_tests = int(summary_match.group(1))
            
            passed_match = re.match(r'\[  PASSED  \] (\d+) tests?', line)
            if passed_match:
                self.passed_tests = int(passed_match.group(1))
            
            failed_match = re.match(r'\[  FAILED  \] (\d+) tests?', line)
            if failed_match:
                self.failed_tests = int(failed_match.group(1))
                
            time_match = re.match(r'\[==========\].* \((\d+) ms total\)', line)
            if time_match:
                self.execution_time = int(time_match.group(1))
    
    def generate_report(self, format='text'):
        """生成报告"""
        if format == 'text':
            return self._generate_text_report()
        elif format == 'json':
            return self._generate_json_report()
        elif format == 'markdown':
            return self._generate_markdown_report()
    
    def _generate_text_report(self):
        """生成文本报告"""
        report = []
        report.append("=" * 60)
        report.append("OpenNPP 测试报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 总体统计
        report.append("总体统计:")
        report.append(f"  总测试数: {self.total_tests}")
        report.append(f"  通过: {self.passed_tests} ({self._percentage(self.passed_tests, self.total_tests)}%)")
        report.append(f"  失败: {self.failed_tests} ({self._percentage(self.failed_tests, self.total_tests)}%)")
        report.append(f"  执行时间: {self.execution_time}ms")
        report.append("")
        
        # 各测试套件详情
        report.append("测试套件详情:")
        for suite_name, suite_data in self.test_suites.items():
            status = "✓" if suite_data['failed'] == 0 else "✗"
            report.append(f"  {status} {suite_name}:")
            report.append(f"      通过: {suite_data['passed']}/{suite_data['total']}")
            if suite_data['failed'] > 0:
                report.append(f"      失败: {suite_data['failed']}")
                # 列出失败的测试
                for test in suite_data['tests']:
                    if test['status'] == 'FAILED':
                        report.append(f"        - {test['name']}")
        
        return "\n".join(report)
    
    def _generate_markdown_report(self):
        """生成Markdown报告"""
        report = []
        report.append("# OpenNPP 测试报告")
        report.append("")
        report.append(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 总体统计表
        report.append("## 总体统计")
        report.append("")
        report.append("| 指标 | 数值 | 百分比 |")
        report.append("|------|------|--------|")
        report.append(f"| 总测试数 | {self.total_tests} | 100% |")
        report.append(f"| ✅ 通过 | {self.passed_tests} | {self._percentage(self.passed_tests, self.total_tests)}% |")
        report.append(f"| ❌ 失败 | {self.failed_tests} | {self._percentage(self.failed_tests, self.total_tests)}% |")
        report.append(f"| ⏱️ 执行时间 | {self.execution_time}ms | - |")
        report.append("")
        
        # 测试套件详情
        report.append("## 测试套件详情")
        report.append("")
        report.append("| 测试套件 | 状态 | 通过/总数 | 成功率 |")
        report.append("|----------|------|-----------|--------|")
        
        for suite_name, suite_data in self.test_suites.items():
            status = "✅" if suite_data['failed'] == 0 else "❌"
            pass_rate = self._percentage(suite_data['passed'], suite_data['total'])
            report.append(f"| {suite_name} | {status} | {suite_data['passed']}/{suite_data['total']} | {pass_rate}% |")
        
        # 失败测试列表
        if self.failed_tests > 0:
            report.append("")
            report.append("## 失败的测试")
            report.append("")
            for suite_name, suite_data in self.test_suites.items():
                failed_tests = [t for t in suite_data['tests'] if t['status'] == 'FAILED']
                if failed_tests:
                    report.append(f"### {suite_name}")
                    for test in failed_tests:
                        report.append(f"- {test['name']}")
        
        return "\n".join(report)
    
    def _generate_json_report(self):
        """生成JSON报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': self.total_tests,
                'passed': self.passed_tests,
                'failed': self.failed_tests,
                'pass_rate': self._percentage(self.passed_tests, self.total_tests),
                'execution_time_ms': self.execution_time
            },
            'suites': self.test_suites
        }
        return json.dumps(report, indent=2)
    
    def _percentage(self, part, total):
        """计算百分比"""
        if total == 0:
            return 0
        return round((part / total) * 100, 2)


def run_tests_and_summarize(test_command, output_format='text'):
    """运行测试并生成汇总"""
    try:
        # 运行测试命令
        result = subprocess.run(
            test_command,
            shell=True,
            capture_output=True,
            text=True
        )
        
        # 解析输出
        summary = TestSummary()
        summary.parse_gtest_output(result.stdout)
        
        # 生成报告
        report = summary.generate_report(output_format)
        
        return report, result.returncode
        
    except Exception as e:
        return f"错误: {str(e)}", 1


def main():
    parser = argparse.ArgumentParser(description='OpenNPP测试结果汇总工具')
    parser.add_argument('-c', '--command', 
                       help='要运行的测试命令')
    parser.add_argument('-f', '--format', 
                       choices=['text', 'json', 'markdown'],
                       default='text',
                       help='输出格式')
    parser.add_argument('-o', '--output',
                       help='输出文件（默认为标准输出）')
    parser.add_argument('-i', '--input',
                       help='从文件读取GTest输出（而不是运行命令）')
    
    args = parser.parse_args()
    
    if args.input:
        # 从文件读取
        with open(args.input, 'r') as f:
            output = f.read()
        summary = TestSummary()
        summary.parse_gtest_output(output)
        report = summary.generate_report(args.format)
        returncode = 0
    elif args.command:
        # 运行命令
        report, returncode = run_tests_and_summarize(args.command, args.format)
    else:
        # 从标准输入读取
        output = sys.stdin.read()
        summary = TestSummary()
        summary.parse_gtest_output(output)
        report = summary.generate_report(args.format)
        returncode = 0
    
    # 输出报告
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"报告已保存到: {args.output}")
    else:
        print(report)
    
    return returncode


if __name__ == '__main__':
    sys.exit(main())