#!/usr/bin/env python3
"""
NPP API状态报告生成器

此脚本分析：
1. gtest XML测试结果
2. API特性列表配置
3. 源码实现情况

生成综合的Markdown状态报告
"""

import os
import sys
import yaml
import xml.etree.ElementTree as ET
import glob
import re
from datetime import datetime
from collections import defaultdict
import argparse

class NPPStatusReporter:
    def __init__(self, project_root):
        self.project_root = project_root
        self.api_features = {}
        self.test_results = {}
        self.source_implementations = {}
        self.test_implementations = {}
        
    def load_api_features(self, features_file):
        """加载API特性列表"""
        with open(features_file, 'r', encoding='utf-8') as f:
            self.api_features = yaml.safe_load(f)
        print(f"已加载API特性列表: {len(self.api_features)}个模块")
        
    def parse_gtest_xml(self, xml_file):
        """解析gtest XML结果"""
        if not os.path.exists(xml_file):
            print(f"警告: 测试结果文件不存在: {xml_file}")
            return
            
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for testsuite in root.findall('testsuite'):
            suite_name = testsuite.get('name')
            
            for testcase in testsuite.findall('testcase'):
                case_name = testcase.get('name')
                full_name = f"{suite_name}.{case_name}"
                
                # 检查是否有失败
                failure = testcase.find('failure')
                error = testcase.find('error')
                
                if failure is not None or error is not None:
                    status = 'FAILED'
                    message = failure.text if failure is not None else error.text
                else:
                    status = 'PASSED'
                    message = None
                    
                self.test_results[full_name] = {
                    'status': status,
                    'time': float(testcase.get('time', 0)),
                    'message': message
                }
                
        print(f"已解析测试结果: {len(self.test_results)}个测试用例")
        
    def scan_source_implementations(self):
        """扫描源码实现情况"""
        src_dir = os.path.join(self.project_root, 'src')
        
        # 扫描C++源文件
        cpp_files = glob.glob(os.path.join(src_dir, '**', '*.cpp'), recursive=True)
        cu_files = glob.glob(os.path.join(src_dir, '**', '*.cu'), recursive=True)
        
        all_files = cpp_files + cu_files
        
        for file_path in all_files:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # 查找函数实现（简单的正则匹配）
                # 匹配形如: NppStatus functionName(...) 或 type functionName(...)
                function_patterns = [
                    r'NppStatus\s+(nppi[A-Za-z_0-9]+)\s*\(',
                    r'NppStatus\s+(npps[A-Za-z_0-9]+)\s*\(',
                    r'void\s+(npp[A-Za-z_0-9]+)\s*\(',
                    r'[A-Za-z_0-9\*]+\s+(npp[A-Za-z_0-9]+)\s*\(',
                ]
                
                for pattern in function_patterns:
                    matches = re.findall(pattern, content)
                    for func_name in matches:
                        if func_name not in self.source_implementations:
                            self.source_implementations[func_name] = []
                        self.source_implementations[func_name].append(file_path)
                        
        print(f"已扫描源码实现: {len(self.source_implementations)}个函数")
        
    def scan_test_implementations(self):
        """扫描测试实现情况"""
        test_dir = os.path.join(self.project_root, 'test')
        
        # 扫描测试文件
        test_files = glob.glob(os.path.join(test_dir, '**', 'test_*.cpp'), recursive=True)
        
        for file_path in test_files:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # 查找测试用例中调用的API函数
                # 匹配形如: nppiFunction(...) 或 nppsFunction(...)
                api_calls = re.findall(r'\b(npp[a-zA-Z_0-9]+)\s*\(', content)
                
                for func_name in api_calls:
                    if func_name not in self.test_implementations:
                        self.test_implementations[func_name] = []
                    if file_path not in self.test_implementations[func_name]:
                        self.test_implementations[func_name].append(file_path)
                        
        print(f"已扫描测试实现: {len(self.test_implementations)}个函数")
        
    def analyze_api_status(self):
        """分析API状态"""
        api_status = {}
        
        for module_name, module_info in self.api_features.items():
            api_status[module_name] = {
                'description': module_info.get('description', ''),
                'functions': {}
            }
            
            for func_info in module_info.get('functions', []):
                func_name = func_info['name']
                
                # 检查源码实现
                has_source = func_name in self.source_implementations
                source_files = self.source_implementations.get(func_name, [])
                
                # 检查测试实现
                has_test = func_name in self.test_implementations
                test_files = self.test_implementations.get(func_name, [])
                
                # 检查测试结果
                test_results = []
                test_passed = 0
                test_failed = 0
                
                for test_name, result in self.test_results.items():
                    if func_name.replace('_', '').lower() in test_name.lower().replace('_', ''):
                        test_results.append({
                            'name': test_name,
                            'status': result['status'],
                            'time': result['time']
                        })
                        if result['status'] == 'PASSED':
                            test_passed += 1
                        else:
                            test_failed += 1
                
                api_status[module_name]['functions'][func_name] = {
                    'description': func_info.get('description', ''),
                    'has_source': has_source,
                    'source_files': source_files,
                    'has_test': has_test,
                    'test_files': test_files,
                    'test_results': test_results,
                    'test_passed': test_passed,
                    'test_failed': test_failed
                }
                
        return api_status
        
    def generate_yaml_report(self, api_status, output_file):
        """生成YAML格式状态报告"""
        # 计算总体统计
        total_apis = sum(len(module['functions']) for module in api_status.values())
        implemented_apis = sum(1 for module in api_status.values() 
                             for func in module['functions'].values() 
                             if func['has_source'])
        tested_apis = sum(1 for module in api_status.values() 
                        for func in module['functions'].values() 
                        if func['has_test'])
        
        passed_tests = [name for name, result in self.test_results.items() if result['status'] == 'PASSED']
        failed_tests = [name for name, result in self.test_results.items() if result['status'] == 'FAILED']
        
        # 构建输出数据结构
        report_data = {
            'metadata': {
                'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'generator_version': '1.0.0',
                'project_root': self.project_root
            },
            'summary': {
                'total_apis': total_apis,
                'implemented_apis': implemented_apis,
                'tested_apis': tested_apis,
                'implementation_rate': round(implemented_apis/total_apis*100, 1) if total_apis > 0 else 0.0,
                'test_coverage_rate': round(tested_apis/total_apis*100, 1) if total_apis > 0 else 0.0,
                'total_tests': len(self.test_results),
                'passed_tests': len(passed_tests),
                'failed_tests': len(failed_tests)
            },
            'modules': {}
        }
        
        # 处理每个模块
        for module_name, module_info in api_status.items():
            module_total = len(module_info['functions'])
            module_implemented = sum(1 for func in module_info['functions'].values() if func['has_source'])
            module_tested = sum(1 for func in module_info['functions'].values() if func['has_test'])
            
            module_data = {
                'description': module_info['description'],
                'statistics': {
                    'total_functions': module_total,
                    'implemented_functions': module_implemented,
                    'tested_functions': module_tested,
                    'implementation_rate': round(module_implemented/module_total*100, 1) if module_total > 0 else 0.0,
                    'test_coverage_rate': round(module_tested/module_total*100, 1) if module_total > 0 else 0.0
                },
                'functions': {}
            }
            
            # 处理每个函数
            for func_name, func_info in module_info['functions'].items():
                function_data = {
                    'description': func_info['description'],
                    'implementation': {
                        'status': 'implemented' if func_info['has_source'] else 'not_implemented',
                        'source_files': [os.path.relpath(f, self.project_root) for f in func_info['source_files']]
                    },
                    'testing': {
                        'status': self._get_test_status(func_info),
                        'test_files': [os.path.relpath(f, self.project_root) for f in func_info['test_files']],
                        'test_results': {
                            'passed': func_info['test_passed'],
                            'failed': func_info['test_failed'],
                            'details': func_info['test_results']
                        }
                    }
                }
                
                module_data['functions'][func_name] = function_data
            
            report_data['modules'][module_name] = module_data
        
        # 添加测试结果详情
        if failed_tests:
            report_data['failed_tests'] = []
            for test_name in failed_tests:
                result = self.test_results[test_name]
                report_data['failed_tests'].append({
                    'name': test_name,
                    'message': result['message'][:200] if result['message'] else 'No message',
                    'time': result['time']
                })
        
        # 写入YAML文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# NPP API实现状态报告\n")
            f.write(f"# 生成时间: {report_data['metadata']['generated_time']}\n\n")
            yaml.dump(report_data, f, default_flow_style=False, allow_unicode=True, indent=2)
                            
        print(f"状态报告已生成: {output_file}")
    
    def _get_test_status(self, func_info):
        """获取测试状态"""
        if not func_info['has_test']:
            return 'not_tested'
        elif func_info['test_passed'] > 0 and func_info['test_failed'] == 0:
            return 'fully_tested'
        elif func_info['test_failed'] > 0:
            return 'test_failed'
        else:
            return 'partially_tested'

def main():
    parser = argparse.ArgumentParser(description='生成NPP API状态报告')
    parser.add_argument('--project-root', default='.', help='项目根目录')
    parser.add_argument('--features-file', default='status/api_features.yaml', help='API特性配置文件')
    parser.add_argument('--xml-results', default='test_results.xml', help='gtest XML结果文件')
    parser.add_argument('--output', default='status/API_STATUS.yaml', help='输出的YAML文件')
    
    args = parser.parse_args()
    
    # 确保路径是绝对路径
    project_root = os.path.abspath(args.project_root)
    features_file = os.path.join(project_root, args.features_file)
    xml_results = os.path.join(project_root, args.xml_results)
    output_file = os.path.join(project_root, args.output)
    
    print(f"NPP API状态报告生成器")
    print(f"项目根目录: {project_root}")
    print(f"特性配置: {features_file}")
    print(f"测试结果: {xml_results}")
    print(f"输出文件: {output_file}")
    print("-" * 50)
    
    # 创建报告生成器
    reporter = NPPStatusReporter(project_root)
    
    # 加载配置和数据
    if os.path.exists(features_file):
        reporter.load_api_features(features_file)
    else:
        print(f"错误: 特性配置文件不存在: {features_file}")
        return 1
        
    # 解析测试结果
    reporter.parse_gtest_xml(xml_results)
    
    # 扫描源码
    reporter.scan_source_implementations()
    reporter.scan_test_implementations()
    
    # 分析状态
    api_status = reporter.analyze_api_status()
    
    # 生成报告
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    reporter.generate_yaml_report(api_status, output_file)
    
    print("✅ 状态报告生成完成!")
    return 0

if __name__ == '__main__':
    sys.exit(main())