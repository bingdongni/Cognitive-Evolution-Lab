"""
工具函数测试模块

测试各种工具函数和辅助功能。
"""

import unittest
import pytest
import numpy as np
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
import sys

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    HardwareDetector,
    PerformanceMonitor,
    DataLoader,
    ConfigManager,
    LoggingSystem,
    FileUtils,
    MathUtils,
    NetworkUtils
)


class TestHardwareDetector(unittest.TestCase):
    """硬件检测器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.hardware_detector = HardwareDetector()
        
    def test_system_info_detection(self):
        """测试系统信息检测"""
        system_info = self.hardware_detector.get_system_info()
        
        self.assertIsInstance(system_info, dict)
        self.assertIn("cpu", system_info)
        self.assertIn("memory", system_info)
        self.assertIn("gpu", system_info)
        self.assertIn("platform", system_info)
        
        # 检查CPU信息
        cpu_info = system_info["cpu"]
        self.assertIn("cores", cpu_info)
        self.assertIn("frequency", cpu_info)
        self.assertIn("architecture", cpu_info)
        
        # 检查内存信息
        memory_info = system_info["memory"]
        self.assertIn("total", memory_info)
        self.assertIn("available", memory_info)
        self.assertIn("used", memory_info)
        
    def test_gpu_detection(self):
        """测试GPU检测"""
        gpu_info = self.hardware_detector.detect_gpu()
        
        self.assertIsInstance(gpu_info, dict)
        if gpu_info["available"]:
            self.assertIn("name", gpu_info)
            self.assertIn("memory", gpu_info)
            self.assertIn("compute_capability", gpu_info)
            self.assertIn("cuda_available", gpu_info)
            
    def test_performance_benchmark(self):
        """测试性能基准测试"""
        benchmark_result = self.hardware_detector.run_performance_benchmark()
        
        self.assertIsInstance(benchmark_result, dict)
        self.assertIn("cpu_score", benchmark_result)
        self.assertIn("memory_score", benchmark_result)
        self.assertIn("gpu_score", benchmark_result)
        self.assertIn("overall_score", benchmark_result)
        
        # 分数应该在合理范围内
        self.assertGreaterEqual(benchmark_result["overall_score"], 0)
        self.assertLessEqual(benchmark_result["overall_score"], 100)
        
    def test_optimization_recommendations(self):
        """测试优化建议"""
        recommendations = self.hardware_detector.get_optimization_recommendations()
        
        self.assertIsInstance(recommendations, list)
        for rec in recommendations:
            self.assertIsInstance(rec, dict)
            self.assertIn("category", rec)
            self.assertIn("recommendation", rec)
            self.assertIn("priority", rec)


class TestPerformanceMonitor(unittest.TestCase):
    """性能监控器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.performance_monitor = PerformanceMonitor()
        
    def test_cpu_monitoring(self):
        """测试CPU监控"""
        cpu_usage = self.performance_monitor.get_cpu_usage()
        
        self.assertIsInstance(cpu_usage, dict)
        self.assertIn("usage_percent", cpu_usage)
        self.assertIn("per_core_usage", cpu_usage)
        self.assertIn("load_average", cpu_usage)
        
        # CPU使用率应该在0-100之间
        self.assertGreaterEqual(cpu_usage["usage_percent"], 0)
        self.assertLessEqual(cpu_usage["usage_percent"], 100)
        
    def test_memory_monitoring(self):
        """测试内存监控"""
        memory_usage = self.performance_monitor.get_memory_usage()
        
        self.assertIsInstance(memory_usage, dict)
        self.assertIn("total", memory_usage)
        self.assertIn("used", memory_usage)
        self.assertIn("free", memory_usage)
        self.assertIn("usage_percent", memory_usage)
        
        # 内存使用率应该在0-100之间
        self.assertGreaterEqual(memory_usage["usage_percent"], 0)
        self.assertLessEqual(memory_usage["usage_percent"], 100)
        
    def test_disk_monitoring(self):
        """测试磁盘监控"""
        disk_usage = self.performance_monitor.get_disk_usage()
        
        self.assertIsInstance(disk_usage, dict)
        self.assertIn("total", disk_usage)
        self.assertIn("used", disk_usage)
        self.assertIn("free", disk_usage)
        self.assertIn("usage_percent", disk_usage)
        
    def test_network_monitoring(self):
        """测试网络监控"""
        network_stats = self.performance_monitor.get_network_stats()
        
        self.assertIsInstance(network_stats, dict)
        self.assertIn("bytes_sent", network_stats)
        self.assertIn("bytes_recv", network_stats)
        self.assertIn("packets_sent", network_stats)
        self.assertIn("packets_recv", network_stats)
        
    def test_process_monitoring(self):
        """测试进程监控"""
        # 获取当前进程信息
        current_process = self.performance_monitor.get_process_info()
        
        self.assertIsInstance(current_process, dict)
        self.assertIn("pid", current_process)
        self.assertIn("name", current_process)
        self.assertIn("cpu_percent", current_process)
        self.assertIn("memory_percent", current_process)
        
    def test_timing_decorator(self):
        """测试计时装饰器"""
        
        @self.performance_monitor.timing
        def slow_function():
            time.sleep(0.1)  # 模拟耗时操作
            return "done"
        
        start_time = time.time()
        result = slow_function()
        end_time = time.time()
        
        self.assertEqual(result, "done")
        self.assertGreaterEqual(end_time - start_time, 0.1)
        
        # 检查是否记录了执行时间
        self.assertGreater(len(self.performance_monitor.timing_log), 0)


class TestDataLoader(unittest.TestCase):
    """数据加载器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.data_loader = DataLoader()
        
        # 创建临时测试文件
        self.temp_dir = tempfile.mkdtemp()
        self.test_json_file = os.path.join(self.temp_dir, "test_data.json")
        self.test_csv_file = os.path.join(self.temp_dir, "test_data.csv")
        
        # 创建测试JSON数据
        test_data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
        with open(self.test_json_file, 'w') as f:
            json.dump(test_data, f)
            
        # 创建测试CSV数据
        import csv
        with open(self.test_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["name", "age", "score"])
            writer.writerow(["Alice", 25, 95])
            writer.writerow(["Bob", 30, 87])
            writer.writerow(["Charlie", 35, 92])
        
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_json_loading(self):
        """测试JSON数据加载"""
        loaded_data = self.data_loader.load_json(self.test_json_file)
        
        self.assertIsInstance(loaded_data, dict)
        self.assertEqual(loaded_data["key1"], "value1")
        self.assertEqual(loaded_data["key2"], 42)
        self.assertEqual(loaded_data["key3"], [1, 2, 3])
        
    def test_csv_loading(self):
        """测试CSV数据加载"""
        loaded_data = self.data_loader.load_csv(self.test_csv_file)
        
        self.assertIsInstance(loaded_data, list)
        self.assertEqual(len(loaded_data), 3)  # 3行数据
        self.assertEqual(len(loaded_data[0]), 3)  # 3列
        
        # 检查第一行数据
        first_row = loaded_data[0]
        self.assertIn("name", first_row)
        self.assertIn("age", first_row)
        self.assertIn("score", first_row)
        
    def test_yaml_loading(self):
        """测试YAML数据加载"""
        yaml_file = os.path.join(self.temp_dir, "test_data.yaml")
        with open(yaml_file, 'w') as f:
            f.write("""
key1: value1
key2: 42
key3:
  - item1
  - item2
  - item3
nested:
  subkey1: subvalue1
  subkey2: subvalue2
            """)
            
        loaded_data = self.data_loader.load_yaml(yaml_file)
        
        self.assertIsInstance(loaded_data, dict)
        self.assertEqual(loaded_data["key1"], "value1")
        self.assertEqual(loaded_data["key2"], 42)
        self.assertIn("key3", loaded_data)
        self.assertIn("nested", loaded_data)
        
    def test_image_loading(self):
        """测试图像数据加载"""
        # 创建测试图像文件
        from PIL import Image
        test_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        image_file = os.path.join(self.temp_dir, "test_image.png")
        test_image.save(image_file)
        
        loaded_image = self.data_loader.load_image(image_file)
        
        self.assertIsInstance(loaded_image, np.ndarray)
        self.assertEqual(loaded_image.shape, (100, 100, 3))
        
    def test_data_validation(self):
        """测试数据验证"""
        valid_data = {"required_field": "value", "optional_field": 42}
        invalid_data = {"wrong_field": "value"}  # 缺少required_field
        
        schema = {
            "required_fields": ["required_field"],
            "optional_fields": ["optional_field"],
            "types": {"required_field": str, "optional_field": int}
        }
        
        # 测试有效数据
        self.assertTrue(self.data_loader.validate_data(valid_data, schema))
        
        # 测试无效数据
        self.assertFalse(self.data_loader.validate_data(invalid_data, schema))


class TestConfigManager(unittest.TestCase):
    """配置管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config_manager = ConfigManager()
        
    def test_config_loading(self):
        """测试配置加载"""
        # 测试默认配置加载
        default_config = self.config_manager.load_default_config()
        self.assertIsInstance(default_config, dict)
        
    def test_config_validation(self):
        """测试配置验证"""
        valid_config = {
            "system": {
                "name": "test_system",
                "version": "1.0.0",
                "debug": False
            },
            "performance": {
                "max_workers": 4,
                "memory_limit": "8GB"
            }
        }
        
        config_schema = {
            "system": {
                "name": str,
                "version": str,
                "debug": bool
            },
            "performance": {
                "max_workers": int,
                "memory_limit": str
            }
        }
        
        is_valid = self.config_manager.validate_config(valid_config, config_schema)
        self.assertTrue(is_valid)
        
    def test_config_merge(self):
        """测试配置合并"""
        base_config = {
            "system": {"name": "base", "debug": True},
            "performance": {"max_workers": 2}
        }
        
        override_config = {
            "system": {"debug": False},
            "performance": {"max_workers": 4, "memory_limit": "8GB"}
        }
        
        merged_config = self.config_manager.merge_configs(base_config, override_config)
        
        self.assertEqual(merged_config["system"]["name"], "base")
        self.assertEqual(merged_config["system"]["debug"], False)
        self.assertEqual(merged_config["performance"]["max_workers"], 4)
        self.assertEqual(merged_config["performance"]["memory_limit"], "8GB")


class TestLoggingSystem(unittest.TestCase):
    """日志系统测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.logging_system = LoggingSystem()
        
    def test_log_levels(self):
        """测试日志级别"""
        # 测试不同级别的日志记录
        self.logging_system.debug("This is a debug message")
        self.logging_system.info("This is an info message")
        self.logging_system.warning("This is a warning message")
        self.logging_system.error("This is an error message")
        
        # 检查日志记录
        logs = self.logging_system.get_recent_logs()
        self.assertGreaterEqual(len(logs), 4)
        
    def test_log_formatting(self):
        """测试日志格式化"""
        formatted_log = self.logging_system.format_log(
            level="INFO",
            message="Test message",
            module="test_module",
            function="test_function"
        )
        
        self.assertIsInstance(formatted_log, str)
        self.assertIn("INFO", formatted_log)
        self.assertIn("Test message", formatted_log)
        
    def test_log_filtering(self):
        """测试日志过滤"""
        # 记录不同级别的日志
        for i in range(10):
            if i % 2 == 0:
                self.logging_system.info(f"Info message {i}")
            else:
                self.logging_system.warning(f"Warning message {i}")
                
        # 过滤日志
        info_logs = self.logging_system.filter_logs(level="INFO")
        warning_logs = self.logging_system.filter_logs(level="WARNING")
        
        self.assertEqual(len(info_logs), 5)  # 5条INFO日志
        self.assertEqual(len(warning_logs), 5)  # 5条WARNING日志


class TestFileUtils(unittest.TestCase):
    """文件工具测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.file_utils = FileUtils()
        
    def test_file_operations(self):
        """测试文件操作"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            test_file = f.name
            f.write("test content")
            
        try:
            # 测试文件读取
            content = self.file_utils.read_file(test_file)
            self.assertEqual(content, "test content")
            
            # 测试文件写入
            self.file_utils.write_file(test_file, "new content")
            content = self.file_utils.read_file(test_file)
            self.assertEqual(content, "new content")
            
        finally:
            os.unlink(test_file)
            
    def test_directory_operations(self):
        """测试目录操作"""
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        test_subdir = os.path.join(temp_dir, "subdir")
        
        try:
            # 创建子目录
            self.file_utils.create_directory(test_subdir)
            self.assertTrue(os.path.exists(test_subdir))
            
            # 列出目录内容
            contents = self.file_utils.list_directory(temp_dir)
            self.assertIn("subdir", contents)
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
            
    def test_file_finding(self):
        """测试文件查找"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 创建测试文件
            test_files = [
                "test1.txt",
                "test2.py",
                "test3.json",
                "subdir/test4.txt"
            ]
            
            for file_path in test_files:
                full_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write("test")
                    
            # 查找.txt文件
            txt_files = self.file_utils.find_files(temp_dir, "*.txt")
            self.assertEqual(len(txt_files), 2)
            
            # 查找.py文件
            py_files = self.file_utils.find_files(temp_dir, "*.py")
            self.assertEqual(len(py_files), 1)
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestMathUtils(unittest.TestCase):
    """数学工具测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.math_utils = MathUtils()
        
    def test_vector_operations(self):
        """测试向量操作"""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([4, 5, 6])
        
        # 测试向量加法
        result = self.math_utils.vector_add(vec1, vec2)
        expected = np.array([5, 7, 9])
        np.testing.assert_array_equal(result, expected)
        
        # 测试向量减法
        result = self.math_utils.vector_subtract(vec2, vec1)
        expected = np.array([3, 3, 3])
        np.testing.assert_array_equal(result, expected)
        
        # 测试向量点积
        dot_product = self.math_utils.vector_dot(vec1, vec2)
        expected = 1*4 + 2*5 + 3*6  # 32
        self.assertEqual(dot_product, expected)
        
    def test_matrix_operations(self):
        """测试矩阵操作"""
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.array([[5, 6], [7, 8]])
        
        # 测试矩阵乘法
        result = self.math_utils.matrix_multiply(mat1, mat2)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result, expected)
        
        # 测试矩阵转置
        result = self.math_utils.matrix_transpose(mat1)
        expected = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(result, expected)
        
    def test_statistical_functions(self):
        """测试统计函数"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # 测试均值
        mean = self.math_utils.calculate_mean(data)
        self.assertEqual(mean, 5.5)
        
        # 测试标准差
        std = self.math_utils.calculate_std(data)
        expected_std = np.std(data)
        self.assertAlmostEqual(std, expected_std, places=6)
        
        # 测试中位数
        median = self.math_utils.calculate_median(data)
        self.assertEqual(median, 5.5)
        
    def test_normalization(self):
        """测试归一化函数"""
        data = np.array([1, 2, 3, 4, 5])
        
        # 测试最小-最大归一化
        normalized = self.math_utils.min_max_normalize(data)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(normalized, expected)
        
        # 测试Z-score标准化
        standardized = self.math_utils.z_score_normalize(data)
        expected_mean = 0
        expected_std = 1
        self.assertAlmostEqual(np.mean(standardized), expected_mean, places=6)
        self.assertAlmostEqual(np.std(standardized), expected_std, places=6)


class TestNetworkUtils(unittest.TestCase):
    """网络工具测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.network_utils = NetworkUtils()
        
    def test_url_validation(self):
        """测试URL验证"""
        valid_urls = [
            "https://www.example.com",
            "http://localhost:8080",
            "ftp://files.example.com"
        ]
        
        invalid_urls = [
            "not_a_url",
            "http://",
            "https://"
        ]
        
        # 测试有效URL
        for url in valid_urls:
            self.assertTrue(self.network_utils.is_valid_url(url))
            
        # 测试无效URL
        for url in invalid_urls:
            self.assertFalse(self.network_utils.is_valid_url(url))
            
    def test_ip_address_validation(self):
        """测试IP地址验证"""
        valid_ips = [
            "192.168.1.1",
            "127.0.0.1",
            "8.8.8.8"
        ]
        
        invalid_ips = [
            "999.999.999.999",
            "192.168.1",
            "not_an_ip"
        ]
        
        # 测试有效IP地址
        for ip in valid_ips:
            self.assertTrue(self.network_utils.is_valid_ip(ip))
            
        # 测试无效IP地址
        for ip in invalid_ips:
            self.assertFalse(self.network_utils.is_valid_ip(ip))
            
    def test_network_speed_test(self):
        """测试网络速度测试"""
        # 模拟网络速度测试
        speed_result = self.network_utils.test_network_speed("8.8.8.8")
        
        self.assertIsInstance(speed_result, dict)
        self.assertIn("latency", speed_result)
        self.assertIn("download_speed", speed_result)
        self.assertIn("upload_speed", speed_result)
        
        # 延迟应该大于0
        self.assertGreater(speed_result["latency"], 0)


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)
