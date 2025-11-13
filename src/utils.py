#!/usr/bin/env python3
"""
Cognitive Evolution Lab - 工具函数
作者: bingdongni

提供项目所需的通用工具函数，包括：
- 配置加载和验证
- 环境检测和验证
- 硬件检测
- 日志设置
- 数据处理和格式化
- 性能监控
- 文件管理
"""

import os
import sys
import json
import yaml
import logging
import logging.handlers
import psutil
import platform
import GPUtil
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime
import asyncio
import traceback

# 硬件检测类
class HardwareDetector:
    """硬件检测器"""
    
    def __init__(self):
        self.hardware_info = self._detect_hardware()
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """检测硬件信息"""
        hardware_info = {
            'cpu': {
                'cores': psutil.cpu_count(logical=True),
                'physical_cores': psutil.cpu_count(logical=False),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                'architecture': platform.machine()
            },
            'memory': {},
            'gpu': {},
            'storage': {},
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            }
        }
        
        # 内存信息
        memory = psutil.virtual_memory()
        hardware_info['memory'] = {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'used': memory.used,
            'free': memory.free
        }
        
        # GPU信息
        try:
            gpus = GPUtil.getGPUs()
            hardware_info['gpu'] = {
                'count': len(gpus),
                'gpus': [
                    {
                        'id': gpu.id,
                        'name': gpu.name,
                        'memory': {
                            'total': gpu.memoryTotal,
                            'free': gpu.memoryFree,
                            'used': gpu.memoryUsed
                        },
                        'load': gpu.load * 100,
                        'temperature': gpu.temperature,
                        'driver': gpu.driver
                    }
                    for gpu in gpus
                ]
            }
        except Exception as e:
            hardware_info['gpu'] = {
                'count': 0,
                'error': str(e)
            }
        
        # 存储信息
        disk_usage = psutil.disk_usage('/')
        hardware_info['storage'] = {
            'total': disk_usage.total,
            'used': disk_usage.used,
            'free': disk_usage.free,
            'percent': (disk_usage.used / disk_usage.total) * 100
        }
        
        return hardware_info
    
    def get_summary(self) -> str:
        """获取硬件摘要"""
        cpu_info = f"{self.hardware_info['cpu']['cores']}核CPU"
        memory_gb = self.hardware_info['memory']['total'] / (1024**3)
        gpu_count = self.hardware_info['gpu']['count']
        
        summary = f"{cpu_info}, {memory_gb:.1f}GB内存"
        
        if gpu_count > 0:
            gpu_info = self.hardware_info['gpu']['gpus'][0]['name']
            summary += f", {gpu_count}x {gpu_info}"
        
        return summary
    
    def get_performance_tier(self) -> str:
        """获取性能等级"""
        cpu_cores = self.hardware_info['cpu']['cores']
        memory_gb = self.hardware_info['memory']['total'] / (1024**3)
        gpu_count = self.hardware_info['gpu']['count']
        
        # 简单的性能分级
        if cpu_cores >= 16 and memory_gb >= 32 and gpu_count >= 1:
            return "ultra_high"
        elif cpu_cores >= 8 and memory_gb >= 16:
            return "high"
        elif cpu_cores >= 4 and memory_gb >= 8:
            return "mid_range"
        else:
            return "low_end"
    
    def is_gpu_available(self) -> bool:
        """检查GPU是否可用"""
        return self.hardware_info['gpu']['count'] > 0 and torch.cuda.is_available()
    
    def get_memory_gb(self) -> float:
        """获取内存大小（GB）"""
        return self.hardware_info['memory']['total'] / (1024**3)


def setup_logging(config: Dict[str, Any] = None):
    """设置日志系统"""
    if config is None:
        config = {'global': {}}
    
    log_config = config.get('global', {})
    log_level = getattr(logging, log_config.get('log_level', 'INFO').upper())
    debug_mode = log_config.get('debug', False)
    
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 创建根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # 文件处理器（普通日志）
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "cognitive_lab.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)
    
    # 错误日志处理器
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=10*1024*1024,
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(error_handler)
    
    # 性能日志处理器（如果开启）
    if debug_mode:
        perf_handler = logging.handlers.RotatingFileHandler(
            log_dir / "performance.log",
            maxBytes=10*1024*1024,
            backupCount=3
        )
        perf_handler.setLevel(logging.DEBUG)
        perf_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(perf_handler)
    
    # 警告设置
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # PyTorch日志抑制
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载配置文件"""
    if config_path is None:
        # 默认配置文件路径
        config_path = "config/config.yaml"
    
    config_file = Path(config_path)
    
    # 如果配置文件不存在，创建默认配置
    if not config_file.exists():
        logging.warning(f"配置文件 {config_path} 不存在，使用默认配置")
        return get_default_config()
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path}")
        
        # 加载环境特定配置
        environment_config = load_environment_config()
        config = merge_configs(config, environment_config)
        
        return config
        
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        return get_default_config()


def load_environment_config() -> Dict[str, Any]:
    """加载环境特定配置"""
    env_config_path = "config/environment_config.yaml"
    
    if not Path(env_config_path).exists():
        return {}
    
    try:
        with open(env_config_path, 'r', encoding='utf-8') as f:
            env_config = yaml.safe_load(f)
        
        # 根据当前平台选择配置
        platform_name = platform.system().lower()
        if platform_name in env_config:
            platform_config = env_config[platform_name]
            # 合并到环境配置中
            env_config['current_platform'] = platform_config
        
        return env_config
        
    except Exception as e:
        logging.error(f"加载环境配置失败: {e}")
        return {}


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        'global': {
            'version': '1.0.0',
            'debug': False,
            'log_level': 'INFO',
            'save_path': './results',
            'random_seed': 42,
            'device': 'auto'
        },
        'world_simulator': {
            'default_world': 'hybrid_world',
            'world_bounds': [[-100, 100], [-100, 100], [0, 50]],
            'timestep': 0.01,
            'gravity': 9.81,
            'social_认知主体s': 50,
            'physics_engine': 'bullet',
            'game_environments': ['CartPole-v1'],
            'real_data_sources': {
                'stock_data': False,
                'social_media': False,
                'weather': False
            }
        },
        'cognitive_models': {
            'vocab_size': 10000,
            'embed_dim': 512,
            'hidden_dim': 768,
            'memory': {
                'type': 'hierarchical',
                'capacity': 10000,
                'episodic_weight': 0.3,
                'semantic_weight': 0.5,
                'procedural_weight': 0.2
            },
            'attention': {
                'type': 'transformer',
                'heads': 8,
                'dropout': 0.1,
                'dynamic_allocation': True
            },
            'reasoning': {
                'type': 'neuro_symbolic',
                'symbolic_ratio': 0.4,
                'neural_ratio': 0.6,
                'logic_depth': 5
            }
        },
        'interactive_systems': {
            'multimodal_perception': {
                'vision': {
                    'enabled': True,
                    'resolution': [224, 224]
                },
                'audio': {
                    'enabled': True,
                    'sample_rate': 16000
                },
                'touch': {
                    'enabled': True,
                    'sensor_count': 100
                }
            },
            'embodied_intelligence': {
                'body_model': 'humanoid',
                'motor_control': 'policy_gradient',
                'sensory_fusion': 'kalman_filter',
                'balance_control': True
            }
        },
        'evolution_engine': {
            'population_size': 100,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elitism_rate': 0.1,
            'evolution_type': 'multi_认知主体',
            'genetic_config': {
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'elitism_rate': 0.1,
                'diversity_threshold': 0.1
            },
            'multi_认知主体_config': {
                'cooperation_threshold': 0.7,
                'competition_weight': 0.3,
                'communication_enabled': True,
                'coalition_formation': True
            }
        },
        'visualization': {
            'render_3d': {
                'engine': 'pyglet',
                'resolution': [1920, 1080],
                'fps': 60,
                'lighting': True,
                'shadows': True
            },
            'dashboard': {
                'framework': 'dash',
                'refresh_rate': 1.0,
                'charts': ['line', 'bar', 'heatmap', 'network']
            },
            'monitoring': {
                'brain_activity': True,
                'learning_curves': True,
                'evolution_progress': True
            }
        },
        'performance': {
            'parallel_processing': {
                'cpu_cores': 'auto',
                'gpu_devices': 'auto',
                'distributed': False
            },
            'memory_management': {
                'cache_size': '1GB',
                'garbage_collection': True,
                'memory_monitoring': True
            }
        }
    }


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """合并配置字典"""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_environment() -> Dict[str, Any]:
    """验证运行环境"""
    validation_results = {
        'python_version': {},
        'dependencies': {},
        'hardware': {},
        'permissions': {},
        'recommendations': []
    }
    
    # Python版本检查
    python_version = sys.version_info
    validation_results['python_version'] = {
        'current': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
        'required': '3.9+',
        'compatible': python_version >= (3, 9)
    }
    
    if not validation_results['python_version']['compatible']:
        validation_results['recommendations'].append("需要Python 3.9或更高版本")
    
    # 依赖检查
    required_packages = [
        'numpy', 'torch', 'pygame', 'yaml', 'psutil'
    ]
    
    optional_packages = [
        'transformers', 'cv2', 'plotly', 'dash', 'networkx'
    ]
    
    validation_results['dependencies'] = {
        'required': {},
        'optional': {},
        'missing_required': [],
        'missing_optional': []
    }
    
    for package in required_packages:
        try:
            __import__(package)
            validation_results['dependencies']['required'][package] = True
        except ImportError:
            validation_results['dependencies']['required'][package] = False
            validation_results['dependencies']['missing_required'].append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
            validation_results['dependencies']['optional'][package] = True
        except ImportError:
            validation_results['dependencies']['optional'][package] = False
            validation_results['dependencies']['missing_optional'].append(package)
    
    if validation_results['dependencies']['missing_required']:
        validation_results['recommendations'].append("安装缺失的必需依赖包")
    
    # 硬件检查
    hardware_detector = HardwareDetector()
    hardware_info = hardware_detector.hardware_info
    
    memory_gb = hardware_detector.get_memory_gb()
    gpu_available = hardware_detector.is_gpu_available()
    
    validation_results['hardware'] = {
        'memory_gb': memory_gb,
        'gpu_available': gpu_available,
        'performance_tier': hardware_detector.get_performance_tier(),
        'meets_minimum_requirements': memory_gb >= 4.0
    }
    
    if memory_gb < 8.0:
        validation_results['recommendations'].append("建议至少8GB内存以获得最佳性能")
    
    if not gpu_available:
        validation_results['recommendations'].append("GPU加速不可用，将使用CPU模式")
    
    # 权限检查
    writable_dirs = ['./results', './logs', './data', './models']
    validation_results['permissions'] = {
        'writable_dirs': {},
        'unwritable_dirs': []
    }
    
    for dir_path in writable_dirs:
        dir_path = Path(dir_path)
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            test_file = dir_path / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            validation_results['permissions']['writable_dirs'][str(dir_path)] = True
        except Exception:
            validation_results['permissions']['writable_dirs'][str(dir_path)] = False
            validation_results['permissions']['unwritable_dirs'].append(str(dir_path))
    
    if validation_results['permissions']['unwritable_dirs']:
        validation_results['recommendations'].append("某些目录无法写入，可能影响数据保存")
    
    return validation_results


def check_dependencies(install_missing: bool = False) -> Dict[str, Any]:
    """检查依赖包"""
    required_packages = {
        'numpy': 'numpy',
        'torch': 'torch',
        'pygame': 'pygame',
        'yaml': 'pyyaml',
        'psutil': 'psutil',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib'
    }
    
    optional_packages = {
        'transformers': 'transformers',
        'cv2': 'opencv-python',
        'plotly': 'plotly',
        'dash': 'dash',
        'networkx': 'networkx',
        'PIL': 'Pillow',
        'librosa': 'librosa'
    }
    
    results = {
        'required': {},
        'optional': {},
        'installation_needed': []
    }
    
    # 检查必需包
    for package_name, pip_name in required_packages.items():
        try:
            __import__(package_name)
            results['required'][package_name] = True
        except ImportError:
            results['required'][package_name] = False
            results['installation_needed'].append(pip_name)
    
    # 检查可选包
    for package_name, pip_name in optional_packages.items():
        try:
            __import__(package_name)
            results['optional'][package_name] = True
        except ImportError:
            results['optional'][package_name] = False
    
    # 自动安装（如果启用）
    if install_missing and results['installation_needed']:
        logging.info("开始自动安装缺失的依赖包...")
        
        for package in results['installation_needed']:
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', package
                ])
                logging.info(f"✅ 成功安装 {package}")
            except subprocess.CalledProcessError as e:
                logging.error(f"❌ 安装 {package} 失败: {e}")
    
    return results


def get_device_config(preferred_device: str = 'auto') -> str:
    """获取设备配置"""
    if preferred_device == 'auto':
        # 自动检测最佳设备
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                # 选择内存最充足的GPU
                max_memory = 0
                best_device = 0
                
                for i in range(device_count):
                    memory = torch.cuda.get_device_properties(i).total_memory
                    if memory > max_memory:
                        max_memory = memory
                        best_device = i
                
                return f"cuda:{best_device}"
            else:
                return "cpu"
        else:
            return "cpu"
    
    return preferred_device


def format_bytes(bytes_value: int) -> str:
    """格式化字节数"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}小时"
    else:
        days = seconds / 86400
        return f"{days:.1f}天"


def safe_import(module_name: str, fallback: Any = None) -> Any:
    """安全导入模块"""
    try:
        return __import__(module_name)
    except ImportError:
        logging.warning(f"无法导入模块 {module_name}，使用fallback")
        return fallback


def validate_file_path(file_path: str, must_exist: bool = False) -> bool:
    """验证文件路径"""
    path = Path(file_path)
    
    if must_exist:
        return path.exists() and path.is_file()
    else:
        # 检查父目录是否存在且可写
        parent = path.parent
        return parent.exists() and parent.is_dir()


def create_directory_if_not_exists(dir_path: str) -> bool:
    """创建目录（如果不存在）"""
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"创建目录失败 {dir_path}: {e}")
        return False


def get_system_info() -> Dict[str, Any]:
    """获取系统信息"""
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'disk_usage': psutil.disk_usage('/').percent,
        'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
        'process_count': len(psutil.pids())
    }


def performance_monitor(func):
    """性能监控装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            logging.debug(f"函数 {func.__name__} 执行时间: {execution_time:.3f}s, "
                         f"内存使用: {format_bytes(memory_usage)}")
    
    return wrapper


def error_handler(func):
    """错误处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"函数 {func.__name__} 执行失败: {e}")
            logging.error(f"错误详情: {traceback.format_exc()}")
            return None
    
    return wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logging.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
                        time.sleep(delay)
                    else:
                        logging.error(f"函数 {func.__name__} 重试 {max_retries} 次后仍然失败")
            
            raise last_exception
        
        return wrapper
    return decorator


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total_steps: int, description: str = "进度"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, step_increment: int = 1):
        """更新进度"""
        self.current_step = min(self.current_step + step_increment, self.total_steps)
        self._display_progress()
    
    def _display_progress(self):
        """显示进度"""
        if self.total_steps > 0:
            percentage = (self.current_step / self.total_steps) * 100
            elapsed_time = time.time() - self.start_time
            
            if self.current_step > 0:
                estimated_total_time = elapsed_time * self.total_steps / self.current_step
                remaining_time = estimated_total_time - elapsed_time
                remaining_str = f", 剩余时间: {format_time(remaining_time)}"
            else:
                remaining_str = ""
            
            print(f"\r{self.description}: {self.current_step}/{self.total_steps} "
                  f"({percentage:.1f}%) - 已用时间: {format_time(elapsed_time)}{remaining_str}", 
                  end="", flush=True)
    
    def finish(self):
        """完成进度"""
        total_time = time.time() - self.start_time
        print(f"\n{self.description}完成! 总用时: {format_time(total_time)}")


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.profiles = {}
        self.start_times = {}
    
    def start(self, name: str):
        """开始分析"""
        self.start_times[name] = time.time()
    
    def end(self, name: str):
        """结束分析"""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            
            if name not in self.profiles:
                self.profiles[name] = []
            
            self.profiles[name].append(duration)
            del self.start_times[name]
            
            return duration
        
        return 0.0
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """获取统计信息"""
        if name not in self.profiles:
            return {}
        
        durations = self.profiles[name]
        return {
            'count': len(durations),
            'total': sum(durations),
            'average': np.mean(durations),
            'min': min(durations),
            'max': max(durations),
            'std': np.std(durations)
        }
    
    def print_summary(self):
        """打印摘要"""
        print("\n性能分析摘要:")
        print("-" * 50)
        
        for name, durations in self.profiles.items():
            stats = self.get_stats(name)
            print(f"{name}:")
            print(f"  调用次数: {stats['count']}")
            print(f"  总时间: {format_time(stats['total'])}")
            print(f"  平均时间: {format_time(stats['average'])}")
            print(f"  最小时间: {format_time(stats['min'])}")
            print(f"  最大时间: {format_time(stats['max'])}")
            print()


# 认知指标类
class CognitiveMetrics:
    """认知指标计算器"""
    
    @staticmethod
    def memory_retention_score(memories: List[Dict[str, Any]]) -> float:
        """计算记忆保留分数"""
        if not memories:
            return 0.0
        
        retention_scores = []
        for memory in memories:
            strength = memory.get('strength', 0.5)
            age = memory.get('age', 0)
            
            # 简化的遗忘曲线
            retention = strength * math.exp(-age / 100.0)
            retention_scores.append(retention)
        
        return np.mean(retention_scores)
    
    @staticmethod
    def reasoning_accuracy(reasoning_chains: List[Dict[str, Any]]) -> float:
        """计算推理准确率"""
        if not reasoning_chains:
            return 0.0
        
        accuracies = []
        for chain in reasoning_chains:
            confidence = chain.get('confidence', 0.5)
            # 将置信度作为准确率近似
            accuracies.append(confidence)
        
        return np.mean(accuracies)
    
    @staticmethod
    def creativity_score(creative_outputs: List[Dict[str, Any]]) -> float:
        """计算创造力分数"""
        if not creative_outputs:
            return 0.0
        
        scores = []
        for output in creative_outputs:
            creativity_score = output.get('creativity_score', 0.5)
            scores.append(creativity_score)
        
        return np.mean(scores)
    
    @staticmethod
    def attention_focus_score(attention_weights: Dict[str, float]) -> float:
        """计算注意力集中度"""
        if not attention_weights:
            return 0.0
        
        weights = list(attention_weights.values())
        return np.std(weights)  # 标准差越大表示越集中


class EvolutionMetrics:
    """进化指标计算器"""
    
    @staticmethod
    def diversity_score(population: List[Dict[str, Any]]) -> float:
        """计算种群多样性"""
        if len(population) < 2:
            return 1.0
        
        diversity_values = []
        
        # 计算基因多样性
        for individual in population:
            genome = individual.get('genome', {})
            
            # 认知参数多样性
            cog_params = genome.get('cognitive_parameters', {})
            cog_values = list(cog_params.values())
            
            # 行为参数多样性
            beh_params = genome.get('behavioral_parameters', {})
            beh_values = list(beh_params.values())
            
            # 计算个体内部多样性
            all_values = cog_values + beh_values
            if len(all_values) > 1:
                individual_diversity = np.std(all_values)
                diversity_values.append(individual_diversity)
        
        return np.mean(diversity_values) if diversity_values else 0.0
    
    @staticmethod
    def convergence_rate(fitness_history: List[float]) -> float:
        """计算收敛速度"""
        if len(fitness_history) < 2:
            return 0.0
        
        # 计算适应度变化率
        changes = []
        for i in range(1, len(fitness_history)):
            change = abs(fitness_history[i] - fitness_history[i-1])
            changes.append(change)
        
        # 收敛速度 = 最后阶段的变化率
        if len(changes) > 10:
            recent_changes = changes[-10:]
            return np.mean(recent_changes)
        else:
            return np.mean(changes) if changes else 0.0
    
    @staticmethod
    def adaptation_efficiency(population: List[Dict[str, Any]], environment_changes: List[Dict[str, Any]]) -> float:
        """计算适应效率"""
        if not environment_changes:
            return 1.0
        
        adaptation_scores = []
        
        for change in environment_changes:
            complexity_change = change.get('complexity_level', 1.0)
            adaptation_pressure = change.get('adaptive_pressure', 0.0)
            
            # 适应效率 = 适应压力下的性能保持
            efficiency = 1.0 / (1.0 + adaptation_pressure * complexity_change)
            adaptation_scores.append(efficiency)
        
        return np.mean(adaptation_scores)


class VisualizationUtils:
    """可视化工具类"""
    
    @staticmethod
    def create_heatmap_data(data: np.ndarray, labels: List[str] = None) -> Dict[str, Any]:
        """创建热图数据"""
        if labels is None:
            labels = [f"Row_{i}" for i in range(data.shape[0])]
        
        heatmap_data = {
            'z': data.tolist(),
            'x': labels if len(labels) == data.shape[1] else list(range(data.shape[1])),
            'y': labels if len(labels) == data.shape[0] else list(range(data.shape[0])),
            'colorscale': 'RdYlBu_r'
        }
        
        return heatmap_data
    
    @staticmethod
    def create_network_data(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建网络图数据"""
        # 节点数据
        node_trace = {
            'x': [node.get('x', 0) for node in nodes],
            'y': [node.get('y', 0) for node in nodes],
            'text': [node.get('label', f'Node_{i}') for i, node in enumerate(nodes)],
            'mode': 'markers+text',
            'marker': {
                'size': [node.get('size', 10) for node in nodes],
                'color': [node.get('color', 0) for node in nodes]
            }
        }
        
        # 边数据
        edge_x = []
        edge_y = []
        
        for edge in edges:
            source = edge.get('source', 0)
            target = edge.get('target', 1)
            
            if source < len(nodes) and target < len(nodes):
                edge_x.extend([node_trace['x'][source], node_trace['x'][target], None])
                edge_y.extend([node_trace['y'][source], node_trace['y'][target], None])
        
        edge_trace = {
            'x': edge_x,
            'y': edge_y,
            'mode': 'lines',
            'line': {'width': 1, 'color': 'lightgray'}
        }
        
        return {
            'data': [edge_trace, node_trace],
            'layout': {
                'showlegend': False,
                'hovermode': 'closest'
            }
        }
    
    @staticmethod
    def create_time_series_data(timestamps: List[float], values: List[float], 
                               labels: List[str] = None) -> Dict[str, Any]:
        """创建时间序列数据"""
        if labels is None:
            labels = ['Series']
        
        traces = []
        
        for i, label in enumerate(labels):
            if i < len(values):
                trace = {
                    'x': timestamps,
                    'y': values[i] if isinstance(values[i], list) else [values[i]] * len(timestamps),
                    'mode': 'lines',
                    'name': label
                }
                traces.append(trace)
        
        return {
            'data': traces,
            'layout': {
                'title': 'Time Series',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Value'}
            }
        }


# 全局实例
hardware_detector = HardwareDetector()
performance_profiler = PerformanceProfiler()

# 便捷函数
def get_gpu_info() -> Optional[Dict[str, Any]]:
    """获取GPU信息"""
    return hardware_detector.hardware_info.get('gpu', {})

def get_cpu_info() -> Dict[str, Any]:
    """获取CPU信息"""
    return hardware_detector.hardware_info.get('cpu', {})

def get_memory_info() -> Dict[str, Any]:
    """获取内存信息"""
    return hardware_detector.hardware_info.get('memory', {})

def is_windows() -> bool:
    """检查是否为Windows系统"""
    return platform.system().lower() == 'windows'

def is_linux() -> bool:
    """检查是否为Linux系统"""
    return platform.system().lower() == 'linux'

def is_macos() -> bool:
    """检查是否为macOS系统"""
    return platform.system().lower() == 'darwin'

# 导入时间模块（在文件末尾避免循环导入）
import time