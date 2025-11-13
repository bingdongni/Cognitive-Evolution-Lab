#!/usr/bin/env python3
"""
Cognitive Evolution Lab - 源代码包初始化
作者: bingdongni

这是一个集成前沿认知计算技术的综合性协同进化实验平台的核心代码包。
实现了外部世界-内部心智-交互行动相结合的综合模型。
"""

__version__ = "1.0.0"
__author__ = "bingdongni"
__email__ = "cognitive.evolution.lab@example.com"
__license__ = "MIT"
__description__ = "集成前沿认知计算技术的综合性协同进化实验平台"

# 核心模块导入
from .world_simulator import VirtualWorld
from .cognitive_models import CognitiveAgent
from .interactive_systems import EmbodiedIntelligence
from .evolution_engine import EvolutionEngine
from .visualization import LabDashboard

# 工具函数导入
from .utils import (
    setup_logging,
    load_config,
    validate_environment,
    HardwareDetector,
    CognitiveMetrics,
    EvolutionMetrics,
    VisualizationUtils
)

# 实验脚本导入
from .experiments import (
    CognitiveTest,
    MultiAgentEvolution,
    LifelongLearning,
    IntegratedExperiment
)

__all__ = [
    # 核心模块
    'VirtualWorld',
    'CognitiveAgent', 
    'EmbodiedIntelligence',
    'EvolutionEngine',
    'LabDashboard',
    
    # 工具函数
    'setup_logging',
    'load_config',
    'validate_environment',
    'HardwareDetector',
    'CognitiveMetrics',
    'EvolutionMetrics',
    'VisualizationUtils',
    
    # 实验脚本
    'CognitiveTest',
    'MultiAgentEvolution',
    'LifelongLearning',
    'IntegratedExperiment'
]

# 版本信息
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable',
    'build': '20231113'
}

def get_version():
    """获取版本字符串"""
    return f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"

def get_full_version():
    """获取完整版本信息"""
    return f"{get_version()}-{VERSION_INFO['release']}-{VERSION_INFO['build']}"

# 快速访问函数
def create_lab(config_path=None):
    """
    快速创建Cognitive Evolution Lab实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        CognitiveEvolutionLab实例
    """
    from .main import CognitiveEvolutionLab
    return CognitiveEvolutionLab(config_path)

def run_demo():
    """运行演示模式"""
    import asyncio
    from .main import main
    
    print("🎯 启动演示模式...")
    asyncio.run(main())

def get_cognitive_capabilities():
    """获取认知能力列表"""
    return [
        'memory',
        'reasoning', 
        'creativity',
        'observation',
        'attention',
        'imagination'
    ]

def get_evolution_types():
    """获取进化类型列表"""
    return [
        'single_认知主体',
        'multi_认知主体',
        'co_evolution',
        'cultural'
    ]

def get_world_types():
    """获取世界类型列表"""
    return [
        'physics_world',
        'social_world', 
        'game_world',
        'data_world',
        'hybrid_world'
    ]

# 性能优化配置
PERFORMANCE_CONFIG = {
    'enable_gpu_acceleration': True,
    'memory_optimization': True,
    'parallel_processing': True,
    'model_cache': True,
    'progressive_loading': True
}

# 默认配置
DEFAULT_CONFIG = {
    'device': 'auto',
    'precision': 'float32',
    'batch_size': 32,
    'learning_rate': 0.001,
    'max_workers': 4,
    'cache_size': '1GB'
}

# 初始化日志
import logging
logger = logging.getLogger(__name__)
logger.info(f"🚀 Cognitive Evolution Lab v{get_version()} 初始化完成")
logger.info("🧠 认知计算 | 🧬 协同进化 | 🌐 多模态感知 | 🤖 具身智能")
