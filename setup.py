#!/usr/bin/env python
"""
Cognitive Evolution Lab - 安装脚本
作者: bingdongni
版本: v1.0.0
"""

import os
import sys
from setuptools import setup, find_packages
from pathlib import Path

# 读取README文件
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# 读取依赖
requirements = []
if Path("requirements.txt").exists():
    with open("requirements.txt", "r", encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="cognitive-evolution-lab",
    version="1.0.0",
    author="bingdongni",
    author_email="cognitive.evolution.lab@example.com",
    description="集成前沿认知计算技术的综合性协同进化实验平台",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bingdongni/Cognitive-Evolution-Lab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial General Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.1.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=0.950',
        ],
        'docs': [
            'sphinx>=5.1.0',
            'sphinx-rtd-theme>=1.0.0',
            'sphinx-autodoc-typehints>=1.12.0',
        ],
        'quantum': [
            'qiskit>=0.40.0',
            'cirq>=1.1.0',
        ],
        'gpu': [
            'torch>=1.12.0',
            'torchvision>=0.13.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'celab=src.main:main',
            'celab-cognitive=src.experiments.cognitive_test:main',
            'celab-evolution=src.experiments.multi_认知主体_evolution:main',
            'celab-visualize=src.visualization.dashboard:main',
        ],
    },
    include_package_data=True,
    package_data={
        'config': ['*.yaml', '*.json'],
        'data': ['*.pkl', '*.json', '*.csv'],
        'models': ['*.pt', '*.h5', '*.pb'],
        'experiments': ['*.py'],
    },
    zip_safe=False,
    keywords=[
        "认知计算", "机器学习", "深度学习", "强化学习", "认知科学",
        "协同进化", "类脑计算", "具身智能", "多模态", "神经符号"
    ],
    project_urls={
        "Bug Reports": "https://github.com/bingdongni/Cognitive-Evolution-Lab/issues",
        "Source": "https://github.com/bingdongni/Cognitive-Evolution-Lab",
        "Documentation": "https://cognitive-evolution-lab.readthedocs.io/",
    },
)

# 安装后验证脚本
def verify_installation():
    """验证安装是否成功"""
    try:
        import torch
        import numpy
        import gym
        print("✅ 核心依赖验证成功!")
        
        # 验证基本功能
        from src.world_simulator import VirtualWorld
        from src.cognitive_models import CognitiveAgent
        from src.evolution_engine import EvolutionEngine
        
        print("✅ 主要模块导入成功!")
        print("🎉 Cognitive Evolution Lab 安装完成!")
        return True
        
    except ImportError as e:
        print(f"❌ 安装验证失败: {e}")
        print("请检查依赖安装是否正确")
        return False

if __name__ == "__main__":
    print("🚀 开始安装 Cognitive Evolution Lab...")
    
    # 检查Python版本
    if sys.version_info < (3, 9):
        print("❌ 需要Python 3.9或更高版本")
        sys.exit(1)
    
    # 检查是否在虚拟环境中
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ 检测到虚拟环境")
    else:
        print("⚠️  建议在虚拟环境中安装")
    
    # 执行安装
    setup()
    
    # 验证安装
    print("\n🔍 验证安装...")
    if verify_installation():
        print("\n📚 使用说明:")
        print("  启动主程序: python src/main.py")
        print("  认知实验:   python src/main.py --experiment=cognitive")
        print("  进化实验:   python src/main.py --experiment=evolution")
        print("  可视化:     python src/main.py --mode=dashboard")
    else:
        print("\n❌ 安装验证失败，请检查错误信息")
