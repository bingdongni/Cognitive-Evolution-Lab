# Cognitive Evolution Lab - 完整安装配置指南

**作者**: bingdongni  
**版本**: v1.0.0  
**更新时间**: 2025-11-13

> 🎯 本指南提供从零开始的完整安装和配置教程，确保即使是零基础用户也能成功运行Cognitive Evolution Lab项目。

---

## 📋 目录

- [系统要求](#-系统要求)
- [快速开始](#-快速开始)
- [详细安装步骤](#-详细安装步骤)
- [环境配置](#-环境配置)
- [依赖包安装](#-依赖包安装)
- [工具配置](#-工具配置)
- [模拟器和游戏系统](#-模拟器和游戏系统)
- [验证安装](#-验证安装)
- [故障排除](#-故障排除)
- [高级配置](#-高级配置)

---

## 🖥️ 系统要求

### 最低配置（基础运行）
- **操作系统**: Windows 11 (推荐) / Windows 10
- **处理器**: Intel Core i5 或 AMD Ryzen 5 (4核心以上)
- **内存**: 8GB RAM
- **存储**: 10GB 可用空间
- **显卡**: 集成显卡（支持基本图形渲染）

### 推荐配置（最佳性能）
- **操作系统**: Windows 11
- **处理器**: Intel Core i7 或 AMD Ryzen 7 (8核心以上)
- **内存**: 16GB RAM 或更高
- **存储**: 50GB 可用空间（SSD推荐）
- **显卡**: NVIDIA GTX 1660 或更高（支持CUDA加速）

### 理想配置（专业研究）
- **操作系统**: Windows 11
- **处理器**: Intel Core i9 或 AMD Ryzen 9 (16核心以上)
- **内存**: 32GB RAM 或更高
- **存储**: 100GB 可用空间（NVMe SSD）
- **显卡**: NVIDIA RTX 3080 或更高（支持CUDA 11.8+）

---

## ⚡ 快速开始

如果您的系统配置较高且希望快速体验项目，请使用以下一键安装命令：

```bash
# 1. 克隆项目
git clone https://github.com/bingdongni/Cognitive-Evolution-Lab.git
cd Cognitive-Evolution-Lab

# 2. 自动环境检测和安装
python setup.py install

# 3. 启动演示
python src/main.py --mode=demo
```

**⏱️ 预计时间**: 10-30分钟（取决于网络速度和系统配置）

---

## 📖 详细安装步骤

### 步骤 1: 环境准备

#### 1.1 安装 Python 3.9+

1. **下载 Python**
   - 访问 [Python官网](https://www.python.org/downloads/)
   - 下载 Python 3.9 或更高版本（推荐 3.11）
   - 选择 "Windows installer (64-bit)"

2. **安装 Python**
   - 运行下载的安装程序
   - ✅ **重要**: 勾选 "Add Python to PATH"
   - 选择 "Install Now"
   - 等待安装完成

3. **验证安装**
   ```bash
   python --version
   pip --version
   ```
   
   如果显示版本信息，说明安装成功。

#### 1.2 安装 Git

1. **下载 Git**
   - 访问 [Git官网](https://git-scm.com/download/win)
   - 下载适用于 Windows 的版本

2. **安装 Git**
   - 运行安装程序，保持默认设置
   - 在 "Configuring the terminal emulator" 步骤，选择 "Use Windows' default console window"

3. **验证安装**
   ```bash
   git --version
   ```

### 步骤 2: 项目获取

```bash
# 1. 克隆项目到本地
git clone https://github.com/bingdongni/Cognitive-Evolution-Lab.git

# 2. 进入项目目录
cd Cognitive-Evolution-Lab

# 3. 查看项目结构
dir  # Windows命令
```

### 步骤 3: 创建虚拟环境（强烈推荐）

```bash
# 1. 创建虚拟环境
python -m venv celab_env

# 2. 激活虚拟环境
# 在 Windows 上：
celab_env\Scripts\activate

# 3. 升级 pip
python -m pip install --upgrade pip
```

> 💡 **提示**: 虚拟环境可以避免不同项目之间的依赖冲突。

---

## ⚙️ 环境配置

### Windows 环境变量配置

#### 1. 设置 Python 环境变量

1. 右击 "此电脑" → "属性"
2. 点击 "高级系统设置"
3. 点击 "环境变量"
4. 在 "系统变量" 中新建：
   - 变量名: `CELAB_HOME`
   - 变量值: `C:\Path\To\Cognitive-Evolution-Lab`（替换为实际路径）

5. 编辑 "系统变量" 中的 "Path"，添加：
   - `%CELAB_HOME%`
   - `%CELAB_HOME%\celab_env\Scripts`
   - `%CELAB_HOME%\src`

#### 2. 设置 CUDA 环境（如果使用 GPU）

1. **安装 CUDA Toolkit**
   - 下载 [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-downloads)
   - 运行安装程序
   - 保持默认安装路径

2. **验证 CUDA 安装**
   ```bash
   nvcc --version
   nvidia-smi
   ```

---

## 📦 依赖包安装

### 自动安装（推荐）

```bash
# 使用项目的自动安装脚本
python setup.py install
```

### 手动安装

#### 核心依赖
```bash
# 基础科学计算
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install scipy>=1.7.0
pip install matplotlib>=3.5.0

# 机器学习
pip install torch>=1.12.0
pip install scikit-learn>=1.1.0

# 游戏和可视化
pip install pygame>=2.1.0
pip install gym>=0.24.0

# 配置和工具
pip install pyyaml>=6.0
pip install psutil>=5.9.0
pip install tqdm>=4.64.0
```

#### 可选依赖（增强功能）
```bash
# 高级可视化
pip install plotly>=5.10.0
pip install dash>=2.6.0

# 深度学习增强
pip install transformers>=4.20.0

# 网络分析
pip install networkx>=2.8.0

# 图像处理
pip install opencv-python>=4.5.0
pip install Pillow>=9.2.0

# 音频处理
pip install librosa>=0.9.0
```

#### 开发依赖
```bash
# 代码质量
pip install black>=22.0.0
pip install flake8>=5.0.0
pip install pytest>=7.1.0

# 文档生成
pip install sphinx>=5.1.0
pip install sphinx-rtd-theme>=1.0.0
```

### GPU 支持安装

如果您的系统有 NVIDIA 显卡并希望启用 GPU 加速：

```bash
# 安装 CUDA 版本的 PyTorch
pip install torch==1.12.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# 验证 GPU 支持
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🔧 工具配置

### 1. Visual Studio Build Tools

某些依赖包需要编译，在 Windows 上需要 Visual Studio Build Tools：

1. **下载 Visual Studio Build Tools**
   - 访问 [Visual Studio官网](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - 下载 "Build Tools for Visual Studio"

2. **安装选项**
   - 选择 "C++ build tools"
   - 确保勾选 "Windows 10/11 SDK"
   - 选择 "Latest v143 build tools"

3. **重启计算机**（重要！）

### 2. CMake（可选，用于高级编译）

```bash
# 安装 CMake
pip install cmake

# 验证安装
cmake --version
```

### 3. Git LFS（大文件支持）

```bash
# 安装 Git LFS
git lfs install

# 配置 Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## 🎮 模拟器和游戏系统

### 1. Unity ML-Agents

#### 安装 Unity
1. **下载 Unity Hub**
   - 访问 [Unity官网](https://unity.cn/)
   - 下载 Unity Hub

2. **安装 Unity 编辑器**
   - 在 Unity Hub 中登录
   - 安装 Unity 2021.3 LTS 版本
   - 安装 "Windows Build Support"

#### 配置 ML-Agents

```bash
# 1. 克隆 ML-Agents 仓库
git clone https://github.com/Unity-Technologies/ml-认知主体s.git

# 2. 安装 Python 包
pip install ml认知主体s

# 3. 测试安装
ml认知主体s-learn --help
```

### 2. OpenAI Gym

```bash
# 安装基础 Gym 环境
pip install gym[classic_control]

# 安装 Atari 环境
pip install atari-py

# 安装其他环境
pip install gym[box2d]
pip install mujoco-py
```

### 3. 自定义游戏环境

项目包含几个预配置的游戏环境：

```python
# 在代码中使用
from src.world_simulator import VirtualWorld

# 创建游戏环境实例
world = VirtualWorld(config={
    'game_environments': ['CartPole-v1', 'Pong-v0', 'Breakout-v0'],
    'unity_认知主体s': False
})

# 启动环境
await world.initialize()
```

---

## ✅ 验证安装

### 1. 运行基础测试

```bash
# 进入项目目录
cd Cognitive-Evolution-Lab

# 运行环境验证
python -m src.utils --validate-environment

# 运行基础功能测试
python src/main.py --mode=demo
```

### 2. 测试各模块

```bash
# 测试认知模型
python -c "from src.cognitive_models import CognitiveAgent; print('✅ 认知模型模块正常')"

# 测试进化引擎
python -c "from src.evolution_engine import EvolutionEngine; print('✅ 进化引擎模块正常')"

# 测试可视化
python -c "from src.visualization import LabDashboard; print('✅ 可视化模块正常')"
```

### 3. 运行完整演示

```bash
# 认知能力测试
python src/main.py --mode=cognitive --experiment=full

# 进化实验
python src/main.py --mode=evolution --experiment=multi_认知主体

# 启动可视化仪表板
python src/main.py --mode=dashboard
```

---

## 🛠️ 故障排除

### 常见问题及解决方案

#### 问题 1: Python 版本不兼容
**症状**: `SyntaxError` 或 `ImportError`

**解决方案**:
```bash
# 检查 Python 版本
python --version

# 如果版本低于 3.9，升级 Python
# 卸载旧版本并从官网安装新版本
```

#### 问题 2: 依赖包安装失败
**症状**: `Microsoft Visual C++ 14.0 is required`

**解决方案**:
```bash
# 1. 安装 Visual Studio Build Tools
# 2. 或者使用预编译的轮子
pip install --only-binary=all package_name
```

#### 问题 3: GPU 不可用
**症状**: `CUDA out of memory` 或 `torch.cuda.is_available() == False`

**解决方案**:
```bash
# 1. 检查 GPU 驱动
nvidia-smi

# 2. 安装正确版本的 CUDA 和 PyTorch
# 3. 验证 GPU 访问
python -c "import torch; print(torch.cuda.is_available())"
```

#### 问题 4: 内存不足
**症状**: `MemoryError` 或系统响应缓慢

**解决方案**:
```bash
# 1. 关闭其他应用程序
# 2. 减少种群大小
# 编辑 config/config.yaml:
# evolution_engine:
#   population_size: 50  # 减少到 50

# 3. 启用内存优化
# performance:
#   memory_management:
#     gradient_checkpointing: true
```

#### 问题 5: Unity 环境无法启动
**症状**: `ml认知主体s-learn command not found`

**解决方案**:
```bash
# 1. 重新安装 ML-Agents
pip uninstall ml认知主体s
pip install ml认知主体s

# 2. 检查 Unity 安装
# 确保 Unity Hub 和编辑器都已正确安装
```

### 性能优化建议

#### 1. CPU 优化
```bash
# 设置并行处理
# 在 config.yaml 中设置:
performance:
  parallel_processing:
    cpu_cores: 4  # 设置为实际核心数
```

#### 2. 内存优化
```bash
# 启用内存优化
# 在 config.yaml 中设置:
performance:
  memory_management:
    gradient_checkpointing: true
    cache_size: "512MB"  # 减少缓存大小
```

#### 3. GPU 优化
```bash
# 设置 GPU 设备
# 在 config.yaml 中设置:
global:
  device: "cuda:0"  # 指定 GPU 设备
```

---

## 🔬 高级配置

### 1. 自定义配置文件

创建自定义配置文件 `config/custom_config.yaml`:

```yaml
global:
  device: "cuda"
  debug: false
  log_level: "INFO"

world_simulator:
  social_认知主体s: 100  # 增加认知主体数量
  physics_engine: "bullet"

cognitive_models:
  memory:
    capacity: 20000  # 增加记忆容量
  
evolution_engine:
  population_size: 200  # 增加种群大小
  mutation_rate: 0.15

visualization:
  render_3d:
    resolution: [2560, 1440]  # 提高分辨率
    fps: 120  # 提高帧率
```

### 2. 分布式计算配置

如果有多台机器，可以配置分布式计算：

```yaml
performance:
  distributed:
    enabled: true
    master_address: "192.168.1.100"
    master_port: 29500
    worker_addresses:
      - "192.168.1.101"
      - "192.168.1.102"
```

### 3. 实验配置模板

创建实验配置文件 `experiments/experiment_template.yaml`:

```yaml
experiment:
  name: "my_cognitive_evolution"
  description: "自定义认知进化实验"
  duration_hours: 24
  
  cognitive_tests:
    - memory
    - reasoning
    - creativity
  
  evolution_settings:
    generations: 500
    population_size: 150
    experiment_type: "multi_认知主体"
  
  output:
    save_frequency: 10  # 每10代保存一次
    export_format: ["json", "csv", "plot"]
```

---

## 📞 获取帮助

如果遇到安装问题：

1. **检查文档**: 首先查看本安装指南
2. **查看日志**: 检查 `logs/` 目录中的错误日志
3. **GitHub Issues**: 在项目页面提交问题
4. **社区支持**: 加入项目讨论群

### 日志文件位置
- 主日志: `logs/cognitive_lab.log`
- 错误日志: `logs/errors.log`
- 性能日志: `logs/performance.log`

### 常用调试命令

```bash
# 环境验证
python -m src.utils --validate-environment

# 依赖检查
python -m src.utils --check-dependencies

# 硬件检测
python -c "from src.utils import HardwareDetector; h = HardwareDetector(); print(h.get_summary())"

# 性能测试
python src/main.py --mode=performance-test
```

---

## 🎉 安装完成检查清单

完成安装后，请验证以下项目：

- [ ] Python 3.9+ 已安装且可运行
- [ ] 项目代码已克隆到本地
- [ ] 虚拟环境已创建并激活
- [ ] 所有核心依赖包已安装成功
- [ ] GPU 支持（如果适用）已配置
- [ ] 基础功能测试通过
- [ ] 可以运行演示模式
- [ ] 可视化仪表板可以启动

如果所有项目都已勾选，恭喜您成功完成了 Cognitive Evolution Lab 的安装配置！

---

**🚀 现在您可以开始探索认知计算的无限可能了！**

---

*本指南会随着项目更新而持续维护，如有疑问或建议，请联系项目维护者。*