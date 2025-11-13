# 认知进化实验室 API 参考文档

## 概述

本文档提供了认知进化实验室项目的完整API参考，包括所有核心模块、公共接口和使用示例。

---

## 核心模块 API

### 1. CognitiveModel (认知模型)

#### 类定义
```python
class CognitiveModel:
    """认知模型核心类，实现6大认知能力"""
```

#### 主要方法

##### `__init__()`
初始化认知模型
```python
def __init__(self):
    """初始化认知模型及其所有子系统"""
```

##### `process_cognitive_input(input_data: Dict) -> Dict`
处理认知输入
```python
:param input_data: 输入数据字典，包含type、content、timestamp等字段
:return: 处理结果字典，包含processed_data、cognitive_state等
:raises ValueError: 当输入数据格式不正确时
```

##### `get_cognitive_state() -> Dict`
获取当前认知状态
```python
:return: 当前认知状态字典，包含各项能力的数值
```

##### `update_cognitive_capability(capability: str, value: float)`
更新认知能力
```python
:param capability: 能力名称 ('memory', 'reasoning', 'creativity', 'attention', 'observation', 'imagination')
:param value: 新能力值 (0.0-1.0)
:raises ValueError: 当capability不在支持范围内或value超出范围时
```

### 2. WorldSimulator (世界模拟器)

#### 类定义
```python
class WorldSimulator:
    """世界模拟器，管理外部环境"""
```

#### 主要方法

##### `register_world(world_id: str, world_instance: BaseWorld)`
注册新的世界实例
```python
:param world_id: 世界唯一标识符
:param world_instance: 世界实例对象
:raises ValueError: 当world_id已存在时
```

##### `simulate_step(world_id: str, dt: float = 1.0) -> Dict`
执行模拟步骤
```python
:param world_id: 世界标识符
:param dt: 时间步长
:return: 模拟结果字典，包含timestamp、state、events等
:raises KeyError: 当world_id不存在时
```

##### `get_world_state(world_id: str) -> Dict`
获取世界当前状态
```python
:param world_id: 世界标识符
:return: 世界状态字典
```

### 3. EvolutionEngine (进化引擎)

#### 类定义
```python
class EvolutionEngine:
    """进化引擎，处理单主体和多主体进化"""
```

#### 主要方法

##### `evolve_population(population_data: Dict) -> List[CognitiveModel]`
进化种群
```python
:param population_data: 种群数据字典，包含population、evaluation_results等
:return: 新的种群列表
```

##### `select_individuals(population: List, fitness_scores: List[float], selection_rate: float = 0.5) -> List`
选择个体
```python
:param population: 当前种群
:param fitness_scores: 适应度分数列表
:param selection_rate: 选择比例 (0.0-1.0)
:return: 选择的个体列表
```

##### `mutate_genome(genome: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray`
基因组变异
```python
:param genome: 基因组数组
:param mutation_rate: 变异率
:return: 变异后的基因组
```

### 4. InteractiveSystem (交互系统)

#### 类定义
```python
class InteractiveSystem:
    """交互系统，管理智能体与环境的交互"""
```

#### 主要方法

##### `perception_action_cycle(sensory_input: Dict) -> Dict`
感知-行动循环
```python
:param sensory_input: 感知输入数据
:return: 循环结果，包含perception_result、action_result、updated_state
```

##### `execute_action(agent_id: str, action: Dict) -> Dict`
执行行动
```python
:param agent_id: 智能体ID
:param action: 行动描述字典
:return: 行动执行结果
:raises KeyError: 当agent_id不存在时
```

### 5. VisualizationSystem (可视化系统)

#### 类定义
```python
class VisualizationSystem:
    """可视化系统，提供3D渲染和仪表板功能"""
```

#### 主要方法

##### `render_scene(scene_data: Dict) -> Dict`
渲染3D场景
```python
:param scene_data: 场景数据
:return: 渲染结果，包含render_buffer、render_time、quality_metrics
```

##### `create_dashboard(layout_config: Dict) -> Dict`
创建交互式仪表板
```python
:param layout_config: 布局配置
:return: 仪表板对象或配置
```

---

## 实验模块 API

### 1. CognitiveTestExperiment (认知测试实验)

#### 类定义
```python
class CognitiveTestExperiment:
    """认知能力测试实验"""
```

#### 主要方法

##### `run_cognitive_assessment(agent: CognitiveModel, test_config: Dict) -> Dict`
运行认知评估
```python
:param agent: 待测试的认知模型
:param test_config: 测试配置
:return: 评估结果，包含各项认知能力分数
```

##### `generate_test_scenario(test_type: str, difficulty: float) -> Dict`
生成测试场景
```python
:param test_type: 测试类型 ('memory', 'reasoning', 'creativity', etc.)
:param difficulty: 难度等级 (0.0-1.0)
:return: 测试场景数据
```

### 2. MultiAgentEvolutionExperiment (多主体进化实验)

#### 类定义
```python
class MultiAgentEvolutionExperiment:
    """多主体进化实验"""
```

#### 主要方法

##### `run_experiment() -> Dict`
运行完整实验
```python
:return: 实验结果，包含population_stats、emergent_behaviors等
```

##### `evaluate_population(population: List[CognitiveModel]) -> List[Dict]`
评估种群
```python
:param population: 种群列表
:return: 评估结果列表，包含每个智能体的性能数据
```

### 3. LifelongLearningExperiment (终身学习实验)

#### 类定义
```python
class LifelongLearningExperiment:
    """终身学习实验"""
```

#### 主要方法

##### `run_learning_session(session: int) -> Dict`
运行学习会话
```python
:param session: 会话编号
:return: 会话结果，包含tasks、session_performance、knowledge_updates
```

##### `execute_learning_task(task: Dict) -> Dict`
执行学习任务
```python
:param task: 任务描述
:return: 任务执行结果，包含performance_score、learning_effectiveness
```

### 4. IntegratedExperiment (集成综合实验)

#### 类定义
```python
class IntegratedExperiment:
    """集成综合实验"""
```

#### 主要方法

##### `initialize_system() -> Dict`
初始化系统
```python
:return: 初始化结果，包含components_initialized、system_ready等
```

##### `run_phase(phase_name: str, duration: int) -> Dict`
运行特定阶段
```python
:param phase_name: 阶段名称
:param duration: 持续时间
:return: 阶段执行结果
```

---

## 工具模块 API

### 1. HardwareDetector (硬件检测器)

#### 主要方法

##### `get_system_info() -> Dict`
获取系统信息
```python
:return: 系统信息字典，包含cpu、memory、gpu等
```

##### `run_performance_benchmark() -> Dict`
运行性能基准测试
```python
:return: 基准测试结果，包含cpu_score、memory_score、gpu_score等
```

### 2. PerformanceMonitor (性能监控器)

#### 主要方法

##### `get_cpu_usage() -> Dict`
获取CPU使用率
```python
:return: CPU使用率信息
```

##### `get_memory_usage() -> Dict`
获取内存使用情况
```python
:return: 内存使用情况信息
```

### 3. DataLoader (数据加载器)

#### 主要方法

##### `load_json(file_path: str) -> Dict`
加载JSON文件
```python
:param file_path: 文件路径
:return: JSON数据
:raises FileNotFoundError: 当文件不存在时
```

##### `load_csv(file_path: str) -> List[Dict]`
加载CSV文件
```python
:param file_path: 文件路径
:return: CSV数据列表
```

---

## 配置系统 API

### 1. ConfigManager (配置管理器)

#### 主要方法

##### `load_config(config_path: str) -> Dict`
加载配置文件
```python
:param config_path: 配置文件路径
:return: 配置字典
```

##### `validate_config(config: Dict, schema: Dict) -> bool`
验证配置格式
```python
:param config: 配置数据
:param schema: 配置模式
:return: 验证是否通过
```

---

## 使用示例

### 基本使用示例

```python
# 1. 创建认知模型
from src.cognitive_models import CognitiveModel

cognitive_model = CognitiveModel()

# 2. 处理认知输入
input_data = {
    "type": "observation",
    "content": "observing a red apple",
    "timestamp": 1234567890
}

result = cognitive_model.process_cognitive_input(input_data)
print(f"认知处理结果: {result}")

# 3. 获取认知状态
state = cognitive_model.get_cognitive_state()
print(f"当前认知状态: {state}")
```

### 进化实验示例

```python
# 1. 创建多主体进化实验
from src.experiments.multi_agent_evolution import MultiAgentEvolutionExperiment, MultiAgentConfig

config = MultiAgentConfig(
    population_size=20,
    num_generations=50,
    cooperation_weight=0.3
)

experiment = MultiAgentEvolutionExperiment(config)

# 2. 运行实验
results = experiment.run_experiment()

# 3. 保存结果
experiment.save_results("experiment_results")
experiment.visualize_results("experiment_results")
```

### 世界模拟示例

```python
# 1. 创建世界模拟器
from src.world_simulator import WorldSimulator, PhysicalWorld

simulator = WorldSimulator()

# 2. 注册世界
physical_world = PhysicalWorld()
simulator.register_world("physics", physical_world)

# 3. 执行模拟
result = simulator.simulate_step("physics", dt=0.1)
print(f"模拟结果: {result}")
```

---

## 错误处理

### 常见异常

#### `ValueError`
当参数值无效时抛出
```python
try:
    cognitive_model.update_cognitive_capability("invalid_capability", 1.5)
except ValueError as e:
    print(f"参数错误: {e}")
```

#### `KeyError`
当访问不存在的键时抛出
```python
try:
    state = simulator.get_world_state("nonexistent_world")
except KeyError as e:
    print(f"世界不存在: {e}")
```

#### `FileNotFoundError`
当文件不存在时抛出
```python
try:
    data = data_loader.load_json("nonexistent_file.json")
except FileNotFoundError as e:
    print(f"文件不存在: {e}")
```

---

## 性能优化建议

### 1. 批量处理
```python
# 推荐：批量处理多个认知输入
inputs = [input_data_1, input_data_2, input_data_3]
results = [cognitive_model.process_cognitive_input(inp) for inp in inputs]
```

### 2. 缓存机制
```python
# 对于频繁访问的数据启用缓存
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(param: str) -> Dict:
    # 耗时计算
    pass
```

### 3. 并行处理
```python
# 对于独立任务使用并行处理
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_agent, agent) for agent in agents]
    results = [future.result() for future in futures]
```

---

## 扩展指南

### 1. 创建自定义认知能力
```python
class CustomCognitiveModel(CognitiveModel):
    def __init__(self):
        super().__init__()
        self.add_capability("custom_ability", self._custom_ability)
    
    def _custom_ability(self, input_data):
        # 自定义能力实现
        pass
```

### 2. 自定义世界类型
```python
from src.world_simulator import BaseWorld

class CustomWorld(BaseWorld):
    def simulate_step(self, dt: float) -> Dict:
        # 自定义世界模拟逻辑
        pass
```

### 3. 添加新的实验类型
```python
class CustomExperiment:
    def __init__(self, config):
        self.config = config
        self.setup_experiment()
    
    def run_experiment(self) -> Dict:
        # 实验实现
        pass
```

---

## 版本兼容性

### API 版本策略
- 主版本号：不兼容的API更改
- 次版本号：向后兼容的功能性新增
- 修订号：向后兼容的问题修正

### 废弃警告
当API即将废弃时，会产生警告：
```python
import warnings

warnings.warn(
    "This function will be deprecated in v2.0. Use new_function() instead.",
    DeprecationWarning
)
```

---

## 技术支持

### 文档更新
- 本文档随代码版本自动更新
- 详细的开发文档请参考 `docs/` 目录

### 问题反馈
- GitHub Issues: [项目地址]/issues
- 技术讨论: [项目地址]/discussions

### 贡献指南
详见 `CONTRIBUTING.md` 文件

---

*最后更新: 2025-11-13*
*文档版本: v1.0.0*
