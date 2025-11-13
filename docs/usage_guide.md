# Cognitive Evolution Lab - 使用教程

**作者**: bingdongni  
**版本**: v1.0.0

> 📚 本教程将带您深入了解如何使用 Cognitive Evolution Lab 进行各种认知和进化实验。

---

## 📋 目录

- [快速入门](#-快速入门)
- [核心概念](#-核心概念)
- [基础操作](#-基础操作)
- [实验指南](#-实验指南)
- [高级功能](#-高级功能)
- [性能优化](#-性能优化)

---

## 🚀 快速入门

### 第一次运行

```bash
# 1. 激活虚拟环境
celab_env\Scripts\activate

# 2. 运行演示模式
python src/main.py --mode=demo
```

这将启动一个简单的演示，展示项目的基本功能。

### 运行认知测试

```bash
# 测试所有认知能力
python src/main.py --mode=cognitive --experiment=full

# 只测试记忆能力
python src/main.py --mode=cognitive --experiment=memory

# 只测试推理能力
python src/main.py --mode=cognitive --experiment=reasoning
```

### 启动可视化界面

```bash
# 启动交互式仪表板
python src/main.py --mode=dashboard

# 访问 http://localhost:8050 查看可视化界面
```

---

## 🧠 核心概念

### 1. 三大核心模块

#### 外部世界模拟器 (World Simulator)
- **功能**: 模拟物理世界、社会世界、游戏世界
- **用途**: 为认知主体提供交互环境
- **特点**: 支持多模态感知、物理仿真、社会交互

#### 内部心智模型 (Cognitive Models)
- **功能**: 模拟人类认知过程
- **能力**: 记忆、推理、注意力、创造力、观察力、想象力
- **架构**: 神经网络 + 符号推理的混合架构

#### 交互行动系统 (Interactive Systems)
- **功能**: 控制认知主体与环境交互
- **能力**: 运动控制、多模态感知、动作规划
- **特点**: 支持具身智能和多认知主体协作

### 2. 认知能力六大支柱

1. **记忆力** - 存储和检索信息的能力
2. **思维力** - 逻辑推理和问题解决能力
3. **创造力** - 产生新颖想法的能力
4. **观察力** - 感知和分析环境信息的能力
5. **注意力** - 选择性关注特定信息的能力
6. **想象力** - 模拟未来情景和可能性

### 3. 进化机制

- **个体进化**: 基于遗传算法的参数优化
- **群体进化**: 多认知主体协作与竞争
- **知识进化**: 经验积累和规则发现
- **环境共演化**: 环境复杂度与认知主体能力协同演化

---

## 🛠️ 基础操作

### 1. 创建认知认知主体

```python
from src.cognitive_models import CognitiveAgent

# 创建认知认知主体
认知主体 = CognitiveAgent(config={
    'memory': {
        'capacity': 5000,
        'hierarchical': True
    },
    'attention': {
        'type': 'transformer',
        'heads': 8
    }
})

# 初始化
await 认知主体.initialize()
```

### 2. 存储和检索记忆

```python
# 存储记忆
await 认知主体.store_memory("这是一个重要的信息", MemoryType.EPISODIC, strength=0.9)

# 检索相关记忆
related_memories = await 认知主体.retrieve_memory("信息", threshold=0.5)

print(f"找到 {len(related_memories)} 条相关记忆")
```

### 3. 执行推理

```python
# 演绎推理
reasoning_chain = await 认知主体.reason([
    "所有鸟类都会飞",
    "企鹅是鸟类"
], ReasoningType.DEDUCTIVE)

print(f"推理结论: {reasoning_chain.conclusion}")
print(f"置信度: {reasoning_chain.confidence:.2f}")
```

### 4. 生成创意

```python
# 生成创意输出
creative_result = await 认知主体.generate_creative_output(
    context="设计一个环保产品",
    style="创新"
)

print(f"创意内容: {creative_result['creative_text']}")
print(f"创造力评分: {creative_result['creativity_score']:.2f}")
```

### 5. 观察环境

```python
# 模拟观察数据
observations = torch.randn(1, 3, 224, 224)

# 观察分析
observation_result = await 认知主体.observe_environment(observations)

print(f"模式特征: {observation_result['pattern_features'].shape}")
print(f"异常分数: {observation_result['anomaly_score']:.2f}")
```

---

## 🔬 实验指南

### 实验 1: 基础认知能力评估

```python
from src.experiments import CognitiveTest

# 创建认知测试
test = CognitiveTest()

# 运行全套测试
results = await test.run_test(认知主体, test_type="full")

print(f"总体认知评分: {results['overall_score']:.3f}")

# 查看各能力评分
for result in results['test_results']:
    print(f"{result['category']}: {result['score']:.3f}")
```

### 实验 2: 多认知主体协同进化

```python
from src.experiments import MultiAgentEvolution

# 创建进化实验
evolution = MultiAgentEvolution(config={
    'population_size': 50,
    'generations': 100,
    'experiment_type': 'multi_认知主体'
})

# 初始化种群
population = await evolution.initialize_population()

# 运行进化
results = await evolution.evolve(population, generations=100)

print(f"最终最佳适应度: {results['final_fitness']:.4f}")
print(f"种群多样性: {results['diversity_score']:.3f}")
```

### 实验 3: 终身学习测试

```python
from src.experiments import LifelongLearning

# 创建终身学习实验
lifelong_learning = LifelongLearning(config={
    'duration_hours': 2.0,
    'task_sequence': [
        'pattern_recognition',
        'sequence_learning',
        'transfer_learning'
    ]
})

# 运行实验
results = await lifelong_learning.run_learning_experiment()

print(f"学习效率: {results['learning_efficiency']:.3f}")
print(f"知识保持率: {results['knowledge_retention']:.3f}")
```

### 实验 4: 集成综合实验

```python
from src.experiments import IntegratedExperiment

# 创建集成实验
integrated = IntegratedExperiment(config={
    'include_cognitive_test': True,
    'include_evolution': True,
    'include_lifelong_learning': True
})

# 运行完整实验流程
results = await integrated.run_integrated_experiment()

print(f"综合实验评分: {results['integrated_score']:.4f}")
```

---

## 🎮 交互式使用

### 1. 世界模拟器

```python
from src.world_simulator import VirtualWorld

# 创建虚拟世界
world = VirtualWorld(config={
    'world_type': 'hybrid_world',
    'social_认知主体s': 30,
    'physics_engine': 'bullet',
    'game_environments': ['CartPole-v1', 'Pong-v0']
})

# 初始化世界
await world.initialize()

# 运行世界仿真
await world.start()

# 模拟一段时间
for step in range(1000):
    await world.step()
    await asyncio.sleep(0.01)  # 控制仿真速度

await world.stop()
```

### 2. 具身智能控制

```python
from src.interactive_systems import EmbodiedIntelligence

# 创建具身智能系统
embodied = EmbodiedIntelligence(config={
    'body_model': 'humanoid',
    'motor_control': 'policy_gradient',
    'multimodal_perception': {
        'vision': {'enabled': True},
        'audio': {'enabled': True},
        'touch': {'enabled': True}
    }
})

# 初始化
await embodied.initialize()

# 感知环境
perception = await embodied.perceive_environment()

# 规划动作
goals = ["move_forward", "avoid_obstacle"]
execution_plan = await embodied.plan_action(goals)

# 执行动作
action = execution_plan['primary_action']
result = await embodied.execute_action(action)

print(f"动作执行结果: {result['success']}")
```

---

## 📊 可视化使用

### 1. 实时监控

```python
from src.visualization import LabDashboard

# 创建可视化仪表板
dashboard = LabDashboard(config={
    'render_3d': {'enabled': True},
    'dashboard': {'framework': 'dash'},
    'monitoring': {
        'brain_activity': True,
        'learning_curves': True,
        'evolution_progress': True
    }
})

# 初始化
await dashboard.initialize()

# 启动仪表板
await dashboard.start_server(port=8050)

# 渲染一帧
await dashboard.render_frame(world_state, cognitive_state, evolution_state)
```

### 2. 性能监控

```python
from src.utils import PerformanceMonitor

# 创建性能监控器
monitor = PerformanceMonitor()

# 开始监控
await monitor.start_monitoring()

# 获取性能摘要
summary = monitor.get_performance_summary()
print(f"当前FPS: {summary['current_metrics']['fps']:.1f}")
print(f"内存使用: {summary['current_metrics']['memory_usage']:.1f}%")
```

---

## 🔧 高级功能

### 1. 自定义认知模型

```python
from src.cognitive_models import CognitiveAgent
from src.cognitive_models import MemoryType, ReasoningType

class CustomCognitiveAgent(CognitiveAgent):
    def __init__(self, config):
        super().__init__(config)
        self.custom_module = CustomModule()
    
    async def custom_cognitive_task(self, input_data):
        # 实现自定义认知任务
        result = await self.custom_module.process(input_data)
        return result

# 使用自定义认知主体
custom_认知主体 = CustomCognitiveAgent(custom_config)
```

### 2. 自定义进化策略

```python
from src.evolution_engine import EvolutionEngine

class CustomEvolutionEngine(EvolutionEngine):
    async def custom_selection_strategy(self, population):
        # 实现自定义选择策略
        selected = []
        for individual in population:
            if individual.fitness > self.custom_threshold:
                selected.append(individual)
        return selected
```

### 3. 插件系统

```python
# 创建自定义插件
class MyPlugin:
    def __init__(self, config):
        self.config = config
    
    async def initialize(self):
        # 插件初始化
        pass
    
    async def process(self, data):
        # 数据处理
        return processed_data
    
    async def cleanup(self):
        # 清理资源
        pass

# 注册插件
认知主体.register_plugin('my_plugin', MyPlugin(plugin_config))
```

### 4. 分布式计算

```python
from src.distributed import DistributedLab

# 创建分布式实验室
distributed_lab = DistributedLab(config={
    'master_address': 'localhost',
    'worker_count': 4,
    'task_distribution': 'balanced'
})

# 初始化分布式环境
await distributed_lab.initialize()

# 运行分布式实验
results = await distributed_lab.run_distributed_experiment(
    experiment_type='cognitive_evolution',
    population_size=1000
)
```

---

## 📈 性能优化

### 1. 内存优化

```python
# 在配置中启用内存优化
config = {
    'performance': {
        'memory_management': {
            'gradient_checkpointing': True,
            'cache_size': '512MB',
            'garbage_collection': True
        }
    }
}

# 使用上下文管理器自动清理内存
async with MemoryManager() as memory_manager:
    # 运行计算密集型任务
    result = await heavy_computation()
    # 内存会自动清理
```

### 2. 并行处理

```python
# 配置并行处理
config = {
    'performance': {
        'parallel_processing': {
            'cpu_cores': 4,
            'gpu_devices': [0, 1],
            'distributed': False
        }
    }
}

# 使用并行执行
await asyncio.gather(
    cognitive_test(),
    evolution_experiment(),
    visualization_update()
)
```

### 3. GPU 加速

```python
# 配置 GPU 使用
config = {
    'global': {
        'device': 'cuda:0'
    }
}

# 手动移动模型到 GPU
model = MyModel()
model = model.cuda()

# 使用 GPU 进行推理
with torch.no_grad():
    result = model(input_data.cuda())
```

### 4. 缓存优化

```python
# 启用结果缓存
cache_config = {
    'cache_enabled': True,
    'cache_size': '1GB',
    'cache_ttl': 3600  # 1小时
}

# 使用缓存
cached_result = await cache.get_or_compute(
    key='experiment_001',
    computation_func=heavy_computation
)
```

---

## 📝 最佳实践

### 1. 代码组织

```
my_experiment/
├── config/
│   ├── experiment_config.yaml
│   └── custom_models.py
├── src/
│   ├── custom_认知主体.py
│   └── custom_evolution.py
├── data/
│   ├── input_data/
│   └── results/
└── scripts/
    └── run_experiment.py
```

### 2. 配置管理

```python
# 使用配置文件
config = load_config('config/experiment_config.yaml')

# 环境特定配置
if is_gpu_available():
    config['device'] = 'cuda'
else:
    config['device'] = 'cpu'

# 根据硬件调整参数
hardware_info = get_hardware_info()
config['population_size'] = min(100, hardware_info['memory_gb'] * 10)
```

### 3. 实验记录

```python
# 实验元数据
experiment_metadata = {
    'name': 'cognitive_evolution_v1',
    'timestamp': datetime.now().isoformat(),
    'config_hash': hashlib.md5(str(config).encode()).hexdigest(),
    'hardware_info': get_hardware_info(),
    'random_seed': 42
}

# 保存实验结果
save_experiment_results(results, experiment_metadata)
```

### 4. 错误处理

```python
import logging
from src.utils import error_handler, retry_on_failure

@retry_on_failure(max_retries=3, delay=1.0)
@error_handler
async def robust_experiment():
    try:
        result = await risky_operation()
        return result
    except Exception as e:
        logging.error(f"实验失败: {e}")
        raise
```

---

## 🎯 常见用例

### 用例 1: 比较不同认知架构

```python
# 测试不同记忆架构
architectures = ['hierarchical', 'flat', 'distributed']

results = {}
for arch in architectures:
    config = {'memory': {'type': arch}}
    认知主体 = CognitiveAgent(config)
    result = await 认知主体.run_cognitive_test("memory")
    results[arch] = result['score']

# 分析结果
best_architecture = max(results, key=results.get)
print(f"最佳记忆架构: {best_architecture}")
```

### 用例 2: 进化参数优化

```python
# 测试不同进化参数
param_combinations = [
    {'mutation_rate': 0.1, 'crossover_rate': 0.8},
    {'mutation_rate': 0.05, 'crossover_rate': 0.9},
    {'mutation_rate': 0.2, 'crossover_rate': 0.6}
]

for params in param_combinations:
    config = {'genetic_config': params}
    evolution = EvolutionEngine(config)
    result = await evolution.evolve(population, generations=50)
    print(f"参数 {params}: 适应度 {result['final_fitness']:.4f}")
```

### 用例 3: 长期跟踪研究

```python
# 长期学习研究
long_term_study = {
    'duration_days': 30,
    'daily_experiments': [
        'cognitive_test',
        'evolution_round',
        'learning_task'
    ],
    'progress_tracking': True
}

# 运行长期研究
study_results = await run_long_term_study(long_term_study)
```

---

## 🎓 学习路径

### 初学者路径
1. 运行基础演示 (`--mode=demo`)
2. 尝试认知测试 (`--mode=cognitive`)
3. 观察可视化界面 (`--mode=dashboard`)
4. 修改简单配置参数

### 进阶用户路径
1. 创建自定义认知测试
2. 设计新的进化实验
3. 集成外部数据源
4. 开发自定义插件

### 研究者路径
1. 发表基于项目的论文
2. 扩展到分布式计算
3. 集成前沿认知计算技术
4. 开源贡献

---

*更多高级功能和详细信息，请参考项目的 API 文档和源码注释。*