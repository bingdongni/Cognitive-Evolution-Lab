# 贡献指南

感谢您对认知进化实验室项目的关注！我们欢迎各种形式的贡献，包括但不限于代码改进、bug修复、文档更新、新功能开发等。

## 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发环境设置](#开发环境设置)
- [代码规范](#代码规范)
- [提交规范](#提交规范)
- [测试](#测试)
- [文档](#文档)
- [问题报告](#问题报告)
- [功能请求](#功能请求)

## 行为准则

我们致力于为所有人创造一个友好、包容的贡献环境。所有参与者都应遵守以下原则：

- 友善和尊重他人
- 欢迎不同背景和观点的人
- 接受建设性批评
- 专注于对社区最有利的事情
- 表现出对他人的同理心

## 如何贡献

您可以通过以下方式为项目做出贡献：

1. **报告Bug** - 在 [Issues](https://github.com/your-repo/issues) 中报告问题
2. **建议功能** - 在 Issues 中提出新功能建议
3. **提交代码** - 通过 Pull Request 提交修复或新功能
4. **改进文档** - 帮助完善项目文档
5. **分享项目** - 向他人介绍这个项目

## 开发环境设置

### 前置要求

- Python 3.8+
- Git
- 推荐的开发环境：VS Code

### 环境配置

1. **Fork 仓库**
   ```bash
   # 访问项目页面并点击 "Fork" 按钮
   # 然后克隆您的 fork
   git clone https://github.com/your-username/Cognitive-Evolution-Lab.git
   cd Cognitive-Evolution-Lab
   ```

2. **添加上游仓库**
   ```bash
   git remote add upstream https://github.com/original-repo/Cognitive-Evolution-Lab.git
   ```

3. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate  # Windows
   ```

4. **安装依赖**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

5. **安装开发依赖**
   ```bash
   pip install -r requirements-dev.txt
   ```

## 代码规范

### Python 代码风格

我们使用以下工具确保代码质量：

- **Black** - 代码格式化
- **isort** - 导入排序
- **flake8** - 代码检查
- **mypy** - 类型检查

运行代码格式化：
```bash
# 格式化代码
black .
isort .

# 检查代码质量
flake8 .
mypy .
```

### 命名约定

- **类名**：使用 PascalCase（如 `CognitiveModel`, `EvolutionEngine`）
- **函数名**：使用 snake_case（如 `train_model`, `evaluate_performance`）
- **变量名**：使用 snake_case（如 `learning_rate`, `population_size`）
- **常量**：使用 UPPER_CASE（如 `MAX_ITERATIONS`, `DEFAULT_CONFIG`）

### 文档字符串

所有公共函数和类都必须包含文档字符串：

```python
def train_cognitive_model(epochs: int, config: dict) -> Model:
    """
    训练认知模型。

    Args:
        epochs: 训练轮数
        config: 模型配置参数

    Returns:
        训练好的模型实例

    Raises:
        ValueError: 当epochs小于1时抛出
    """
    if epochs < 1:
        raise ValueError("训练轮数必须大于0")
    
    # 实现代码...
```

## 提交规范

### 提交信息格式

使用清晰的提交信息格式：

```
类型(范围): 简短描述

详细描述（可选）

- 具体变更点 1
- 具体变更点 2

Closes #issue_number
```

**类型**：
- `feat`: 新功能
- `fix`: bug修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具

**示例**：
```
feat(models): 添加自适应学习率调整机制

- 在 CognitiveModel 中实现学习率动态调整
- 添加相关测试用例
- 更新配置文件格式

Closes #42
```

### 提交前检查清单

在提交代码前，请确保：

- [ ] 代码通过了所有测试
- [ ] 运行了代码格式化工具
- [ ] 添加了适当的测试
- [ ] 更新了相关文档
- [ ] 提交信息符合规范

## 测试

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_cognitive_models.py

# 运行测试并生成覆盖率报告
pytest --cov=src tests/
```

### 编写测试

- 为新功能添加测试
- 确保测试覆盖率在80%以上
- 使用描述性的测试名称
- 为复杂逻辑添加注释

测试示例：
```python
def test_cognitive_model_initialization():
    """测试认知模型的初始化"""
    model = CognitiveModel()
    assert model.learning_rate == 0.001
    assert len(model.parameters) > 0

def test_evolution_engine_step():
    """测试进化引擎的演化步骤"""
    engine = EvolutionEngine(population_size=10)
    initial_fitness = engine.evaluate_population()
    engine.evolve_generation()
    final_fitness = engine.evaluate_population()
    assert final_fitness >= initial_fitness
```

## 文档

### 文档更新

- 为新功能添加文档
- 更新 API 文档
- 添加使用示例
- 确保文档与代码同步

### 文档格式

- 使用 Markdown 格式
- 添加适当的标题层级
- 使用代码块展示示例
- 包含图表说明（如果适用）

## 问题报告

### Bug 报告

使用以下模板报告 bug：

```markdown
**Bug 描述**
简洁清楚地描述 bug

**复现步骤**
1. 运行 '...'
2. 使用参数 '...'
3. 看到错误 '...'

**期望行为**
简洁描述您期望发生的情况

**截图**
如果适用，请添加截图

**环境信息**
- OS: [e.g. Ubuntu 20.04]
- Python 版本: [e.g. 3.8.10]
- PyTorch 版本: [e.g. 1.12.0]
- 项目版本: [e.g. 1.2.0]

**额外信息**
任何其他有助于解决问题的信息
```

### 功能请求

使用以下模板提出功能请求：

```markdown
**功能描述**
清晰描述您希望的功能

**问题背景**
描述遇到的问题或需要此功能的原因

**期望解决方案**
描述您理想中的解决方案

**替代方案**
描述您考虑过的其他解决方案

**额外信息**
任何其他相关信息、截图等
```

## 功能请求

我们欢迎新功能建议！在提出功能请求时，请考虑：

1. **相关性** - 功能是否与项目目标相关？
2. **可行性** - 是否可以在合理时间内实现？
3. **影响** - 功能是否对现有用户有帮助？
4. **复杂性** - 实施成本是否合理？

## 致谢

感谢所有为这个项目做出贡献的开发者和研究人员！

如果您有任何问题，请随时：

- 在 GitHub Issues 中提问
- 参与现有讨论
- 联系维护团队

再次感谢您的贡献！🙏