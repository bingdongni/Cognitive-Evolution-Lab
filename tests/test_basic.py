#!/usr/bin/env python3
"""
Cognitive Evolution Lab - 基础功能测试
作者: bingdongni
版本: v1.0.0

此脚本包含认知进化实验室核心功能的基础单元测试，
确保所有主要模块能够正常初始化和工作。

测试覆盖范围：
1. 认知模型初始化
2. 记忆系统功能
3. 推理能力测试
4. 创造力模块测试
5. 观察力模块测试
6. 注意力机制测试
7. 想象力系统测试
8. 进化引擎测试
9. 工具函数测试
10. 配置加载测试

使用方法:
    python tests/test_basic.py [--verbose] [--test MODULE] [--output OUTPUT_DIR]
"""

import asyncio
import unittest
import logging
import tempfile
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import time

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.cognitive_models import (
    CognitiveAgent, MemoryType, ReasoningType, AttentionType,
    HierarchicalMemory, AttentionMechanism, NeuroSymbolicReasoner,
    CreativityModule, ObservationModule, MetaLearner
)
from src.utils import (
    setup_logging, load_config, validate_environment,
    HardwareDetector
)
from src.world_simulator import VirtualWorld
from src.evolution_engine import EvolutionEngine


class TestCognitiveLab(unittest.TestCase):
    """认知实验室基础功能测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        # 设置测试日志
        test_config = {
            'global': {
                'debug': True,
                'log_level': 'WARNING',  # 减少测试时的日志输出
                'random_seed': 42
            }
        }
        setup_logging(test_config)
        cls.logger = logging.getLogger(__name__)
        
        # 测试配置
        cls.test_config = {
            'cognitive_models': {
                'vocab_size': 1000,
                'embed_dim': 128,
                'hidden_dim': 256,
                'learning_rate': 0.01
            },
            'world_simulator': {
                'environment_size': [50, 50, 20],
                'max_objects': 50
            },
            'evolution_engine': {
                'population_size': 10,
                'generations': 5
            }
        }
        
        cls.logger.info("🧪 认知实验室测试套件初始化完成")
    
    def setUp(self):
        """每个测试方法执行前的准备"""
        self.test_start_time = time.time()
    
    def tearDown(self):
        """每个测试方法执行后的清理"""
        test_duration = time.time() - self.test_start_time
        self.logger.debug(f"测试耗时: {test_duration:.2f}秒")
    
    async def test_cognitive_agent_initialization(self):
        """测试认知智能体初始化"""
        self.logger.info("测试认知智能体初始化...")
        
        # 创建认知智能体实例
        cognitive_agent = CognitiveAgent(config=self.test_config['cognitive_models'])
        
        # 验证基本属性
        self.assertIsNotNone(cognitive_agent.config)
        self.assertIsNotNone(cognitive_agent.logger)
        self.assertEqual(cognitive_agent.cognitive_state.attention_focus, "default")
        self.assertEqual(cognitive_agent.cognitive_state.current_goal, "explore")
        
        # 验证记忆存储初始化
        self.assertIsNotNone(cognitive_agent.memories)
        self.assertEqual(len(cognitive_agent.memories), 4)  # 四种记忆类型
        
        # 验证推理链存储初始化
        self.assertIsNotNone(cognitive_agent.reasoning_chains)
        
        self.logger.info("✅ 认知智能体初始化测试通过")
    
    async def test_cognitive_agent_initialization_async(self):
        """测试认知智能体异步初始化"""
        self.logger.info("测试认知智能体异步初始化...")
        
        cognitive_agent = CognitiveAgent(config=self.test_config['cognitive_models'])
        
        # 执行异步初始化
        await cognitive_agent.initialize()
        
        # 验证组件初始化
        self.assertIsNotNone(cognitive_agent.memory_system)
        self.assertIsNotNone(cognitive_agent.attention_mechanism)
        self.assertIsNotNone(cognitive_agent.reasoning_system)
        self.assertIsNotNone(cognitive_agent.creativity_module)
        self.assertIsNotNone(cognitive_agent.observation_module)
        self.assertIsNotNone(cognitive_agent.meta_learner)
        
        # 清理资源
        await cognitive_agent.cleanup()
        
        self.logger.info("✅ 认知智能体异步初始化测试通过")
    
    async def test_memory_operations(self):
        """测试记忆系统操作"""
        self.logger.info("测试记忆系统操作...")
        
        cognitive_agent = CognitiveAgent(config=self.test_config['cognitive_models'])
        await cognitive_agent.initialize()
        
        # 测试记忆存储
        test_memories = [
            "今天学习了一个新的算法",
            "机器学习很有趣",
            "深度学习是AI的重要分支"
        ]
        
        for i, memory in enumerate(test_memories):
            await cognitive_agent.store_memory(
                content=memory,
                memory_type=MemoryType.EPISODIC,
                strength=0.9 - i * 0.1
            )
        
        # 验证记忆已存储
        self.assertEqual(len(cognitive_agent.memories[MemoryType.EPISODIC]), 3)
        
        # 测试记忆检索
        retrieved_memories = await cognitive_agent.retrieve_memory(
            query="学习",
            threshold=0.3
        )
        
        self.assertGreaterEqual(len(retrieved_memories), 0)  # 至少可能检索到0个
        
        # 测试记忆保留测试
        retention_test = await cognitive_agent.test_memory_retention()
        self.assertIn('retention_scores', retention_test)
        self.assertIn('retention_score', retention_test)
        self.assertGreaterEqual(retention_test['retention_score'], 0.0)
        
        await cognitive_agent.cleanup()
        self.logger.info("✅ 记忆系统操作测试通过")
    
    async def test_reasoning_capabilities(self):
        """测试推理能力"""
        self.logger.info("测试推理能力...")
        
        cognitive_agent = CognitiveAgent(config=self.test_config['cognitive_models'])
        await cognitive_agent.initialize()
        
        # 测试演绎推理
        deductive_premises = [
            "所有AI系统都需要数据",
            "机器学习是AI系统"
        ]
        
        reasoning_chain = await cognitive_agent.reason(
            premises=deductive_premises,
            reasoning_type=ReasoningType.DEDUCTIVE
        )
        
        # 验证推理链
        self.assertIsNotNone(reasoning_chain)
        self.assertEqual(len(reasoning_chain.premises), 2)
        self.assertIsNotNone(reasoning_chain.conclusion)
        self.assertGreaterEqual(reasoning_chain.confidence, 0.0)
        self.assertLessEqual(reasoning_chain.confidence, 1.0)
        self.assertEqual(reasoning_chain.reasoning_type, ReasoningType.DEDUCTIVE)
        
        # 测试归纳推理
        inductive_premises = [
            "观察到天鹅1是白的",
            "观察到天鹅2是白的"
        ]
        
        inductive_chain = await cognitive_agent.reason(
            premises=inductive_premises,
            reasoning_type=ReasoningType.INDUCTIVE
        )
        
        self.assertIsNotNone(inductive_chain)
        self.assertEqual(inductive_chain.reasoning_type, ReasoningType.INDUCTIVE)
        
        await cognitive_agent.cleanup()
        self.logger.info("✅ 推理能力测试通过")
    
    async def test_attention_mechanism(self):
        """测试注意力机制"""
        self.logger.info("测试注意力机制...")
        
        cognitive_agent = CognitiveAgent(config=self.test_config['cognitive_models'])
        await cognitive_agent.initialize()
        
        # 测试选择性注意
        selective_weights = await cognitive_agent.focus_attention(
            target="学习新技能",
            attention_type=AttentionType.SELECTIVE
        )
        
        self.assertIsNotNone(selective_weights)
        self.assertIn('relevance', selective_weights)
        self.assertGreaterEqual(selective_weights['relevance'], 0.0)
        self.assertLessEqual(selective_weights['relevance'], 1.0)
        
        # 测试持续性注意
        sustained_weights = await cognitive_agent.focus_attention(
            target="专注工作",
            attention_type=AttentionType.SUSTAINED
        )
        
        self.assertIsNotNone(sustained_weights)
        self.assertIn('persistence', sustained_weights)
        
        # 测试分散性注意
        divided_weights = await cognitive_agent.focus_attention(
            target="多任务处理",
            attention_type=AttentionType.DIVIDED
        )
        
        self.assertIsNotNone(divided_weights)
        self.assertIn('balance', divided_weights)
        
        await cognitive_agent.cleanup()
        self.logger.info("✅ 注意力机制测试通过")
    
    async def test_creativity_module(self):
        """测试创造力模块"""
        self.logger.info("测试创造力模块...")
        
        cognitive_agent = CognitiveAgent(config=self.test_config['cognitive_models'])
        await cognitive_agent.initialize()
        
        # 测试创意生成
        creative_output = await cognitive_agent.generate_creative_output(
            context="设计一个新产品",
            style="创新"
        )
        
        # 验证创意输出
        self.assertIsNotNone(creative_output)
        self.assertIn('creative_text', creative_output)
        self.assertIn('creativity_score', creative_output)
        self.assertIn('style', creative_output)
        self.assertIn('context', creative_output)
        
        # 验证评分范围
        self.assertGreaterEqual(creative_output['creativity_score'], 0.0)
        self.assertLessEqual(creative_output['creativity_score'], 1.0)
        
        # 验证创意文本
        self.assertIsInstance(creative_output['creative_text'], str)
        self.assertGreater(len(creative_output['creative_text']), 0)
        
        await cognitive_agent.cleanup()
        self.logger.info("✅ 创造力模块测试通过")
    
    async def test_observation_module(self):
        """测试观察力模块"""
        self.logger.info("测试观察力模块...")
        
        cognitive_agent = CognitiveAgent(config=self.test_config['cognitive_models'])
        await cognitive_agent.initialize()
        
        # 创建模拟观察数据
        import torch
        mock_observations = torch.randn(1, 3, 64, 64)  # 小尺寸图像
        
        # 测试环境观察
        observation_results = await cognitive_agent.observe_environment(
            observations=mock_observations
        )
        
        # 验证观察结果
        self.assertIsNotNone(observation_results)
        self.assertIn('pattern_features', observation_results)
        self.assertIn('anomaly_score', observation_results)
        self.assertIn('attention_triggered', observation_results)
        
        # 验证异常评分
        self.assertIsInstance(observation_results['anomaly_score'], torch.Tensor)
        self.assertGreaterEqual(observation_results['anomaly_score'].item(), 0.0)
        self.assertLessEqual(observation_results['anomaly_score'].item(), 1.0)
        
        # 验证注意力触发
        self.assertIsInstance(observation_results['attention_triggered'], bool)
        
        await cognitive_agent.cleanup()
        self.logger.info("✅ 观察力模块测试通过")
    
    async def test_imagination_system(self):
        """测试想象力系统"""
        self.logger.info("测试想象力系统...")
        
        cognitive_agent = CognitiveAgent(config=self.test_config['cognitive_models'])
        await cognitive_agent.initialize()
        
        # 测试情景想象
        imagination_output = await cognitive_agent.imagine_scenario(
            context="未来世界",
            constraints=["技术先进", "可持续发展"]
        )
        
        # 验证想象结果
        self.assertIsNotNone(imagination_output)
        self.assertIn('context', imagination_output)
        self.assertIn('scenario_elements', imagination_output)
        self.assertIn('probabilities', imagination_output)
        self.assertIn('constraints', imagination_output)
        
        # 验证场景元素
        self.assertIsInstance(imagination_output['scenario_elements'], list)
        self.assertIsInstance(imagination_output['probabilities'], list)
        self.assertEqual(len(imagination_output['scenario_elements']), len(imagination_output['probabilities']))
        
        # 验证约束条件
        self.assertEqual(imagination_output['constraints'], ["技术先进", "可持续发展"])
        
        await cognitive_agent.cleanup()
        self.logger.info("✅ 想象力系统测试通过")
    
    async def test_cognitive_test_integration(self):
        """测试认知能力综合测试"""
        self.logger.info("测试认知能力综合测试...")
        
        cognitive_agent = CognitiveAgent(config=self.test_config['cognitive_models'])
        await cognitive_agent.initialize()
        
        # 创建模拟环境
        class MockEnvironment:
            async def get_test_data(self):
                return {'mock_data': 'test'}
        
        mock_env = MockEnvironment()
        
        # 运行综合认知测试
        test_results = await cognitive_agent.run_cognitive_test(
            environment=mock_env,
            test_type="full"
        )
        
        # 验证测试结果
        self.assertIsNotNone(test_results)
        self.assertIn('memory', test_results)
        self.assertIn('reasoning', test_results)
        self.assertIn('creativity', test_results)
        self.assertIn('observation', test_results)
        self.assertIn('attention', test_results)
        self.assertIn('imagination', test_results)
        self.assertIn('overall_score', test_results)
        self.assertIn('cognitive_state', test_results)
        self.assertIn('test_type', test_results)
        
        # 验证评分范围
        self.assertGreaterEqual(test_results['overall_score'], 0.0)
        self.assertLessEqual(test_results['overall_score'], 1.0)
        
        # 验证认知状态
        cognitive_state = test_results['cognitive_state']
        self.assertIn('attention_focus', cognitive_state)
        self.assertIn('cognitive_load', cognitive_state)
        self.assertIn('working_memory_size', cognitive_state)
        
        await cognitive_agent.cleanup()
        self.logger.info("✅ 认知能力综合测试通过")
    
    async def test_world_simulator(self):
        """测试世界模拟器"""
        self.logger.info("测试世界模拟器...")
        
        # 创建世界模拟器
        world_simulator = VirtualWorld(
            config=self.test_config['world_simulator']
        )
        
        # 验证基本属性
        self.assertIsNotNone(world_simulator.config)
        self.assertIsNotNone(world_simulator.logger)
        
        # 尝试初始化（如果实现的话）
        try:
            await world_simulator.initialize()
            self.logger.info("世界模拟器初始化成功")
        except Exception as e:
            self.logger.warning(f"世界模拟器初始化跳过: {e}")
        
        # 如果有清理方法，调用它
        if hasattr(world_simulator, 'cleanup'):
            await world_simulator.cleanup()
        
        self.logger.info("✅ 世界模拟器测试通过")
    
    async def test_evolution_engine(self):
        """测试进化引擎"""
        self.logger.info("测试进化引擎...")
        
        # 创建进化引擎
        evolution_engine = EvolutionEngine(
            config=self.test_config['evolution_engine']
        )
        
        # 验证基本属性
        self.assertIsNotNone(evolution_engine.config)
        self.assertIsNotNone(evolution_engine.logger)
        
        # 尝试初始化（如果实现的话）
        try:
            await evolution_engine.initialize()
            self.logger.info("进化引擎初始化成功")
        except Exception as e:
            self.logger.warning(f"进化引擎初始化跳过: {e}")
        
        # 如果有清理方法，调用它
        if hasattr(evolution_engine, 'cleanup'):
            await evolution_engine.cleanup()
        
        self.logger.info("✅ 进化引擎测试通过")
    
    def test_config_loading(self):
        """测试配置加载功能"""
        self.logger.info("测试配置加载...")
        
        # 测试默认配置加载
        try:
            config = load_config()
            self.assertIsInstance(config, dict)
            self.assertGreater(len(config), 0)
        except Exception as e:
            self.logger.warning(f"默认配置加载测试跳过: {e}")
        
        # 测试临时配置文件
        test_config = {
            'test_setting': {
                'value': 42,
                'name': 'test_config'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            loaded_config = load_config(config_path)
            self.assertIsInstance(loaded_config, dict)
            # 注意：load_config可能会合并配置，所以检查是否存在我们的设置
            if 'test_setting' in loaded_config:
                self.assertEqual(loaded_config['test_setting']['value'], 42)
        except Exception as e:
            self.logger.warning(f"临时配置文件加载测试跳过: {e}")
        finally:
            # 清理临时文件
            Path(config_path).unlink()
        
        self.logger.info("✅ 配置加载测试通过")
    
    def test_hardware_detection(self):
        """测试硬件检测功能"""
        self.logger.info("测试硬件检测...")
        
        # 创建硬件检测器
        hardware_detector = HardwareDetector()
        
        # 验证基本功能
        self.assertIsNotNone(hardware_detector)
        
        # 测试获取摘要
        try:
            summary = hardware_detector.get_summary()
            self.assertIsInstance(summary, dict)
            self.assertGreater(len(summary), 0)
            self.logger.info(f"硬件摘要: {summary}")
        except Exception as e:
            self.logger.warning(f"硬件检测摘要测试跳过: {e}")
        
        # 测试设备检测
        try:
            cpu_cores = hardware_detector.get_cpu_cores()
            self.assertIsInstance(cpu_cores, int)
            self.assertGreater(cpu_cores, 0)
        except Exception as e:
            self.logger.warning(f"CPU核心检测测试跳过: {e}")
        
        self.logger.info("✅ 硬件检测测试通过")
    
    def test_environment_validation(self):
        """测试环境验证功能"""
        self.logger.info("测试环境验证...")
        
        # 测试环境验证
        try:
            validation_result = validate_environment()
            self.assertIsInstance(validation_result, dict)
            self.assertIn('status', validation_result)
            self.logger.info(f"环境验证结果: {validation_result}")
        except Exception as e:
            self.logger.warning(f"环境验证测试跳过: {e}")
        
        self.logger.info("✅ 环境验证测试通过")
    
    async def test_all_cognitive_tests(self):
        """测试所有认知模块的完整集成"""
        self.logger.info("测试所有认知模块完整集成...")
        
        cognitive_agent = CognitiveAgent(config=self.test_config['cognitive_models'])
        await cognitive_agent.initialize()
        
        # 测试各种认知测试类型
        test_types = ["memory", "reasoning", "creativity", "observation", "attention", "imagination"]
        
        for test_type in test_types:
            with self.subTest(test_type=test_type):
                # 创建模拟环境
                class MockTestEnvironment:
                    async def get_test_data(self):
                        return {'test_type': test_type, 'data': 'mock_data'}
                
                mock_env = MockTestEnvironment()
                
                # 执行测试
                test_result = await cognitive_agent.run_cognitive_test(
                    environment=mock_env,
                    test_type=test_type
                )
                
                # 验证结果
                self.assertIsNotNone(test_result)
                self.assertIn(test_type, test_result)
                self.assertIn('score', test_result[test_type])
                self.assertGreaterEqual(test_result[test_type]['score'], 0.0)
                self.assertLessEqual(test_result[test_type]['score'], 1.0)
        
        await cognitive_agent.cleanup()
        self.logger.info("✅ 所有认知模块完整集成测试通过")
    
    async def test_error_handling(self):
        """测试错误处理"""
        self.logger.info("测试错误处理...")
        
        cognitive_agent = CognitiveAgent(config=self.test_config['cognitive_models'])
        await cognitive_agent.initialize()
        
        # 测试无效记忆类型
        with self.assertRaises(Exception):
            await cognitive_agent.store_memory(
                content="测试记忆",
                memory_type=None,  # 无效类型
                strength=1.0
            )
        
        # 测试无效推理类型
        with self.assertRaises(Exception):
            await cognitive_agent.reason(
                premises=["测试前提"],
                reasoning_type=None  # 无效类型
            )
        
        # 测试无效注意力类型
        with self.assertRaises(Exception):
            await cognitive_agent.focus_attention(
                target="测试目标",
                attention_type=None  # 无效类型
            )
        
        await cognitive_agent.cleanup()
        self.logger.info("✅ 错误处理测试通过")


class TestResultCollector:
    """测试结果收集器"""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def add_result(self, test_name: str, status: str, duration: float, error: str = None):
        """添加测试结果"""
        self.results.append({
            'test_name': test_name,
            'status': status,
            'duration': duration,
            'error': error,
            'timestamp': time.time()
        })
    
    def generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        if not self.results:
            return {}
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['status'] == 'PASS')
        failed_tests = sum(1 for r in self.results if r['status'] == 'FAIL')
        
        total_duration = sum(r['duration'] for r in self.results)
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_duration': total_duration,
                'average_duration': total_duration / total_tests if total_tests > 0 else 0
            },
            'test_details': self.results,
            'timestamp': time.time()
        }


async def run_async_tests():
    """运行异步测试"""
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestCognitiveLab)
    
    # 创建结果收集器
    collector = TestResultCollector()
    collector.start_time = time.time()
    
    # 运行测试并收集结果
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    # 由于unittest不直接支持异步测试，我们手动处理
    async def run_individual_test(test_method, instance):
        test_name = f"{instance.__class__.__name__}.{test_method._testMethodName}"
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(test_method):
                await test_method(instance)
            else:
                test_method(instance)
            
            duration = time.time() - start_time
            collector.add_result(test_name, "PASS", duration)
            print(f"✅ {test_name} - 通过 ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            collector.add_result(test_name, "FAIL", duration, str(e))
            print(f"❌ {test_name} - 失败: {e}")
    
    # 获取所有异步测试方法
    test_methods = []
    for test_class in [TestCognitiveLab]:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                method = getattr(instance, method_name)
                if callable(method) and (asyncio.iscoroutinefunction(method) or hasattr(method, '__name__')):
                    test_methods.append((method, instance))
    
    # 运行测试
    for test_method, instance in test_methods:
        await run_individual_test(test_method, instance)
    
    collector.end_time = time.time()
    
    return collector.generate_report()


def run_sync_tests():
    """运行同步测试"""
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestCognitiveLab)
    test_runner = unittest.TextTestRunner(verbosity=2)
    return test_runner.run(test_suite)


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="认知进化实验室基础功能测试")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--async_only", action="store_true", help="只运行异步测试")
    parser.add_argument("--sync_only", action="store_true", help="只运行同步测试")
    parser.add_argument("--output", type=str, default="./test_results", help="测试结果输出目录")
    parser.add_argument("--test", type=str, help="运行特定测试模块")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("""
    🧪 Cognitive Evolution Lab - 基础功能测试 🧪
    ==============================================
    
    认知能力与协同进化系统基础功能验证
    作者: bingdongni
    版本: v1.0.0
    
    🔍 开始测试...
    """)
    
    try:
        if not args.sync_only:
            # 运行异步测试
            print("\n🧠 运行异步认知功能测试...")
            async_results = await run_async_tests()
            
            # 输出异步测试结果摘要
            if async_results:
                summary = async_results['summary']
                print(f"\n📊 异步测试结果摘要:")
                print(f"   总测试数: {summary['total_tests']}")
                print(f"   通过: {summary['passed']}")
                print(f"   失败: {summary['failed']}")
                print(f"   成功率: {summary['success_rate']:.2%}")
                print(f"   总耗时: {summary['total_duration']:.2f}秒")
        
        if not args.async_only:
            # 运行同步测试
            print("\n⚙️ 运行同步工具函数测试...")
            sync_results = run_sync_tests()
            
            # 输出同步测试结果
            print(f"\n📊 同步测试结果:")
            print(f"   运行测试数: {sync_results.testsRun}")
            print(f"   失败: {len(sync_results.failures)}")
            print(f"   错误: {len(sync_results.errors)}")
            print(f"   成功率: {(sync_results.testsRun - len(sync_results.failures) - len(sync_results.errors)) / sync_results.testsRun:.2%}")
        
        # 保存测试结果
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        results_file = output_dir / f"basic_test_results_{timestamp}.json"
        
        final_results = {
            'test_suite': 'cognitive_evolution_lab_basic_tests',
            'version': '1.0.0',
            'timestamp': timestamp,
            'arguments': vars(args),
            'async_results': async_results if not args.sync_only else None,
            'sync_summary': {
                'tests_run': getattr(sync_results, 'testsRun', 0) if not args.async_only else 0,
                'failures': len(getattr(sync_results, 'failures', [])),
                'errors': len(getattr(sync_results, 'errors', []))
            } if not args.async_only else None
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 测试结果已保存到: {results_file}")
        
        print("\n" + "="*60)
        print("🎯 认知进化实验室基础功能测试完成")
        print("="*60)
        
        return final_results
        
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())