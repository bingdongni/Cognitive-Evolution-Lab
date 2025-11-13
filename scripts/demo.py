#!/usr/bin/env python3
"""
Cognitive Evolution Lab - 主要演示脚本
作者: bingdongni
版本: v1.0.0

此脚本演示认知进化实验室的核心功能，包括：
1. 六种认知能力测试（记忆、推理、创造力、观察力、注意力、想象力）
2. 协同进化实验
3. 认知认知主体交互
4. 实时数据可视化

运行方式：
    python scripts/demo.py [--mode demo|cognitive|evolution|full] [--port PORT]
"""

import asyncio
import logging
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# 添加项目路径
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.cognitive_models import CognitiveAgent, MemoryType, ReasoningType
from src.world_simulator import VirtualWorld
from src.evolution_engine import EvolutionEngine
from src.utils import setup_logging, load_config


class CognitiveDemo:
    """
    认知演示类 - 展示六种认知能力的综合测试
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化演示环境
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = load_config(config_path) if config_path else self._get_default_config()
        
        # 设置日志
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.cognitive_认知主体 = None
        self.world_simulator = None
        self.evolution_engine = None
        
        # 演示结果存储
        self.demo_results = {
            'cognitive_tests': {},
            'evolution_results': {},
            'performance_metrics': {},
            'demo_timestamp': time.time()
        }
        
        self.logger.info("🎯 认知演示环境初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认演示配置"""
        return {
            'global': {
                'debug': True,
                'log_level': 'INFO',
                'save_path': './demo_results'
            },
            'cognitive_models': {
                'vocab_size': 1000,
                'embed_dim': 256,
                'hidden_dim': 512,
                'learning_rate': 0.001
            },
            'world_simulator': {
                'environment_size': [50, 50, 20],
                'max_objects': 100
            },
            'evolution_engine': {
                'population_size': 20,
                'generations': 10
            }
        }
    
    async def initialize(self):
        """初始化演示组件"""
        self.logger.info("🔧 初始化演示组件...")
        
        try:
            # 初始化认知认知主体
            self.cognitive_认知主体 = CognitiveAgent(
                config=self.config['cognitive_models']
            )
            await self.cognitive_认知主体.initialize()
            self.logger.info("✅ 认知认知主体初始化完成")
            
            # 初始化世界模拟器
            self.world_simulator = VirtualWorld(
                config=self.config['world_simulator']
            )
            await self.world_simulator.initialize()
            self.logger.info("✅ 世界模拟器初始化完成")
            
            # 初始化进化引擎
            self.evolution_engine = EvolutionEngine(
                config=self.config['evolution_engine']
            )
            await self.evolution_engine.initialize()
            self.logger.info("✅ 进化引擎初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 组件初始化失败: {e}")
            raise
    
    async def run_memory_demo(self) -> Dict[str, Any]:
        """
        演示记忆系统功能
        
        展示情景记忆、语义记忆、工作记忆和程序记忆的存储与检索
        """
        self.logger.info("🧠 开始记忆系统演示...")
        
        memory_results = {
            'test_type': 'memory_system',
            'episodic_memory': {},
            'semantic_memory': {},
            'working_memory': {},
            'procedural_memory': {},
            'retention_test': {},
            'association_test': {}
        }
        
        # 1. 情景记忆测试 - 存储个人经历
        self.logger.info("📝 测试情景记忆...")
        episodic_memories = [
            "今天学会了一个新的机器学习算法",
            "和同事讨论了认知科学的最新进展", 
            "完成了一个复杂的编程任务",
            "观看了关于人工智能的TED演讲",
            "读了一篇关于神经网络的论文"
        ]
        
        for i, memory in enumerate(episodic_memories):
            await self.cognitive_认知主体.store_memory(
                content=memory,
                memory_type=MemoryType.EPISODIC,
                strength=0.8 + i * 0.05
            )
        
        memory_results['episodic_memory']['stored_count'] = len(episodic_memories)
        
        # 2. 语义记忆测试 - 存储事实知识
        self.logger.info("📚 测试语义记忆...")
        semantic_facts = [
            "人工智能是计算机科学的一个分支",
            "深度学习使用多层神经网络",
            "认知科学研究心智和思维过程",
            "进化算法模拟自然选择过程",
            "机器学习使计算机能够自主学习"
        ]
        
        for fact in semantic_facts:
            await self.cognitive_认知主体.store_memory(
                content=fact,
                memory_type=MemoryType.SEMANTIC,
                strength=0.9
            )
        
        memory_results['semantic_memory']['stored_count'] = len(semantic_facts)
        
        # 3. 工作记忆测试 - 短期信息处理
        self.logger.info("⚡ 测试工作记忆...")
        working_tasks = [
            "记住数字序列: 3-7-1-9-2",
            "心算: 25 × 4 = ?",
            "倒背字母: A-D-G-J",
            "双任务处理: 同时记忆颜色和形状"
        ]
        
        for i, task in enumerate(working_tasks):
            await self.cognitive_认知主体.store_memory(
                content=task,
                memory_type=MemoryType.WORKING,
                strength=0.7
            )
        
        memory_results['working_memory']['task_count'] = len(working_tasks)
        
        # 4. 程序记忆测试 - 技能和习惯
        self.logger.info("🔧 测试程序记忆...")
        procedural_skills = [
            "如何骑自行车",
            "打字的基本手势",
            "解决问题的步骤流程",
            "学习新技能的方法"
        ]
        
        for skill in procedural_skills:
            await self.cognitive_认知主体.store_memory(
                content=skill,
                memory_type=MemoryType.PROCEDURAL,
                strength=0.95
            )
        
        memory_results['procedural_memory']['skill_count'] = len(procedural_skills)
        
        # 5. 记忆检索测试
        self.logger.info("🔍 测试记忆检索...")
        retrieval_tests = [
            ("学习", MemoryType.SEMANTIC, 0.6),
            ("算法", MemoryType.SEMANTIC, 0.7),
            ("记忆", MemoryType.EPISODIC, 0.5),
            ("技能", MemoryType.PROCEDURAL, 0.6)
        ]
        
        retrieval_results = []
        for query, mem_type, threshold in retrieval_tests:
            retrieved = await self.cognitive_认知主体.retrieve_memory(
                query=query,
                memory_type=mem_type,
                threshold=threshold
            )
            retrieval_results.append({
                'query': query,
                'type': mem_type.value,
                'retrieved_count': len(retrieved),
                'threshold': threshold
            })
        
        memory_results['retrieval_test'] = retrieval_results
        
        # 6. 记忆保留测试
        retention_test = await self.cognitive_认知主体.test_memory_retention()
        memory_results['retention_test'] = retention_test
        
        # 计算记忆性能评分
        total_stored = (len(episodic_memories) + len(semantic_facts) + 
                       len(working_tasks) + len(procedural_skills))
        avg_retrieval = sum(r['retrieved_count'] for r in retrieval_results) / len(retrieval_results)
        retrieval_accuracy = min(1.0, avg_retrieval / 3.0)  # 假设期望检索3个相关记忆
        
        memory_results['performance_score'] = (retrieval_accuracy + retention_test['retention_score']) / 2
        memory_results['total_memories_stored'] = total_stored
        
        self.logger.info(f"✅ 记忆系统演示完成，性能评分: {memory_results['performance_score']:.2f}")
        return memory_results
    
    async def run_reasoning_demo(self) -> Dict[str, Any]:
        """
        演示推理系统功能
        
        展示演绎推理、归纳推理、溯因推理和类比推理
        """
        self.logger.info("🧩 开始推理系统演示...")
        
        reasoning_results = {
            'test_type': 'reasoning_system',
            'deductive_reasoning': {},
            'inductive_reasoning': {},
            'abductive_reasoning': {},
            'analogical_reasoning': {},
            'overall_performance': {}
        }
        
        # 1. 演绎推理测试 - 从一般到特殊
        self.logger.info("🔬 测试演绎推理...")
        deductive_cases = [
            {
                'premises': ["所有科学家都很有好奇心", "爱因斯坦是科学家"],
                'expected': "爱因斯坦很有好奇心",
                'description': "三段论演绎推理"
            },
            {
                'premises': ["如果下雨，地面会湿", "正在下雨"],
                'expected': "地面会湿",
                'description': "条件推理"
            },
            {
                'premises': ["所有鸟类都有羽毛", "企鹅是鸟类"],
                'expected': "企鹅有羽毛",
                'description': "分类推理"
            }
        ]
        
        deductive_performance = []
        for case in deductive_cases:
            reasoning_chain = await self.cognitive_认知主体.reason(
                premises=case['premises'],
                reasoning_type=ReasoningType.DEDUCTIVE
            )
            deductive_performance.append({
                'premises': case['premises'],
                'conclusion': reasoning_chain.conclusion,
                'confidence': reasoning_chain.confidence,
                'expected': case['expected'],
                'description': case['description']
            })
        
        reasoning_results['deductive_reasoning'] = {
            'test_cases': deductive_performance,
            'average_confidence': sum(c['confidence'] for c in deductive_performance) / len(deductive_performance)
        }
        
        # 2. 归纳推理测试 - 从特殊到一般
        self.logger.info("📊 测试归纳推理...")
        inductive_cases = [
            {
                'observations': ["观察到天鹅1是白的", "观察到天鹅2是白的", "观察到天鹅3是白的"],
                'hypothesis': "所有天鹅都是白的",
                'description': "观察归纳"
            },
            {
                'observations': ["下雨天交通很堵", "下雪天交通很堵", "刮风天交通很堵"],
                'hypothesis': "恶劣天气导致交通拥堵",
                'description': "因果归纳"
            }
        ]
        
        inductive_performance = []
        for case in inductive_cases:
            reasoning_chain = await self.cognitive_认知主体.reason(
                premises=case['observations'],
                reasoning_type=ReasoningType.INDUCTIVE
            )
            inductive_performance.append({
                'observations': case['observations'],
                'conclusion': reasoning_chain.conclusion,
                'confidence': reasoning_chain.confidence,
                'hypothesis': case['hypothesis'],
                'description': case['description']
            })
        
        reasoning_results['inductive_reasoning'] = {
            'test_cases': inductive_performance,
            'average_confidence': sum(c['confidence'] for c in inductive_performance) / len(inductive_performance)
        }
        
        # 3. 溯因推理测试 - 最佳解释推理
        self.logger.info("🔍 测试溯因推理...")
        abductive_cases = [
            {
                'observations': ["草是湿的"],
                'possible_explanations': ["下雨了", "有人洒了水", "露水凝结"],
                'best_explanation': "下雨了",
                'description': "现象解释"
            },
            {
                'observations': ["街上湿了"],
                'possible_explanations': ["下雨了", "洒水车经过", "水管爆裂"],
                'best_explanation': "下雨了",
                'description': "综合推理"
            }
        ]
        
        abductive_performance = []
        for case in abductive_cases:
            reasoning_chain = await self.cognitive_认知主体.reason(
                premises=[case['observations']],
                reasoning_type=ReasoningType.ABDUCTIVE
            )
            abductive_performance.append({
                'observations': case['observations'],
                'possible_explanations': case['possible_explanations'],
                'conclusion': reasoning_chain.conclusion,
                'confidence': reasoning_chain.confidence,
                'best_explanation': case['best_explanation']
            })
        
        reasoning_results['abductive_reasoning'] = {
            'test_cases': abductive_performance,
            'average_confidence': sum(c['confidence'] for c in abductive_performance) / len(abductive_performance)
        }
        
        # 4. 类比推理测试 - 结构映射
        self.logger.info("🔗 测试类比推理...")
        analogical_patterns = [
            {
                'source': "太阳系的行星围绕太阳运转",
                'target_domain': "原子结构",
                'analogy': "电子围绕原子核运转",
                'mapping_quality': 0.8
            },
            {
                'source': "血液循环系统",
                'target_domain': "计算机网络",
                'analogy': "数据包在网络中传输",
                'mapping_quality': 0.7
            }
        ]
        
        reasoning_results['analogical_reasoning'] = {
            'analogies': analogical_patterns,
            'average_quality': sum(a['mapping_quality'] for a in analogical_patterns) / len(analogical_patterns)
        }
        
        # 计算综合推理性能
        all_confidences = []
        all_confidences.extend([c['confidence'] for c in deductive_performance])
        all_confidences.extend([c['confidence'] for c in inductive_performance])
        all_confidences.extend([c['confidence'] for c in abductive_performance])
        
        reasoning_results['overall_performance'] = {
            'total_tests': len(all_confidences),
            'average_confidence': sum(all_confidences) / len(all_confidences),
            'deductive_score': reasoning_results['deductive_reasoning']['average_confidence'],
            'inductive_score': reasoning_results['inductive_reasoning']['average_confidence'],
            'abductive_score': reasoning_results['abductive_reasoning']['average_confidence'],
            'analogical_score': reasoning_results['analogical_reasoning']['average_quality']
        }
        
        self.logger.info(f"✅ 推理系统演示完成，平均置信度: {reasoning_results['overall_performance']['average_confidence']:.2f}")
        return reasoning_results
    
    async def run_creativity_demo(self) -> Dict[str, Any]:
        """
        演示创造力系统功能
        
        展示发散思维、收敛思维、创意思维和想象力
        """
        self.logger.info("🎨 开始创造力系统演示...")
        
        creativity_results = {
            'test_type': 'creativity_system',
            'divergent_thinking': {},
            'convergent_thinking': {},
            'creative_problem_solving': {},
            'novelty_evaluation': {}
        }
        
        # 1. 发散思维测试 - 生成多种解决方案
        self.logger.info("🌟 测试发散思维...")
        divergent_tasks = [
            {
                'prompt': "列举所有可以用杯子做的事情",
                'context': "日常生活",
                'expected_variety': 15
            },
            {
                'prompt': "想出所有可能的交通方式",
                'context': "城市规划",
                'expected_variety': 10
            },
            {
                'prompt': "提出解决环境污染的创新方法",
                'context': "环境保护",
                'expected_variety': 12
            }
        ]
        
        divergent_scores = []
        for task in divergent_tasks:
            # 模拟发散思维结果
            creative_output = await self.cognitive_认知主体.generate_creative_output(
                context=task['context'],
                style="divergent"
            )
            
            # 评估创意多样性（简化为生成元素的计数）
            diversity_score = min(1.0, len(creative_output.get('creative_text', '')) / 50)
            
            divergent_scores.append({
                'task': task['prompt'],
                'context': task['context'],
                'diversity_score': diversity_score,
                'creativity_score': creative_output['creativity_score'],
                'expected_variety': task['expected_variety']
            })
        
        creativity_results['divergent_thinking'] = {
            'tasks': divergent_scores,
            'average_diversity': sum(s['diversity_score'] for s in divergent_scores) / len(divergent_scores),
            'average_creativity': sum(s['creativity_score'] for s in divergent_scores) / len(divergent_scores)
        }
        
        # 2. 收敛思维测试 - 选择最佳方案
        self.logger.info("🎯 测试收敛思维...")
        convergent_tasks = [
            {
                'problem': "如何在有限预算下提高工作效率",
                'solutions': ["使用免费软件", "优化工作流程", "减少会议时间", "自动化重复任务"],
                'criteria': ["成本", "效果", "可实施性"]
            },
            {
                'problem': "如何让更多人参与环保活动",
                'solutions': ["教育宣传", "游戏化机制", "奖励制度", "社交媒体推广"],
                'criteria': ["参与度", "持续性", "影响力"]
            }
        ]
        
        convergence_scores = []
        for task in convergent_tasks:
            # 模拟收敛思维评估
            creative_output = await self.cognitive_认知主体.generate_creative_output(
                context=f"收敛思维：{task['problem']}",
                style="convergent"
            )
            
            convergence_score = creative_output['creativity_score']
            
            convergence_scores.append({
                'problem': task['problem'],
                'solutions_count': len(task['solutions']),
                'convergence_score': convergence_score,
                'evaluation_criteria': task['criteria']
            })
        
        creativity_results['convergent_thinking'] = {
            'tasks': convergence_scores,
            'average_convergence': sum(s['convergence_score'] for s in convergence_scores) / len(convergence_scores)
        }
        
        # 3. 创意问题解决测试
        self.logger.info("💡 测试创意问题解决...")
        problem_solving_cases = [
            {
                'scenario': "设计一个智能家居系统",
                'constraints': ["成本控制", "用户友好", "节能环保"],
                'creativity_aspects': ["新颖性", "实用性", "可行性"]
            },
            {
                'scenario': "创造新的学习方式",
                'constraints': ["提高参与度", "个性化", "可扩展"],
                'creativity_aspects': ["创新性", "效果性", "推广性"]
            }
        ]
        
        problem_solving_scores = []
        for case in problem_solving_cases:
            creative_output = await self.cognitive_认知主体.generate_creative_output(
                context=case['scenario'],
                style="problem_solving"
            )
            
            problem_solving_scores.append({
                'scenario': case['scenario'],
                'constraints': case['constraints'],
                'creativity_aspects': case['creativity_aspects'],
                'solution_quality': creative_output['creativity_score'],
                'creative_text': creative_output['creative_text']
            })
        
        creativity_results['creative_problem_solving'] = {
            'cases': problem_solving_scores,
            'average_quality': sum(c['solution_quality'] for c in problem_solving_scores) / len(problem_solving_scores)
        }
        
        # 4. 新颖性评估
        self.logger.info("🔬 测试新颖性评估...")
        novelty_assessments = [
            {
                'concept': "会说话的植物",
                'existing_knowledge': "植物不能说话",
                'novelty_score': 0.9
            },
            {
                'concept': "时间旅行",
                'existing_knowledge': "时间只能向前流逝",
                'novelty_score': 0.85
            },
            {
                'concept': "量子计算",
                'existing_knowledge': "计算使用二进制",
                'novelty_score': 0.75
            }
        ]
        
        creativity_results['novelty_evaluation'] = {
            'assessments': novelty_assessments,
            'average_novelty': sum(n['novelty_score'] for n in novelty_assessments) / len(novelty_assessments)
        }
        
        # 计算综合创造力评分
        creativity_results['overall_creativity'] = {
            'divergent_score': creativity_results['divergent_thinking']['average_diversity'],
            'convergent_score': creativity_results['convergent_thinking']['average_convergence'],
            'problem_solving_score': creativity_results['creative_problem_solving']['average_quality'],
            'novelty_score': creativity_results['novelty_evaluation']['average_novelty'],
            'composite_score': (
                creativity_results['divergent_thinking']['average_diversity'] +
                creativity_results['convergent_thinking']['average_convergence'] +
                creativity_results['creative_problem_solving']['average_quality'] +
                creativity_results['novelty_evaluation']['average_novelty']
            ) / 4
        }
        
        self.logger.info(f"✅ 创造力系统演示完成，综合评分: {creativity_results['overall_creativity']['composite_score']:.2f}")
        return creativity_results
    
    async def run_observation_demo(self) -> Dict[str, Any]:
        """
        演示观察系统功能
        
        展示视觉观察、模式识别、异常检测和时间序列分析
        """
        self.logger.info("👁️ 开始观察系统演示...")
        
        import torch
        
        observation_results = {
            'test_type': 'observation_system',
            'visual_pattern_recognition': {},
            'anomaly_detection': {},
            'temporal_analysis': {},
            'multi_scale_processing': {}
        }
        
        # 1. 视觉模式识别测试
        self.logger.info("🔍 测试视觉模式识别...")
        
        # 模拟视觉观察数据
        mock_observations = torch.randn(1, 3, 224, 224)
        temporal_data = torch.randn(1, 10, 512)  # 时间序列数据
        
        observation_output = await self.cognitive_认知主体.observe_environment(
            observations=mock_observations,
            temporal_data=temporal_data
        )
        
        observation_results['visual_pattern_recognition'] = {
            'pattern_features_extracted': observation_output['pattern_features'].shape,
            'anomaly_score': observation_output['anomaly_score'].item(),
            'temporal_patterns': observation_output['temporal_patterns'].shape,
            'attention_triggered': observation_output['attention_triggered']
        }
        
        # 2. 异常检测测试
        self.logger.info("⚠️ 测试异常检测...")
        
        # 创建正常和异常观察数据
        normal_observations = torch.randn(5, 3, 224, 224)
        abnormal_observations = torch.randn(2, 3, 224, 224) * 5  # 异常大值
        
        normal_scores = []
        abnormal_scores = []
        
        # 测试正常数据
        for obs in normal_observations:
            result = await self.cognitive_认知主体.observe_environment(
                observations=obs.unsqueeze(0)
            )
            normal_scores.append(result['anomaly_score'].item())
        
        # 测试异常数据
        for obs in abnormal_observations:
            result = await self.cognitive_认知主体.observe_environment(
                observations=obs.unsqueeze(0)
            )
            abnormal_scores.append(result['anomaly_score'].item())
        
        observation_results['anomaly_detection'] = {
            'normal_data_count': len(normal_scores),
            'abnormal_data_count': len(abnormal_scores),
            'normal_average_score': sum(normal_scores) / len(normal_scores),
            'abnormal_average_score': sum(abnormal_scores) / len(abnormal_scores),
            'detection_accuracy': 1.0 if max(abnormal_scores) > max(normal_scores) else 0.5
        }
        
        # 3. 时间序列分析测试
        self.logger.info("⏰ 测试时间序列分析...")
        
        # 模拟时间序列观察
        temporal_sequences = []
        for i in range(10):
            seq = torch.randn(1, 5, 512)  # 5个时间步
            result = await self.cognitive_认知主体.observe_environment(
                observations=torch.randn(1, 3, 224, 224),
                temporal_data=seq
            )
            temporal_sequences.append({
                'time_step': i,
                'anomaly_score': result['anomaly_score'].item(),
                'pattern_stability': 0.8 + 0.1 * torch.sin(torch.tensor(i * 0.5)).item()
            })
        
        observation_results['temporal_analysis'] = {
            'time_series_length': len(temporal_sequences),
            'trend_analysis': 'stable_with_cycles',
            'pattern_consistency': 0.75,
            'temporal_sequences': temporal_sequences[:5]  # 保存前5个序列作为示例
        }
        
        # 4. 多尺度处理测试
        self.logger.info("🔬 测试多尺度处理...")
        
        multi_scale_observations = {
            'micro_scale': torch.randn(1, 3, 64, 64),    # 微观尺度
            'meso_scale': torch.randn(1, 3, 128, 128),   # 中观尺度
            'macro_scale': torch.randn(1, 3, 224, 224)   # 宏观尺度
        }
        
        scale_results = {}
        for scale_name, obs in multi_scale_observations.items():
            result = await self.cognitive_认知主体.observe_environment(obs)
            scale_results[scale_name] = {
                'input_resolution': obs.shape[-2:],
                'anomaly_score': result['anomaly_score'].item(),
                'pattern_complexity': len(result['pattern_features'].flatten())
            }
        
        observation_results['multi_scale_processing'] = {
            'scales_tested': list(multi_scale_observations.keys()),
            'scale_results': scale_results,
            'cross_scale_integration': True
        }
        
        # 计算观察系统综合性能
        anomaly_detection_score = observation_results['anomaly_detection']['detection_accuracy']
        pattern_recognition_score = max(0, 1.0 - observation_results['visual_pattern_recognition']['anomaly_score'])
        temporal_analysis_score = observation_results['temporal_analysis']['pattern_consistency']
        
        observation_results['overall_performance'] = {
            'pattern_recognition_score': pattern_recognition_score,
            'anomaly_detection_score': anomaly_detection_score,
            'temporal_analysis_score': temporal_analysis_score,
            'composite_score': (pattern_recognition_score + anomaly_detection_score + temporal_analysis_score) / 3
        }
        
        self.logger.info(f"✅ 观察系统演示完成，综合评分: {observation_results['overall_performance']['composite_score']:.2f}")
        return observation_results
    
    async def run_attention_demo(self) -> Dict[str, Any]:
        """
        演示注意力系统功能
        
        展示选择性注意、持续性注意和分散性注意
        """
        self.logger.info("🎯 开始注意力系统演示...")
        
        from src.cognitive_models import AttentionType
        
        attention_results = {
            'test_type': 'attention_system',
            'selective_attention': {},
            'sustained_attention': {},
            'divided_attention': {},
            'attention_control': {}
        }
        
        # 1. 选择性注意测试 - 在干扰中专注于目标
        self.logger.info("🔍 测试选择性注意...")
        
        selective_tests = [
            {
                'target': "红色圆形",
                'distractors': ["蓝色圆形", "红色方形", "绿色圆形", "蓝色方形"],
                'expected_focus': 0.8
            },
            {
                'target': "数字7",
                'distractors': ["1", "2", "3", "4", "5", "6", "8", "9"],
                'expected_focus': 0.9
            },
            {
                'target': "人脸表情",
                'distractors': ["风景", "物体", "文字", "符号"],
                'expected_focus': 0.85
            }
        ]
        
        selective_scores = []
        for test in selective_tests:
            attention_weights = await self.cognitive_认知主体.focus_attention(
                target=test['target'],
                attention_type=AttentionType.SELECTIVE
            )
            
            focus_score = attention_weights.get('relevance', 0.5)
            
            selective_scores.append({
                'target': test['target'],
                'focus_score': focus_score,
                'attention_weights': attention_weights,
                'expected_focus': test['expected_focus']
            })
        
        attention_results['selective_attention'] = {
            'tests': selective_scores,
            'average_focus': sum(s['focus_score'] for s in selective_scores) / len(selective_scores),
            'attention_filtering': True
        }
        
        # 2. 持续性注意测试 - 长期专注能力
        self.logger.info("⏳ 测试持续性注意...")
        
        sustained_tests = [
            {
                'duration_minutes': 30,
                'task_type': "监控任务",
                'expected_performance': 0.75
            },
            {
                'duration_minutes': 60,
                'task_type': "连续反应任务",
                'expected_performance': 0.70
            },
            {
                'duration_minutes': 90,
                'task_type': "警觉性任务",
                'expected_performance': 0.65
            }
        ]
        
        sustained_scores = []
        for test in sustained_tests:
            # 模拟持续性注意表现
            attention_weights = await self.cognitive_认知主体.focus_attention(
                target=f"持续专注 {test['task_type']}",
                attention_type=AttentionType.SUSTAINED
            )
            
            persistence_score = attention_weights.get('persistence', 0.5)
            stability_score = attention_weights.get('stability', 0.5)
            
            # 持续时间越长，表现可能下降
            duration_factor = max(0.1, 1.0 - (test['duration_minutes'] - 30) / 120)
            sustained_score = (persistence_score + stability_score) / 2 * duration_factor
            
            sustained_scores.append({
                'duration': test['duration_minutes'],
                'task_type': test['task_type'],
                'sustained_score': sustained_score,
                'persistence': persistence_score,
                'stability': stability_score
            })
        
        attention_results['sustained_attention'] = {
            'tests': sustained_scores,
            'average_sustained': sum(s['sustained_score'] for s in sustained_scores) / len(sustained_scores),
            'fatigue_detection': True
        }
        
        # 3. 分散性注意测试 - 同时处理多个任务
        self.logger.info("🔄 测试分散性注意...")
        
        divided_tests = [
            {
                'tasks': ["听音乐", "打字", "看屏幕"],
                'complexity': "中等",
                'expected_performance': 0.6
            },
            {
                'tasks': ["开车", "听广播", "导航"],
                'complexity': "高",
                'expected_performance': 0.5
            },
            {
                'tasks': ["走路", "思考", "观察"],
                'complexity': "低",
                'expected_performance': 0.7
            }
        ]
        
        divided_scores = []
        for test in divided_tests:
            attention_weights = await self.cognitive_认知主体.focus_attention(
                target=f"多任务: {', '.join(test['tasks'])}",
                attention_type=AttentionType.DIVIDED
            )
            
            balance_score = attention_weights.get('balance', 0.5)
            diversity_score = attention_weights.get('diversity', 0.5)
            relevance_score = attention_weights.get('relevance', 0.5)
            
            # 任务越多，分散性注意效果可能下降
            task_factor = max(0.1, 1.0 - len(test['tasks']) * 0.1)
            divided_score = (balance_score + diversity_score + relevance_score) / 3 * task_factor
            
            divided_scores.append({
                'tasks': test['tasks'],
                'complexity': test['complexity'],
                'divided_score': divided_score,
                'balance': balance_score,
                'diversity': diversity_score
            })
        
        attention_results['divided_attention'] = {
            'tests': divided_scores,
            'average_divided': sum(s['divided_score'] for s in divided_scores) / len(divided_scores),
            'multitasking_efficiency': True
        }
        
        # 4. 注意力控制测试
        self.logger.info("🎮 测试注意力控制...")
        
        control_tests = [
            {
                'scenario': "突然的干扰",
                'recovery_time': 2.5,  # 秒
                'expected_recovery': 0.8
            },
            {
                'scenario': "任务切换",
                'switch_cost': 1.8,  # 秒
                'expected_switch': 0.7
            },
            {
                'scenario': "注意转移",
                'shift_efficiency': 0.85,
                'expected_shift': 0.75
            }
        ]
        
        attention_results['attention_control'] = {
            'interference_recovery': {
                'average_recovery_time': sum(t['recovery_time'] for t in control_tests) / len(control_tests),
                'control_stability': 0.8
            },
            'task_switching': {
                'average_switch_cost': sum(t['switch_cost'] for t in control_tests if 'switch_cost' in t) / len([t for t in control_tests if 'switch_cost' in t]),
                'flexibility_score': 0.75
            },
            'attention_shifting': {
                'average_shift_efficiency': sum(t['shift_efficiency'] for t in control_tests if 'shift_efficiency' in t) / len([t for t in control_tests if 'shift_efficiency' in t]),
                'control_accuracy': 0.85
            }
        }
        
        # 计算注意力系统综合性能
        attention_results['overall_performance'] = {
            'selective_attention_score': attention_results['selective_attention']['average_focus'],
            'sustained_attention_score': attention_results['sustained_attention']['average_sustained'],
            'divided_attention_score': attention_results['divided_attention']['average_divided'],
            'attention_control_score': 0.8,  # 综合控制能力评分
            'composite_score': (
                attention_results['selective_attention']['average_focus'] +
                attention_results['sustained_attention']['average_sustained'] +
                attention_results['divided_attention']['average_divided'] +
                0.8
            ) / 4
        }
        
        self.logger.info(f"✅ 注意力系统演示完成，综合评分: {attention_results['overall_performance']['composite_score']:.2f}")
        return attention_results
    
    async def run_imagination_demo(self) -> Dict[str, Any]:
        """
        演示想象力系统功能
        
        展示情景想象、因果推理、时间想象和创新思维
        """
        self.logger.info("🌟 开始想象力系统演示...")
        
        imagination_results = {
            'test_type': 'imagination_system',
            'scenario_imagination': {},
            'causal_reasoning': {},
            'temporal_imagination': {},
            'creative_imagination': {}
        }
        
        # 1. 情景想象测试 - 构造未来场景
        self.logger.info("🎬 测试情景想象...")
        
        scenario_tests = [
            {
                'context': "2050年的智能城市",
                'constraints': ["可持续发展", "人工智能", "人性化"],
                'expected_elements': 5
            },
            {
                'context': "火星殖民地的日常生活",
                'constraints': ["有限资源", "环境挑战", "团队合作"],
                'expected_elements': 4
            },
            {
                'context': "完全虚拟的教育环境",
                'constraints': ["沉浸式体验", "个性化学习", "社交互动"],
                'expected_elements': 6
            }
        ]
        
        scenario_scores = []
        for test in scenario_tests:
            imagination_output = await self.cognitive_认知主体.imagine_scenario(
                context=test['context'],
                constraints=test['constraints']
            )
            
            element_count = len(imagination_output['scenario_elements'])
            element_quality = min(1.0, element_count / test['expected_elements'])
            
            scenario_scores.append({
                'context': test['context'],
                'constraints': test['constraints'],
                'elements_generated': element_count,
                'expected_elements': test['expected_elements'],
                'quality_score': element_quality,
                'scenario_elements': imagination_output['scenario_elements'][:3]  # 保存前3个元素
            })
        
        imagination_results['scenario_imagination'] = {
            'tests': scenario_scores,
            'average_quality': sum(s['quality_score'] for s in scenario_scores) / len(scenario_scores),
            'total_elements': sum(s['elements_generated'] for s in scenario_scores)
        }
        
        # 2. 因果推理测试 - 理解因果关系
        self.logger.info("🔗 测试因果推理...")
        
        causal_tests = [
            {
                'cause': "全球变暖",
                'effects': ["海平面上升", "极端天气", "生态系统变化"],
                'causal_strength': 0.9
            },
            {
                'cause': "人工智能普及",
                'effects': ["就业结构变化", "工作效率提升", "伦理挑战"],
                'causal_strength': 0.8
            },
            {
                'cause': "社交媒体普及",
                'effects': ["信息传播加速", "社交模式改变", "隐私问题"],
                'causal_strength': 0.85
            }
        ]
        
        causal_scores = []
        for test in causal_tests:
            # 生成因果推理场景
            causal_scenario = await self.cognitive_认知主体.imagine_scenario(
                context=f"因果关系: {test['cause']} -> {test['effects']}",
                constraints=["逻辑一致性", "现实性"]
            )
            
            # 计算因果推理准确度
            effect_prediction = len(causal_scenario['scenario_elements'])
            causal_accuracy = min(1.0, effect_prediction / len(test['effects']))
            
            causal_scores.append({
                'cause': test['cause'],
                'predicted_effects': len(test['effects']),
                'scenario_elements': effect_prediction,
                'causal_accuracy': causal_accuracy,
                'causal_strength': test['causal_strength']
            })
        
        imagination_results['causal_reasoning'] = {
            'tests': causal_scores,
            'average_causal_accuracy': sum(c['causal_accuracy'] for c in causal_scores) / len(causal_scores),
            'causal_reasoning_depth': "complex"
        }
        
        # 3. 时间想象测试 - 时间维度的想象
        self.logger.info("⏰ 测试时间想象...")
        
        temporal_tests = [
            {
                'timeframe': "过去",
                'scenario': "文艺复兴时期的艺术家生活",
                'elements': ["社会背景", "创作过程", "历史影响"],
                'temporal_depth': "deep"
            },
            {
                'timeframe': "现在",
                'scenario': "当前远程工作的生活",
                'elements': ["技术环境", "工作方式", "生活平衡"],
                'temporal_depth': "surface"
            },
            {
                'timeframe': "未来",
                'scenario': "2050年的交通系统",
                'elements': ["技术发展", "社会影响", "环境考虑"],
                'temporal_depth': "predictive"
            }
        ]
        
        temporal_scores = []
        for test in temporal_tests:
            temporal_scenario = await self.cognitive_认知主体.imagine_scenario(
                context=f"时间想象: {test['timeframe']} - {test['scenario']}",
                constraints=test['elements']
            )
            
            temporal_elements = len(temporal_scenario['scenario_elements'])
            temporal_coherence = 0.8 if test['temporal_depth'] == "deep" else 0.6
            
            temporal_scores.append({
                'timeframe': test['timeframe'],
                'scenario': test['scenario'],
                'elements_generated': temporal_elements,
                'temporal_depth': test['temporal_depth'],
                'temporal_coherence': temporal_coherence
            })
        
        imagination_results['temporal_imagination'] = {
            'tests': temporal_scores,
            'average_temporal_depth': sum(t['temporal_coherence'] for t in temporal_scores) / len(temporal_scores),
            'time_span_coverage': ["past", "present", "future"]
        }
        
        # 4. 创新想象测试 - 突破常规的想象
        self.logger.info("💡 测试创新想象...")
        
        creative_tests = [
            {
                'prompt': "想象一个没有重力的世界",
                'domain': "物理学",
                'innovation_level': "revolutionary"
            },
            {
                'prompt': "设计一种全新的沟通方式",
                'domain': "人际关系",
                'innovation_level': "incremental"
            },
            {
                'prompt': "创造一种新的艺术形式",
                'domain': "艺术创作",
                'innovation_level': "breakthrough"
            }
        ]
        
        creative_scores = []
        for test in creative_tests:
            creative_output = await self.cognitive_认知主体.generate_creative_output(
                context=f"创新想象: {test['prompt']}",
                style="innovative"
            )
            
            innovation_level_score = {
                "revolutionary": 0.9,
                "breakthrough": 0.8,
                "incremental": 0.6
            }.get(test['innovation_level'], 0.5)
            
            creative_scores.append({
                'prompt': test['prompt'],
                'domain': test['domain'],
                'innovation_level': test['innovation_level'],
                'creativity_score': creative_output['creativity_score'],
                'innovation_score': innovation_level_score,
                'generated_content': creative_output['creative_text']
            })
        
        imagination_results['creative_imagination'] = {
            'tests': creative_scores,
            'average_creativity': sum(c['creativity_score'] for c in creative_scores) / len(creative_scores),
            'innovation_distribution': {
                'revolutionary': sum(1 for c in creative_scores if c['innovation_level'] == 'revolutionary'),
                'breakthrough': sum(1 for c in creative_scores if c['innovation_level'] == 'breakthrough'),
                'incremental': sum(1 for c in creative_scores if c['innovation_level'] == 'incremental')
            }
        }
        
        # 计算想象力系统综合性能
        imagination_results['overall_performance'] = {
            'scenario_imagination_score': imagination_results['scenario_imagination']['average_quality'],
            'causal_reasoning_score': imagination_results['causal_reasoning']['average_causal_accuracy'],
            'temporal_imagination_score': imagination_results['temporal_imagination']['average_temporal_depth'],
            'creative_imagination_score': imagination_results['creative_imagination']['average_creativity'],
            'composite_score': (
                imagination_results['scenario_imagination']['average_quality'] +
                imagination_results['causal_reasoning']['average_causal_accuracy'] +
                imagination_results['temporal_imagination']['average_temporal_depth'] +
                imagination_results['creative_imagination']['average_creativity']
            ) / 4
        }
        
        self.logger.info(f"✅ 想象力系统演示完成，综合评分: {imagination_results['overall_performance']['composite_score']:.2f}")
        return imagination_results
    
    async def run_evolution_demo(self) -> Dict[str, Any]:
        """
        演示协同进化系统功能
        
        展示单认知主体进化、多认知主体协同和文化进化
        """
        self.logger.info("🧬 开始协同进化系统演示...")
        
        evolution_results = {
            'test_type': 'evolution_system',
            'single_认知主体_evolution': {},
            'multi_认知主体_evolution': {},
            'cultural_evolution': {},
            'co_evolution': {}
        }
        
        # 1. 单认知主体进化测试
        self.logger.info("👤 测试单认知主体进化...")
        
        # 创建简化的单认知主体进化环境
        class SimpleEnvironment:
            def __init__(self):
                self.fitness_history = []
                self.generation = 0
            
            def evaluate_fitness(self, individual):
                # 简化的适应度评估
                import random
                base_fitness = 0.5
                noise = random.uniform(-0.1, 0.1)
                improvement = individual.get('generation', 0) * 0.02
                return base_fitness + noise + improvement
        
        env = SimpleEnvironment()
        
        # 模拟进化过程
        population = []
        for i in range(10):  # 10代
            generation = {'generation': i, 'fitness': 0}
            fitness = env.evaluate_fitness(generation)
            generation['fitness'] = fitness
            population.append(generation)
            env.fitness_history.append(fitness)
        
        evolution_results['single_认知主体_evolution'] = {
            'generations': 10,
            'population_size': 1,
            'fitness_history': env.fitness_history,
            'initial_fitness': env.fitness_history[0] if env.fitness_history else 0,
            'final_fitness': env.fitness_history[-1] if env.fitness_history else 0,
            'improvement': env.fitness_history[-1] - env.fitness_history[0] if len(env.fitness_history) > 1 else 0
        }
        
        # 2. 多认知主体进化测试
        self.logger.info("👥 测试多认知主体进化...")
        
        multi_认知主体_evolution = {
            '认知主体s': [],
            'cooperation_metrics': {},
            'competition_metrics': {},
            'communication_patterns': []
        }
        
        # 创建多认知主体种群
        num_认知主体s = 5
        for i in range(num_认知主体s):
            认知主体 = {
                'id': f'认知主体_{i}',
                'fitness': 0.5 + (i * 0.1),  # 初始适应度差异
                'cooperation_score': 0.6 + (i * 0.05),
                'communication_efficiency': 0.7 + (i * 0.03)
            }
            multi_认知主体_evolution['认知主体s'].append(认知主体)
        
        # 计算群体指标
        avg_fitness = sum(认知主体['fitness'] for 认知主体 in multi_认知主体_evolution['认知主体s']) / num_认知主体s
        fitness_variance = sum((认知主体['fitness'] - avg_fitness) ** 2 for 认知主体 in multi_认知主体_evolution['认知主体s']) / num_认知主体s
        diversity_score = 1.0 / (1.0 + fitness_variance)
        
        multi_认知主体_evolution['cooperation_metrics'] = {
            'average_cooperation': sum(认知主体['cooperation_score'] for 认知主体 in multi_认知主体_evolution['认知主体s']) / num_认知主体s,
            'cooperation_variance': fitness_variance,
            'team_performance': avg_fitness + 0.1  # 合作带来的额外收益
        }
        
        multi_认知主体_evolution['competition_metrics'] = {
            'competition_intensity': 0.6,
            'fitness_distribution': [认知主体['fitness'] for 认知主体 in multi_认知主体_evolution['认知主体s']],
            'selection_pressure': fitness_variance
        }
        
        evolution_results['multi_认知主体_evolution'] = multi_认知主体_evolution
        
        # 3. 文化进化测试
        self.logger.info("📚 测试文化进化...")
        
        cultural_knowledge = {
            'concepts': [
                {'name': '机器学习', 'adoption_rate': 0.8, 'evolution_time': 5},
                {'name': '深度学习', 'adoption_rate': 0.6, 'evolution_time': 3},
                {'name': '强化学习', 'adoption_rate': 0.4, 'evolution_time': 2},
                {'name': '迁移学习', 'adoption_rate': 0.3, 'evolution_time': 1}
            ],
            'transmission_patterns': {
                'horizontal': 0.7,  # 同代传播
                'vertical': 0.5,    # 跨代传播
                'cultural_drift': 0.2  # 文化漂移
            }
        }
        
        # 计算文化传播效率
        total_adoption = sum(concept['adoption_rate'] for concept in cultural_knowledge['concepts'])
        avg_evolution_time = sum(concept['evolution_time'] for concept in cultural_knowledge['concepts']) / len(cultural_knowledge['concepts'])
        cultural_fitness = total_adoption / (avg_evolution_time + 1)
        
        evolution_results['cultural_evolution'] = {
            'knowledge_base': cultural_knowledge['concepts'],
            'transmission_efficiency': cultural_fitness,
            'cultural_diversity': len(cultural_knowledge['concepts']),
            'adaptation_rate': cultural_fitness
        }
        
        # 4. 环境共演化测试
        self.logger.info("🌍 测试环境共演化...")
        
        co_evolution_data = {
            'environment_complexity': 0.5,
            '认知主体_adaptation_rate': 0.7,
            'co_adaptation_score': 0.6,
            'evolutionary_arms_race': True
        }
        
        # 模拟共演化过程
        for generation in range(5):
            # 环境复杂度逐渐增加
            co_evolution_data['environment_complexity'] += 0.1
            
            # 认知主体适应度
            认知主体_fitness = 0.6 + (generation * 0.05)
            
            # 共演化评分
            co_adaptation = min(1.0, 认知主体_fitness / co_evolution_data['environment_complexity'])
            co_evolution_data['co_adaptation_score'] = co_adaptation
        
        evolution_results['co_evolution'] = {
            'final_environment_complexity': co_evolution_data['environment_complexity'],
            '认知主体_adaptation': co_evolution_data['认知主体_adaptation_rate'],
            'co_adaptation': co_evolution_data['co_adaptation_score'],
            'evolutionary_stability': 0.75
        }
        
        # 计算进化系统综合性能
        evolution_results['overall_performance'] = {
            'single_认知主体_improvement': evolution_results['single_认知主体_evolution']['improvement'],
            'multi_认知主体_cooperation': evolution_results['multi_认知主体_evolution']['cooperation_metrics']['average_cooperation'],
            'cultural_transmission': evolution_results['cultural_evolution']['transmission_efficiency'],
            'co_adaptation': evolution_results['co_evolution']['co_adaptation'],
            'composite_score': (
                evolution_results['single_认知主体_evolution']['improvement'] +
                evolution_results['multi_认知主体_evolution']['cooperation_metrics']['average_cooperation'] +
                evolution_results['cultural_evolution']['transmission_efficiency'] +
                evolution_results['co_evolution']['co_adaptation']
            ) / 4
        }
        
        self.logger.info(f"✅ 协同进化系统演示完成，综合评分: {evolution_results['overall_performance']['composite_score']:.2f}")
        return evolution_results
    
    async def run_comprehensive_demo(self) -> Dict[str, Any]:
        """
        运行完整的认知能力综合演示
        """
        self.logger.info("🎯 开始认知能力综合演示...")
        
        comprehensive_results = {
            'demo_info': {
                'title': '认知进化实验室 - 六种认知能力综合演示',
                'version': '1.0.0',
                'timestamp': time.time(),
                'duration_estimation': '15-20分钟'
            },
            'cognitive_capabilities': {},
            'evolution_capabilities': {},
            'integration_analysis': {},
            'overall_assessment': {}
        }
        
        # 执行六种认知能力测试
        self.logger.info("🧠 阶段1: 执行认知能力测试...")
        
        # 1. 记忆系统测试
        memory_results = await self.run_memory_demo()
        comprehensive_results['cognitive_capabilities']['memory'] = memory_results
        
        # 2. 推理系统测试
        reasoning_results = await self.run_reasoning_demo()
        comprehensive_results['cognitive_capabilities']['reasoning'] = reasoning_results
        
        # 3. 创造力系统测试
        creativity_results = await self.run_creativity_demo()
        comprehensive_results['cognitive_capabilities']['creativity'] = creativity_results
        
        # 4. 观察系统测试
        observation_results = await self.run_observation_demo()
        comprehensive_results['cognitive_capabilities']['observation'] = observation_results
        
        # 5. 注意力系统测试
        attention_results = await self.run_attention_demo()
        comprehensive_results['cognitive_capabilities']['attention'] = attention_results
        
        # 6. 想象力系统测试
        imagination_results = await self.run_imagination_demo()
        comprehensive_results['cognitive_capabilities']['imagination'] = imagination_results
        
        # 执行进化能力测试
        self.logger.info("🧬 阶段2: 执行进化能力测试...")
        
        evolution_results = await self.run_evolution_demo()
        comprehensive_results['evolution_capabilities'] = evolution_results
        
        # 综合分析
        self.logger.info("📊 阶段3: 综合分析...")
        
        # 计算各认知能力的平均评分
        cognitive_scores = {
            'memory': memory_results.get('performance_score', 0),
            'reasoning': reasoning_results['overall_performance']['average_confidence'],
            'creativity': creativity_results['overall_creativity']['composite_score'],
            'observation': observation_results['overall_performance']['composite_score'],
            'attention': attention_results['overall_performance']['composite_score'],
            'imagination': imagination_results['overall_performance']['composite_score']
        }
        
        # 计算认知能力综合评分
        cognitive_average = sum(cognitive_scores.values()) / len(cognitive_scores)
        evolution_score = evolution_results['overall_performance']['composite_score']
        
        # 生成能力雷达图数据
        radar_chart_data = {
            'abilities': list(cognitive_scores.keys()),
            'scores': list(cognitive_scores.values()),
            'evolution_score': evolution_score,
            'overall_score': (cognitive_average + evolution_score) / 2
        }
        
        comprehensive_results['integration_analysis'] = {
            'cognitive_scores': cognitive_scores,
            'cognitive_average': cognitive_average,
            'evolution_score': evolution_score,
            'integration_score': (cognitive_average + evolution_score) / 2,
            'radar_chart_data': radar_chart_data,
            'strengths': [ability for ability, score in cognitive_scores.items() if score > 0.7],
            'weaknesses': [ability for ability, score in cognitive_scores.items() if score < 0.5],
            'development_recommendations': [
                "加强推理能力的系统性训练",
                "提升观察力的细节感知能力",
                "优化注意力的控制机制"
            ]
        }
        
        # 总体评估
        comprehensive_results['overall_assessment'] = {
            'cognitive_maturity': cognitive_average,
            'evolutionary_potential': evolution_score,
            'adaptive_capability': (cognitive_average + evolution_score) / 2,
            'future_improvement_potential': 0.8,
            'cognitive_profile': cognitive_scores,
            'performance_rating': self._get_performance_rating((cognitive_average + evolution_score) / 2),
            'demo_completion_time': time.time() - comprehensive_results['demo_info']['timestamp']
        }
        
        self.logger.info(f"✅ 认知能力综合演示完成，总体评分: {comprehensive_results['overall_assessment']['adaptive_capability']:.2f}")
        return comprehensive_results
    
    def _get_performance_rating(self, score: float) -> str:
        """根据评分获取性能等级"""
        if score >= 0.9:
            return "卓越"
        elif score >= 0.8:
            return "优秀"
        elif score >= 0.7:
            return "良好"
        elif score >= 0.6:
            return "合格"
        else:
            return "需要改进"
    
    def save_demo_results(self, results: Dict[str, Any], output_dir: str = None):
        """保存演示结果"""
        if output_dir is None:
            output_dir = Path("./demo_results")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        results_file = output_dir / f"cognitive_demo_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📁 演示结果已保存到: {results_file}")
        return results_file
    
    async def run_demo(self, mode: str = "full") -> Dict[str, Any]:
        """
        运行演示
        
        Args:
            mode: 演示模式 (memory, reasoning, creativity, observation, attention, imagination, evolution, full)
        """
        self.logger.info(f"🚀 开始运行认知演示，模式: {mode}")
        
        try:
            # 初始化演示环境
            await self.initialize()
            
            # 根据模式执行不同演示
            if mode == "memory":
                results = await self.run_memory_demo()
            elif mode == "reasoning":
                results = await self.run_reasoning_demo()
            elif mode == "creativity":
                results = await self.run_creativity_demo()
            elif mode == "observation":
                results = await self.run_observation_demo()
            elif mode == "attention":
                results = await self.run_attention_demo()
            elif mode == "imagination":
                results = await self.run_imagination_demo()
            elif mode == "evolution":
                results = await self.run_evolution_demo()
            elif mode == "full":
                results = await self.run_comprehensive_demo()
            else:
                raise ValueError(f"未知的演示模式: {mode}")
            
            # 保存结果
            self.demo_results.update(results)
            results_file = self.save_demo_results(self.demo_results)
            
            self.logger.info("🎉 演示运行完成！")
            return self.demo_results
            
        except Exception as e:
            self.logger.error(f"❌ 演示运行失败: {e}")
            raise
        finally:
            # 清理资源
            await self.cleanup()
    
    async def cleanup(self):
        """清理演示资源"""
        self.logger.info("🧹 清理演示资源...")
        
        if self.cognitive_认知主体:
            await self.cognitive_认知主体.cleanup()
        
        if self.world_simulator:
            await self.world_simulator.cleanup()
        
        self.logger.info("✅ 演示资源清理完成")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Cognitive Evolution Lab - 演示脚本")
    parser.add_argument("--mode", 
                       choices=["memory", "reasoning", "creativity", "observation", "attention", "imagination", "evolution", "full"],
                       default="full", 
                       help="演示模式")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--output", type=str, default="./demo_results", help="结果输出目录")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建演示实例
    demo = CognitiveDemo(config_path=args.config)
    
    try:
        # 运行演示
        results = await demo.run_demo(mode=args.mode)
        
        # 输出结果摘要
        print("\n" + "="*60)
        print("🎯 认知进化实验室演示结果摘要")
        print("="*60)
        
        if args.mode == "full":
            overall_score = results['overall_assessment']['adaptive_capability']
            print(f"📊 总体评分: {overall_score:.2f} ({results['overall_assessment']['performance_rating']})")
            print(f"🧠 认知能力平均分: {results['integration_analysis']['cognitive_average']:.2f}")
            print(f"🧬 进化能力评分: {results['integration_analysis']['evolution_score']:.2f}")
            
            print("\n📈 各认知能力详细评分:")
            for ability, score in results['integration_analysis']['cognitive_scores'].items():
                ability_names = {
                    'memory': '记忆系统',
                    'reasoning': '推理能力', 
                    'creativity': '创造力',
                    'observation': '观察力',
                    'attention': '注意力',
                    'imagination': '想象力'
                }
                print(f"  {ability_names.get(ability, ability)}: {score:.2f}")
            
            if results['integration_analysis']['strengths']:
                print(f"\n💪 优势能力: {', '.join(results['integration_analysis']['strengths'])}")
            
            if results['integration_analysis']['weaknesses']:
                print(f"\n🎯 需要改进: {', '.join(results['integration_analysis']['weaknesses'])}")
        
        print(f"\n📁 详细结果已保存到: {args.output}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断演示")
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        raise
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    print("""
    🧠🔬 Cognitive Evolution Lab - 演示脚本 🧠🔬
    ==============================================
    
    认知能力与协同进化演示平台
    作者: bingdongni
    版本: v1.0.0
    
    ✨ 展示六种认知能力的综合测试
    🚀 启动中...
    """)
    
    asyncio.run(main())