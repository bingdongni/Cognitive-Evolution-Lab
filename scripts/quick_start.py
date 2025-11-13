#!/usr/bin/env python3
"""
Cognitive Evolution Lab - 快速入门脚本
作者: bingdongni
版本: v1.0.0

这是一个简单易用的快速入门脚本，帮助新用户快速体验认知进化实验室的核心功能。
包含三个预设的演示场景：基础认知测试、创造力展示、进化演示。

使用方法:
    python scripts/quick_start.py [--scenario basic|creative|evolution|all] [--output OUTPUT_DIR]
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
from src.utils import setup_logging, load_config


class QuickStartDemo:
    """
    快速入门演示类 - 提供简单易用的认知功能体验
    """
    
    def __init__(self):
        """初始化快速入门演示"""
        # 简化的演示配置
        self.config = {
            'cognitive_models': {
                'vocab_size': 500,      # 降低词汇量以提高演示速度
                'embed_dim': 128,       # 降低嵌入维度
                'hidden_dim': 256,      # 降低隐藏层维度
                'learning_rate': 0.01   # 提高学习率以更快看到效果
            }
        }
        
        # 设置日志
        setup_logging({'global': {'debug': False, 'log_level': 'INFO'}})
        self.logger = logging.getLogger(__name__)
        
        # 初始化认知智能体
        self.cognitive_agent = None
        
        self.logger.info("🚀 快速入门演示初始化完成")
    
    async def initialize(self):
        """初始化认知智能体"""
        try:
            self.cognitive_agent = CognitiveAgent(config=self.config['cognitive_models'])
            await self.cognitive_agent.initialize()
            self.logger.info("✅ 认知智能体初始化完成")
        except Exception as e:
            self.logger.error(f"❌ 认知智能体初始化失败: {e}")
            raise
    
    async def demo_basic_cognition(self) -> Dict[str, Any]:
        """
        基础认知功能演示
        
        展示记忆、推理、注意力等基础认知能力
        """
        self.logger.info("🧠 开始基础认知功能演示...")
        
        results = {
            'scenario': '基础认知测试',
            'timestamp': time.time(),
            'memory_demo': {},
            'reasoning_demo': {},
            'attention_demo': {},
            'summary': {}
        }
        
        # 1. 简单记忆测试
        self.logger.info("📝 测试记忆功能...")
        test_memories = [
            "我今天学习了一个新的算法",
            "这个算法的名字叫Transformer",
            "它用于处理序列数据",
            "发明者是Google的研究团队"
        ]
        
        # 存储记忆
        for i, memory in enumerate(test_memories):
            await self.cognitive_agent.store_memory(
                content=memory,
                memory_type=MemoryType.EPISODIC,
                strength=0.9 - i * 0.1
            )
        
        # 检索记忆
        retrieved_memories = await self.cognitive_agent.retrieve_memory(
            query="算法",
            threshold=0.3
        )
        
        results['memory_demo'] = {
            'stored_count': len(test_memories),
            'retrieved_count': len(retrieved_memories),
            'retrieval_rate': len(retrieved_memories) / len(test_memories),
            'sample_retrieved': [mem.content for mem in retrieved_memories[:2]]
        }
        
        # 2. 简单推理测试
        self.logger.info("🧩 测试推理功能...")
        reasoning_cases = [
            {
                'premises': ["所有AI系统都需要数据", "机器学习是AI系统"],
                'expected': "机器学习需要数据"
            },
            {
                'premises': ["下雨了", "下雨会导致地面湿"],
                'expected': "地面应该是湿的"
            }
        ]
        
        reasoning_results = []
        for case in reasoning_cases:
            reasoning_chain = await self.cognitive_agent.reason(
                premises=case['premises'],
                reasoning_type=ReasoningType.DEDUCTIVE
            )
            reasoning_results.append({
                'premises': case['premises'],
                'conclusion': reasoning_chain.conclusion,
                'confidence': reasoning_chain.confidence
            })
        
        results['reasoning_demo'] = {
            'cases_tested': len(reasoning_cases),
            'results': reasoning_results,
            'average_confidence': sum(r['confidence'] for r in reasoning_results) / len(reasoning_results)
        }
        
        # 3. 注意力测试
        self.logger.info("🎯 测试注意力功能...")
        attention_tasks = ["学习新知识", "解决问题", "创意思考"]
        
        attention_results = []
        for task in attention_tasks:
            attention_weights = await self.cognitive_agent.focus_attention(task)
            attention_results.append({
                'task': task,
                'focus_level': attention_weights.get('relevance', 0.5)
            })
        
        results['attention_demo'] = {
            'tasks_tested': len(attention_tasks),
            'results': attention_results,
            'average_focus': sum(r['focus_level'] for r in attention_results) / len(attention_results)
        }
        
        # 4. 生成摘要
        memory_score = results['memory_demo']['retrieval_rate']
        reasoning_score = results['reasoning_demo']['average_confidence']
        attention_score = results['attention_demo']['average_focus']
        
        results['summary'] = {
            'cognitive_capabilities': {
                'memory': memory_score,
                'reasoning': reasoning_score,
                'attention': attention_score
            },
            'overall_score': (memory_score + reasoning_score + attention_score) / 3,
            'performance_level': self._get_performance_level((memory_score + reasoning_score + attention_score) / 3)
        }
        
        self.logger.info(f"✅ 基础认知演示完成，总体评分: {results['summary']['overall_score']:.2f}")
        return results
    
    async def demo_creativity(self) -> Dict[str, Any]:
        """
        创造力演示
        
        展示创意生成、问题解决等创造性思维
        """
        self.logger.info("🎨 开始创造力演示...")
        
        results = {
            'scenario': '创造力展示',
            'timestamp': time.time(),
            'creative_tasks': [],
            'problem_solving': [],
            'innovation_assessment': {},
            'summary': {}
        }
        
        # 创意生成任务
        creative_prompts = [
            "设计一个智能家居产品",
            "想出一种新的学习方法",
            "创造一个有趣的游戏概念",
            "提出解决交通拥堵的创新方案"
        ]
        
        for prompt in creative_prompts:
            creative_output = await self.cognitive_agent.generate_creative_output(
                context=prompt,
                style="creative"
            )
            
            results['creative_tasks'].append({
                'prompt': prompt,
                'creativity_score': creative_output['creativity_score'],
                'creative_text': creative_output['creative_text'][:100] + "..." if len(creative_output['creative_text']) > 100 else creative_output['creative_text']
            })
        
        # 问题解决任务
        problems = [
            {
                'problem': "如何在有限预算下学习编程",
                'constraints': ["低成本", "高质量", "实用"]
            },
            {
                'problem': "设计一个环保的出行方案",
                'constraints': ["零排放", "便利性", "经济性"]
            }
        ]
        
        for problem in problems:
            solution = await self.cognitive_agent.generate_creative_output(
                context=f"解决{problem['problem']}",
                style="problem_solving"
            )
            
            results['problem_solving'].append({
                'problem': problem['problem'],
                'constraints': problem['constraints'],
                'solution_quality': solution['creativity_score'],
                'solution': solution['creative_text']
            })
        
        # 创新性评估
        creative_scores = [task['creativity_score'] for task in results['creative_tasks']]
        problem_solving_scores = [ps['solution_quality'] for ps in results['problem_solving']]
        
        results['innovation_assessment'] = {
            'creative_tasks_count': len(results['creative_tasks']),
            'average_creativity': sum(creative_scores) / len(creative_scores) if creative_scores else 0,
            'problem_solving_effectiveness': sum(problem_solving_scores) / len(problem_solving_scores) if problem_solving_scores else 0,
            'innovation_index': (sum(creative_scores) + sum(problem_solving_scores)) / (len(creative_scores) + len(problem_solving_scores)) if creative_scores and problem_solving_scores else 0
        }
        
        # 生成摘要
        results['summary'] = {
            'creativity_metrics': {
                'creative_generation': results['innovation_assessment']['average_creativity'],
                'problem_solving': results['innovation_assessment']['problem_solving_effectiveness'],
                'innovation_index': results['innovation_assessment']['innovation_index']
            },
            'overall_score': results['innovation_assessment']['innovation_index'],
            'creativity_level': self._get_creativity_level(results['innovation_assessment']['innovation_index'])
        }
        
        self.logger.info(f"✅ 创造力演示完成，创新指数: {results['summary']['overall_score']:.2f}")
        return results
    
    async def demo_evolution(self) -> Dict[str, Any]:
        """
        进化演示
        
        简化的协同进化模拟
        """
        self.logger.info("🧬 开始进化演示...")
        
        results = {
            'scenario': '协同进化演示',
            'timestamp': time.time(),
            'evolution_simulation': {},
            'learning_progress': {},
            'adaptation_metrics': {},
            'summary': {}
        }
        
        # 模拟简单的进化过程
        generations = 5
        population_size = 3
        fitness_history = []
        
        for generation in range(generations):
            # 模拟适应度评估
            generation_fitness = []
            for individual in range(population_size):
                # 简化的适应度函数
                base_fitness = 0.5
                improvement = generation * 0.1
                noise = 0.05 * (individual - 1)  # 个体差异
                fitness = base_fitness + improvement + noise
                fitness = max(0.1, min(1.0, fitness))  # 限制在[0.1, 1.0]
                generation_fitness.append(fitness)
            
            avg_fitness = sum(generation_fitness) / len(generation_fitness)
            fitness_history.append(avg_fitness)
            
            # 模拟适应性学习
            learning_rate = max(0.001, 0.1 / (generation + 1))  # 学习率递减
            adaptation_rate = min(0.9, 0.3 + generation * 0.1)  # 适应性递增
        
        results['evolution_simulation'] = {
            'generations': generations,
            'population_size': population_size,
            'fitness_evolution': fitness_history,
            'improvement_rate': fitness_history[-1] - fitness_history[0] if len(fitness_history) > 1 else 0,
            'convergence_status': "converged" if len(fitness_history) > 2 and abs(fitness_history[-1] - fitness_history[-2]) < 0.01 else "evolving"
        }
        
        # 学习进度分析
        memory_retention = 0.8
        transfer_learning = 0.6
        knowledge_retention = 0.75
        
        results['learning_progress'] = {
            'memory_retention': memory_retention,
            'transfer_learning': transfer_learning,
            'knowledge_retention': knowledge_retention,
            'learning_efficiency': (memory_retention + transfer_learning + knowledge_retention) / 3
        }
        
        # 适应性指标
        environmental_change = [0.3, 0.5, 0.7, 0.4, 0.6]  # 环境变化强度
        adaptation_responses = []
        
        for change_strength in environmental_change:
            # 适应性响应 = 环境变化 * 响应能力
            response_ability = 0.5 + (fitness_history[-1] * 0.5)  # 基于当前适应度
            adaptation_score = min(1.0, change_strength * response_ability)
            adaptation_responses.append(adaptation_score)
        
        results['adaptation_metrics'] = {
            'environmental_changes': environmental_change,
            'adaptation_responses': adaptation_responses,
            'adaptability_score': sum(adaptation_responses) / len(adaptation_responses),
            'responsiveness': max(adaptation_responses) - min(adaptation_responses)
        }
        
        # 生成摘要
        evolution_score = results['evolution_simulation']['improvement_rate']
        learning_score = results['learning_progress']['learning_efficiency']
        adaptation_score = results['adaptation_metrics']['adaptability_score']
        
        results['summary'] = {
            'evolution_metrics': {
                'evolution_progress': evolution_score,
                'learning_efficiency': learning_score,
                'adaptability': adaptation_score
            },
            'overall_score': (evolution_score + learning_score + adaptation_score) / 3,
            'evolution_stage': "mature" if evolution_score > 0.2 else "developing"
        }
        
        self.logger.info(f"✅ 进化演示完成，总体评分: {results['summary']['overall_score']:.2f}")
        return results
    
    def _get_performance_level(self, score: float) -> str:
        """根据评分获取性能等级"""
        if score >= 0.8:
            return "优秀"
        elif score >= 0.6:
            return "良好"
        elif score >= 0.4:
            return "合格"
        else:
            return "需要改进"
    
    def _get_creativity_level(self, score: float) -> str:
        """根据评分获取创造力等级"""
        if score >= 0.8:
            return "高度创新"
        elif score >= 0.6:
            return "中等创新"
        elif score >= 0.4:
            return "一般创新"
        else:
            return "需要提升"
    
    def save_results(self, results: Dict[str, Any], output_dir: str) -> Path:
        """保存演示结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        results_file = output_path / f"quick_start_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📁 结果已保存到: {results_file}")
        return results_file
    
    async def run_quick_start(self, scenario: str = "all") -> Dict[str, Any]:
        """
        运行快速入门演示
        
        Args:
            scenario: 演示场景 (basic, creative, evolution, all)
        """
        self.logger.info(f"🚀 开始快速入门演示，场景: {scenario}")
        
        try:
            # 初始化
            await self.initialize()
            
            all_results = {
                'quick_start_info': {
                    'title': '认知进化实验室 - 快速入门演示',
                    'version': '1.0.0',
                    'timestamp': time.time(),
                    'scenario': scenario
                },
                'demos': {},
                'overall_summary': {}
            }
            
            # 根据场景执行演示
            start_time = time.time()
            
            if scenario in ["basic", "all"]:
                basic_results = await self.demo_basic_cognition()
                all_results['demos']['basic_cognition'] = basic_results
            
            if scenario in ["creative", "all"]:
                creative_results = await self.demo_creativity()
                all_results['demos']['creativity'] = creative_results
            
            if scenario in ["evolution", "all"]:
                evolution_results = await self.demo_evolution()
                all_results['demos']['evolution'] = evolution_results
            
            # 生成总体摘要
            demo_scores = []
            if 'basic_cognition' in all_results['demos']:
                demo_scores.append(all_results['demos']['basic_cognition']['summary']['overall_score'])
            if 'creativity' in all_results['demos']:
                demo_scores.append(all_results['demos']['creativity']['summary']['overall_score'])
            if 'evolution' in all_results['demos']:
                demo_scores.append(all_results['demos']['evolution']['summary']['overall_score'])
            
            overall_score = sum(demo_scores) / len(demo_scores) if demo_scores else 0
            
            all_results['overall_summary'] = {
                'demos_completed': len(all_results['demos']),
                'overall_score': overall_score,
                'performance_level': self._get_performance_level(overall_score),
                'demo_duration': time.time() - start_time,
                'recommendations': self._generate_recommendations(all_results['demos'])
            }
            
            self.logger.info(f"✅ 快速入门演示完成，总体评分: {overall_score:.2f}")
            return all_results
            
        except Exception as e:
            self.logger.error(f"❌ 快速入门演示失败: {e}")
            raise
        finally:
            await self.cleanup()
    
    def _generate_recommendations(self, demos: Dict[str, Any]) -> List[str]:
        """根据演示结果生成建议"""
        recommendations = []
        
        if 'basic_cognition' in demos:
            memory_score = demos['basic_cognition']['summary']['cognitive_capabilities']['memory']
            reasoning_score = demos['basic_cognition']['summary']['cognitive_capabilities']['reasoning']
            
            if memory_score < 0.6:
                recommendations.append("💡 建议增加记忆训练，如使用记忆宫殿法")
            if reasoning_score < 0.6:
                recommendations.append("🧩 建议多练习逻辑推理，如解决数学谜题")
        
        if 'creativity' in demos:
            creativity_score = demos['creativity']['summary']['overall_score']
            if creativity_score < 0.6:
                recommendations.append("🎨 建议进行创意训练，如头脑风暴、联想练习")
        
        if 'evolution' in demos:
            evolution_score = demos['evolution']['summary']['overall_score']
            if evolution_score < 0.6:
                recommendations.append("🧬 建议加强学习策略，如制定学习计划、反思总结")
        
        if not recommendations:
            recommendations = ["🌟 表现优秀！继续保持当前的学习和思考方式"]
        
        return recommendations
    
    async def cleanup(self):
        """清理资源"""
        if self.cognitive_agent:
            await self.cognitive_agent.cleanup()


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Cognitive Evolution Lab - 快速入门")
    parser.add_argument("--scenario", 
                       choices=["basic", "creative", "evolution", "all"],
                       default="all", 
                       help="演示场景")
    parser.add_argument("--output", type=str, default="./quick_start_results", help="结果输出目录")
    parser.add_argument("--quiet", action="store_true", help="安静模式（减少输出）")
    
    args = parser.parse_args()
    
    # 创建演示实例
    demo = QuickStartDemo()
    
    try:
        # 运行快速入门演示
        results = await demo.run_quick_start(scenario=args.scenario)
        
        # 输出结果
        if not args.quiet:
            print("\n" + "="*50)
            print("🎯 认知进化实验室 - 快速入门结果")
            print("="*50)
            
            print(f"📊 演示场景: {args.scenario}")
            print(f"⏱️  演示时长: {results['overall_summary']['demo_duration']:.1f}秒")
            print(f"🏆 总体评分: {results['overall_summary']['overall_score']:.2f}")
            print(f"⭐ 性能等级: {results['overall_summary']['performance_level']}")
            
            # 显示各模块得分
            if 'basic_cognition' in results['demos']:
                scores = results['demos']['basic_cognition']['summary']['cognitive_capabilities']
                print(f"\n🧠 基础认知能力:")
                print(f"   记忆系统: {scores['memory']:.2f}")
                print(f"   推理能力: {scores['reasoning']:.2f}")
                print(f"   注意力:   {scores['attention']:.2f}")
            
            if 'creativity' in results['demos']:
                creativity_score = results['demos']['creativity']['summary']['overall_score']
                print(f"\n🎨 创造力:")
                print(f"   创新指数: {creativity_score:.2f}")
            
            if 'evolution' in results['demos']:
                evolution_score = results['demos']['evolution']['summary']['overall_score']
                print(f"\n🧬 进化能力:")
                print(f"   进化进度: {evolution_score:.2f}")
            
            # 显示建议
            print(f"\n💡 个性化建议:")
            for i, rec in enumerate(results['overall_summary']['recommendations'], 1):
                print(f"   {i}. {rec}")
            
            print(f"\n📁 详细结果已保存到: {args.output}")
            print("="*50)
        
        # 保存结果
        results_file = demo.save_results(results, args.output)
        
        return results
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断快速入门演示")
    except Exception as e:
        print(f"❌ 快速入门演示失败: {e}")
        raise
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    print("""
    🚀 Cognitive Evolution Lab - 快速入门 🚀
    ======================================
    
    简单易用的认知功能体验平台
    作者: bingdongni
    版本: v1.0.0
    
    ⚡ 快速体验认知进化实验室的核心功能
    🎯 启动中...
    """)
    
    asyncio.run(main())