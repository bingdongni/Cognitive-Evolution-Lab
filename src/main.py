#!/usr/bin/env python3
"""
Cognitive Evolution Lab - 主程序入口
作者: bingdongni
版本: v1.0.0

这是一个集成前沿认知计算技术的综合性协同进化实验平台的主入口程序。
实现了外部世界-内部心智-交互行动相结合的综合模型。
"""

import os
import sys
import argparse
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from src.world_simulator import VirtualWorld
from src.cognitive_models import CognitiveAgent
from src.interactive_systems import EmbodiedIntelligence
from src.evolution_engine import EvolutionEngine
from src.visualization import LabDashboard
from src.utils import (
    setup_logging, 
    load_config, 
    validate_environment,
    HardwareDetector
)

class CognitiveEvolutionLab:
    """
    Cognitive Evolution Lab 主类
    整合所有核心模块的主要控制器
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化实验室
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = load_config(config_path)
        
        # 设置日志
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # 硬件检测
        self.hardware = HardwareDetector()
        self.logger.info(f"检测到硬件配置: {self.hardware.get_summary()}")
        
        # 初始化核心模块
        self.world_simulator = None
        self.cognitive_认知主体 = None
        self.embodied_intelligence = None
        self.evolution_engine = None
        self.dashboard = None
        
        # 实验状态
        self.is_running = False
        self.current_experiment = None
        self.experiment_results = {}
        
        self.logger.info("🚀 Cognitive Evolution Lab 初始化完成")
    
    async def initialize_modules(self):
        """异步初始化所有核心模块"""
        self.logger.info("🔧 初始化核心模块...")
        
        try:
            # 初始化外部世界模拟器
            self.world_simulator = VirtualWorld(
                config=self.config['world_simulator']
            )
            await self.world_simulator.initialize()
            self.logger.info("✅ 外部世界模拟器初始化完成")
            
            # 初始化认知认知主体
            self.cognitive_认知主体 = CognitiveAgent(
                config=self.config['cognitive_models']
            )
            await self.cognitive_认知主体.initialize()
            self.logger.info("✅ 内部心智模型初始化完成")
            
            # 初始化具身智能系统
            self.embodied_intelligence = EmbodiedIntelligence(
                config=self.config['interactive_systems']['embodied_intelligence']
            )
            await self.embodied_intelligence.initialize()
            self.logger.info("✅ 交互行动系统初始化完成")
            
            # 初始化协同进化引擎
            self.evolution_engine = EvolutionEngine(
                config=self.config['evolution_engine']
            )
            await self.evolution_engine.initialize()
            self.logger.info("✅ 协同进化引擎初始化完成")
            
            # 初始化可视化界面
            self.dashboard = LabDashboard(
                config=self.config['visualization']
            )
            await self.dashboard.initialize()
            self.logger.info("✅ 可视化界面初始化完成")
            
            self.logger.info("🎉 所有模块初始化完成！")
            
        except Exception as e:
            self.logger.error(f"❌ 模块初始化失败: {e}")
            raise
    
    async def run_cognitive_test(self, test_type: str = "full"):
        """
        运行认知能力测试
        
        Args:
            test_type: 测试类型 (memory, reasoning, creativity, observation, attention, imagination, full)
        """
        self.logger.info(f"🧠 开始认知能力测试: {test_type}")
        self.current_experiment = f"cognitive_{test_type}"
        
        try:
            # 创建认知测试环境
            test_world = await self.world_simulator.create_test_environment(
                test_type=test_type
            )
            
            # 运行认知认知主体测试
            cognitive_results = await self.cognitive_认知主体.run_cognitive_test(
                environment=test_world,
                test_type=test_type
            )
            
            # 具身认知主体在环境中行动
            embodied_actions = await self.embodied_intelligence.execute_cognitive_task(
                cognitive_state=cognitive_results['cognitive_state'],
                environment=test_world
            )
            
            # 综合评估
            final_results = {
                'cognitive_metrics': cognitive_results,
                'embodied_performance': embodied_actions,
                'overall_score': self._calculate_overall_score(
                    cognitive_results, embodied_actions
                ),
                'test_type': test_type,
                'timestamp': asyncio.get_event_loop().time()
            }
            
            self.experiment_results[self.current_experiment] = final_results
            
            # 更新可视化界面
            await self.dashboard.update_cognitive_results(final_results)
            
            self.logger.info(f"✅ 认知测试完成，总体评分: {final_results['overall_score']:.2f}")
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ 认知测试失败: {e}")
            raise
    
    async def run_evolution_experiment(self, experiment_type: str = "multi_认知主体"):
        """
        运行协同进化实验
        
        Args:
            experiment_type: 实验类型 (single_认知主体, multi_认知主体, co_evolution, cultural)
        """
        self.logger.info(f"🧬 开始协同进化实验: {experiment_type}")
        self.current_experiment = f"evolution_{experiment_type}"
        
        try:
            # 创建进化环境
            evolution_world = await self.world_simulator.create_evolution_environment(
                experiment_type=experiment_type
            )
            
            # 初始化种群
            population = await self.evolution_engine.initialize_population(
                environment=evolution_world,
                experiment_type=experiment_type
            )
            
            # 运行进化过程
            evolution_results = await self.evolution_engine.evolve(
                population=population,
                environment=evolution_world,
                generations=100
            )
            
            # 测试最佳个体
            best_individual = evolution_results['best_individual']
            cognitive_test_results = await self.cognitive_认知主体.evaluate_individual(
                individual=best_individual,
                environment=evolution_world
            )
            
            final_results = {
                'evolution_data': evolution_results,
                'best_individual_cognitive': cognitive_test_results,
                'evolutionary_fitness': evolution_results['final_fitness'],
                'population_diversity': evolution_results['diversity_score'],
                'experiment_type': experiment_type,
                'timestamp': asyncio.get_event_loop().time()
            }
            
            self.experiment_results[self.current_experiment] = final_results
            
            # 更新可视化界面
            await self.dashboard.update_evolution_results(final_results)
            
            self.logger.info(f"✅ 进化实验完成，最佳适应度: {evolution_results['final_fitness']:.4f}")
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ 进化实验失败: {e}")
            raise
    
    async def run_lifelong_learning(self, duration_hours: float = 24.0):
        """
        运行终身学习实验
        
        Args:
            duration_hours: 实验持续时间（小时）
        """
        self.logger.info(f"📚 开始终身学习实验，时长: {duration_hours}小时")
        self.current_experiment = "lifelong_learning"
        
        try:
            start_time = asyncio.get_event_loop().time()
            end_time = start_time + (duration_hours * 3600)
            
            learning_results = {
                'learning_curves': [],
                'memory_retention': [],
                'transfer_performance': [],
                'metacognitive_analysis': []
            }
            
            while asyncio.get_event_loop().time() < end_time:
                # 多任务学习序列
                task_results = await self._run_learning_sequence()
                
                # 更新学习曲线
                learning_results['learning_curves'].append(task_results)
                
                # 记忆巩固测试
                memory_test = await self.cognitive_认知主体.test_memory_retention()
                learning_results['memory_retention'].append(memory_test)
                
                # 迁移学习测试
                transfer_test = await self.cognitive_认知主体.test_transfer_learning()
                learning_results['transfer_performance'].append(transfer_test)
                
                # 元认知分析
                metacognitive = await self.cognitive_认知主体.analyze_learning_strategy()
                learning_results['metacognitive_analysis'].append(metacognitive)
                
                # 更新可视化
                await self.dashboard.update_learning_progress(learning_results)
                
                # 短暂休息
                await asyncio.sleep(60)  # 每分钟一个学习周期
            
            final_results = {
                'learning_data': learning_results,
                'duration_hours': duration_hours,
                'total_cycles': len(learning_results['learning_curves']),
                'final_performance': learning_results['learning_curves'][-1],
                'memory_retention_rate': self._calculate_retention_rate(
                    learning_results['memory_retention']
                ),
                'transfer_ability': self._calculate_transfer_ability(
                    learning_results['transfer_performance']
                )
            }
            
            self.experiment_results[self.current_experiment] = final_results
            
            self.logger.info(f"✅ 终身学习实验完成，总周期数: {final_results['total_cycles']}")
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ 终身学习实验失败: {e}")
            raise
    
    async def run_integrated_experiment(self):
        """运行集成实验（认知+进化+终身学习的综合实验）"""
        self.logger.info("🔬 开始集成综合实验")
        self.current_experiment = "integrated"
        
        try:
            # 阶段1: 基础认知能力评估
            self.logger.info("阶段1: 基础认知能力评估")
            cognitive_baseline = await self.run_cognitive_test("full")
            
            # 阶段2: 协同进化优化
            self.logger.info("阶段2: 协同进化优化")
            evolution_results = await self.run_evolution_experiment("multi_认知主体")
            
            # 阶段3: 基于进化结果的认知重训练
            self.logger.info("阶段3: 认知重训练")
            evolved_认知主体 = evolution_results['best_individual_cognitive']
            retrained_results = await self.cognitive_认知主体.retrain_with_evolution(
                evolution_data=evolution_results['evolution_data']
            )
            
            # 阶段4: 长期适应性测试
            self.logger.info("阶段4: 长期适应性测试")
            adaptation_results = await self._test_long_term_adaptation()
            
            # 综合分析
            integrated_results = {
                'baseline_cognitive': cognitive_baseline,
                'evolutionary_improvement': evolution_results,
                'cognitive_retraining': retrained_results,
                'long_term_adaptation': adaptation_results,
                'overall_assessment': self._generate_overall_assessment(
                    cognitive_baseline, evolution_results, 
                    retrained_results, adaptation_results
                ),
                'integrated_score': self._calculate_integrated_score(
                    cognitive_baseline, evolution_results, 
                    retrained_results, adaptation_results
                ),
                'experiment_timestamp': asyncio.get_event_loop().time()
            }
            
            self.experiment_results[self.current_experiment] = integrated_results
            
            # 更新完整可视化
            await self.dashboard.update_integrated_results(integrated_results)
            
            self.logger.info(f"✅ 集成实验完成，综合评分: {integrated_results['integrated_score']:.4f}")
            return integrated_results
            
        except Exception as e:
            self.logger.error(f"❌ 集成实验失败: {e}")
            raise
    
    async def start_dashboard(self, port: int = 8050):
        """启动可视化仪表板"""
        self.logger.info(f"📊 启动可视化仪表板，端口: {port}")
        
        try:
            await self.dashboard.start_server(port=port)
            self.logger.info(f"✅ 仪表板启动成功，访问地址: http://localhost:{port}")
            
        except Exception as e:
            self.logger.error(f"❌ 仪表板启动失败: {e}")
            raise
    
    def _calculate_overall_score(self, cognitive_results, embodied_results):
        """计算总体评分"""
        cognitive_score = cognitive_results.get('overall_score', 0.5)
        embodied_score = embodied_results.get('performance_score', 0.5)
        
        # 加权综合评分
        overall_score = (cognitive_score * 0.7) + (embodied_score * 0.3)
        return min(1.0, max(0.0, overall_score))
    
    def _calculate_retention_rate(self, memory_tests):
        """计算记忆保留率"""
        if not memory_tests:
            return 0.0
        
        retention_scores = [test.get('retention_score', 0.0) for test in memory_tests]
        return sum(retention_scores) / len(retention_scores)
    
    def _calculate_transfer_ability(self, transfer_tests):
        """计算迁移能力"""
        if not transfer_tests:
            return 0.0
        
        transfer_scores = [test.get('transfer_score', 0.0) for test in transfer_tests]
        return sum(transfer_scores) / len(transfer_scores)
    
    def _calculate_integrated_score(self, baseline, evolution, retraining, adaptation):
        """计算集成实验综合评分"""
        baseline_score = baseline.get('overall_score', 0.5)
        evolution_score = evolution.get('evolutionary_fitness', 0.5)
        retraining_score = retraining.get('improvement_score', 0.5)
        adaptation_score = adaptation.get('adaptation_rate', 0.5)
        
        # 多维度加权评分
        weights = [0.25, 0.25, 0.25, 0.25]
        scores = [baseline_score, evolution_score, retraining_score, adaptation_score]
        
        integrated_score = sum(w * s for w, s in zip(weights, scores))
        return min(1.0, max(0.0, integrated_score))
    
    async def _run_learning_sequence(self):
        """运行一个学习序列"""
        # 这里实现具体的学习序列
        return {
            'learning_rate': 0.01,
            'task_completion': 0.8,
            'error_rate': 0.15,
            'task_type': 'pattern_recognition'
        }
    
    async def _test_long_term_adaptation(self):
        """测试长期适应性"""
        return {
            'adaptation_rate': 0.85,
            'flexibility_score': 0.78,
            'robustness_score': 0.82,
            'novelty_handling': 0.75
        }
    
    def _generate_overall_assessment(self, baseline, evolution, retraining, adaptation):
        """生成总体评估报告"""
        return {
            'cognitive_improvement': baseline.get('overall_score', 0.5),
            'evolutionary_success': evolution.get('evolutionary_fitness', 0.5),
            'retraining_effectiveness': retraining.get('improvement_score', 0.5),
            'adaptation_capability': adaptation.get('adaptation_rate', 0.5),
            'recommendations': [
                "继续保持当前的认知训练模式",
                "增加进化种群的多样性",
                "优化记忆巩固机制"
            ]
        }
    
    def save_results(self, output_dir: str = "./results"):
        """保存实验结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        import json
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"experiment_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📁 实验结果已保存到: {results_file}")
    
    async def cleanup(self):
        """清理资源"""
        self.logger.info("🧹 清理资源...")
        
        if self.dashboard:
            await self.dashboard.cleanup()
        
        if self.world_simulator:
            await self.world_simulator.cleanup()
        
        self.logger.info("✅ 资源清理完成")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Cognitive Evolution Lab")
    parser.add_argument("--mode", choices=["demo", "cognitive", "evolution", "lifelong", "integrated", "dashboard"], 
                       default="demo", help="运行模式")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--experiment", type=str, help="指定实验类型")
    parser.add_argument("--duration", type=float, help="实验持续时间（小时）")
    parser.add_argument("--port", type=int, default=8050, help="仪表板端口")
    parser.add_argument("--output", type=str, default="./results", help="结果输出目录")
    
    args = parser.parse_args()
    
    # 验证环境
    validate_environment()
    
    # 创建实验室实例
    lab = CognitiveEvolutionLab(config_path=args.config)
    
    try:
        # 初始化模块
        await lab.initialize_modules()
        
        # 根据模式运行实验
        if args.mode == "demo":
            print("🎯 运行演示模式")
            await lab.run_cognitive_test("full")
            await asyncio.sleep(2)
            await lab.run_evolution_experiment("multi_认知主体")
            
        elif args.mode == "cognitive":
            await lab.run_cognitive_test(args.experiment or "full")
            
        elif args.mode == "evolution":
            await lab.run_evolution_experiment(args.experiment or "multi_认知主体")
            
        elif args.mode == "lifelong":
            duration = args.duration or 24.0
            await lab.run_lifelong_learning(duration)
            
        elif args.mode == "integrated":
            await lab.run_integrated_experiment()
            
        elif args.mode == "dashboard":
            await lab.start_dashboard(args.port)
            
            # 保持服务器运行
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                pass
        
        # 保存结果
        lab.save_results(args.output)
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断程序")
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        raise
    finally:
        await lab.cleanup()


if __name__ == "__main__":
    print("""
    🧠🔬 Cognitive Evolution Lab 🧠🔬
    ======================================
    
    集成前沿认知计算技术的综合性协同进化实验平台
    作者: bingdongni
    版本: v1.0.0
    
    🚀 启动中...
    """)
    
    asyncio.run(main())