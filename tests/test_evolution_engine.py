"""
进化引擎测试模块

测试协同进化引擎的功能，包括单主体进化和多主体进化。
"""

import unittest
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evolution_engine import (
    EvolutionEngine,
    SingleAgentEvolution,
    MultiAgentEvolution,
    KnowledgeEvolution,
    EnvironmentCoevolution,
    PopulationManager
)


class TestEvolutionEngine(unittest.TestCase):
    """进化引擎基础测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.evolution_engine = EvolutionEngine()
        
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.evolution_engine, EvolutionEngine)
        self.assertIsNotNone(self.evolution_engine.single_evolution)
        self.assertIsNotNone(self.evolution_engine.multi_evolution)
        self.assertIsNotNone(self.evolution_engine.knowledge_evolution)
        
    def test_evolution_cycle(self):
        """测试进化周期"""
        # 设置初始种群
        initial_population = [
            {"genome": np.random.rand(10), "fitness": 0.5},
            {"genome": np.random.rand(10), "fitness": 0.6},
            {"genome": np.random.rand(10), "fitness": 0.4}
        ]
        
        # 执行一个进化周期
        result = self.evolution_engine.evolve_population(initial_population)
        self.assertIsInstance(result, dict)
        self.assertIn("new_population", result)
        self.assertIn("generation_stats", result)
        self.assertIn("evolution_progress", result)
        
    def test_adaptation_strategy_selection(self):
        """测试适应策略选择"""
        # 模拟不同的环境条件
        environment_conditions = {
            "complexity": 0.8,
            "stability": 0.3,
            "resource_availability": 0.6
        }
        
        strategy = self.evolution_engine.select_adaptation_strategy(environment_conditions)
        self.assertIsInstance(strategy, dict)
        self.assertIn("strategy_type", strategy)
        self.assertIn("parameters", strategy)
        self.assertIn("expected_outcome", strategy)


class TestSingleAgentEvolution(unittest.TestCase):
    """单主体进化测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.single_evolution = SingleAgentEvolution()
        
    def test_lifetime_learning(self):
        """测试终身学习"""
        # 模拟学习序列
        learning_sequence = [
            {"experience": "experience_1", "importance": 0.8},
            {"experience": "experience_2", "importance": 0.6},
            {"experience": "experience_3", "importance": 0.9}
        ]
        
        # 执行终身学习
        learning_result = self.single_evolution.lifetime_learning(learning_sequence)
        self.assertIsInstance(learning_result, dict)
        self.assertIn("knowledge_base", learning_result)
        self.assertIn("learning_progress", learning_result)
        self.assertIn("adaptation_metrics", learning_result)
        
    def test_adaptive_cognition(self):
        """测试适应性认知"""
        # 设置认知能力参数
        cognitive_capabilities = {
            "memory": 0.5,
            "reasoning": 0.6,
            "creativity": 0.4,
            "learning_rate": 0.3
        }
        
        # 执行认知适应
        adaptation_result = self.single_evolution.adapt_cognition(cognitive_capabilities)
        self.assertIsInstance(adaptation_result, dict)
        self.assertIn("updated_capabilities", adaptation_result)
        self.assertIn("adaptation_success", adaptation_result)
        self.assertIn("cognitive_evolution", adaptation_result)
        
    def test_performance_optimization(self):
        """测试性能优化"""
        # 模拟性能指标
        performance_metrics = {
            "task_accuracy": 0.75,
            "processing_speed": 0.6,
            "resource_efficiency": 0.8,
            "adaptability": 0.5
        }
        
        optimization_result = self.single_evolution.optimize_performance(performance_metrics)
        self.assertIsInstance(optimization_result, dict)
        self.assertIn("optimization_strategy", optimization_result)
        self.assertIn("expected_improvement", optimization_result)
        self.assertIn("resource_requirements", optimization_result)
        
    def test_genetic_operators(self):
        """测试遗传操作"""
        parent_genome = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # 测试变异
        mutated = self.single_evolution.mutate_genome(parent_genome)
        self.assertEqual(len(mutated), len(parent_genome))
        
        # 测试交叉
        parent2_genome = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
        offspring = self.single_evolution.crossover_genomes(parent_genome, parent2_genome)
        self.assertEqual(len(offspring), len(parent_genome))
        
        # 测试选择
        population = [parent_genome, parent2_genome, mutated]
        fitness_scores = [0.8, 0.7, 0.6]
        selected = self.single_evolution.select_individuals(population, fitness_scores)
        self.assertIsInstance(selected, list)


class TestMultiAgentEvolution(unittest.TestCase):
    """多主体进化测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.multi_evolution = MultiAgentEvolution()
        
    def test_population_dynamics(self):
        """测试种群动态"""
        # 初始化种群
        initial_population = [
            {
                "id": f"agent_{i}",
                "genome": np.random.rand(10),
                "traits": {
                    "cooperation": np.random.rand(),
                    "competition": np.random.rand(),
                    "communication": np.random.rand()
                },
                "fitness": np.random.rand()
            }
            for i in range(10)
        ]
        
        # 执行种群动态更新
        dynamics_result = self.multi_evolution.update_population_dynamics(initial_population)
        self.assertIsInstance(dynamics_result, dict)
        self.assertIn("population_changes", dynamics_result)
        self.assertIn("interaction_matrix", dynamics_result)
        self.assertIn("emergent_behaviors", dynamics_result)
        
    def test_cooperative_evolution(self):
        """测试合作进化"""
        # 模拟合作群体
        cooperative_group = [
            {"id": "agent_1", "cooperation_score": 0.8, "fitness": 0.6},
            {"id": "agent_2", "cooperation_score": 0.9, "fitness": 0.7},
            {"id": "agent_3", "cooperation_score": 0.7, "fitness": 0.5}
        ]
        
        cooperation_result = self.multi_evolution.evolve_cooperation(cooperative_group)
        self.assertIsInstance(cooperation_result, dict)
        self.assertIn("cooperation_strategies", cooperation_result)
        self.assertIn("group_performance", cooperation_result)
        self.assertIn("social_structure", cooperation_result)
        
    def test_competitive_interactions(self):
        """测试竞争互动"""
        # 模拟竞争场景
        competitors = [
            {"id": "agent_1", "resource_claim": 0.6, "fitness": 0.7},
            {"id": "agent_2", "resource_claim": 0.8, "fitness": 0.5},
            {"id": "agent_3", "resource_claim": 0.4, "fitness": 0.8}
        ]
        
        competition_result = self.multi_evolution.process_competition(competitors)
        self.assertIsInstance(competition_result, dict)
        self.assertIn("resource_allocation", competition_result)
        self.assertIn("fitness_updates", competition_result)
        self.assertIn("competition_outcomes", competition_result)
        
    def test_swarm_intelligence(self):
        """测试群体智能"""
        # 模拟群体行为
        swarm_state = {
            "agents": [f"agent_{i}" for i in range(20)],
            "collective_goal": "optimize_objective_function",
            "interaction_radius": 0.5
        }
        
        swarm_result = self.multi_evolution.evolve_swarm_intelligence(swarm_state)
        self.assertIsInstance(swarm_result, dict)
        self.assertIn("collective_behavior", swarm_result)
        self.assertIn("emergent_solution", swarm_result)
        self.assertIn("swarm_optimization", swarm_result)


class TestKnowledgeEvolution(unittest.TestCase):
    """知识进化测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.knowledge_evolution = KnowledgeEvolution()
        
    def test_knowledge_acquisition(self):
        """测试知识获取"""
        # 模拟学习数据
        learning_data = [
            {
                "source": "experience",
                "content": "object_recognition_pattern_1",
                "confidence": 0.8,
                "timestamp": 1234567890
            },
            {
                "source": "social_learning",
                "content": "collaboration_strategy",
                "confidence": 0.9,
                "timestamp": 1234567891
            }
        ]
        
        acquisition_result = self.knowledge_evolution.acquire_knowledge(learning_data)
        self.assertIsInstance(acquisition_result, dict)
        self.assertIn("knowledge_updates", acquisition_result)
        self.assertIn("learning_progress", acquisition_result)
        self.assertIn("knowledge_base_changes", acquisition_result)
        
    def test_knowledge_integration(self):
        """测试知识整合"""
        # 模拟不同来源的知识
        knowledge_sources = {
            "experiential": {"facts": ["A causes B"], "confidence": 0.8},
            "social": {"facts": ["B is preferred"], "confidence": 0.7},
            "abstract": {"rules": ["If A then B"], "confidence": 0.9}
        }
        
        integration_result = self.knowledge_evolution.integrate_knowledge(knowledge_sources)
        self.assertIsInstance(integration_result, dict)
        self.assertIn("unified_knowledge", integration_result)
        self.assertIn("conflict_resolution", integration_result)
        self.assertIn("knowledge_consistency", integration_result)
        
    def test_knowledge_forgetting(self):
        """测试知识遗忘"""
        # 模拟知识库
        knowledge_base = {
            "old_fact_1": {"importance": 0.1, "last_accessed": 1234567800},
            "old_fact_2": {"importance": 0.2, "last_accessed": 1234567700},
            "recent_fact": {"importance": 0.9, "last_accessed": 1234567890}
        }
        
        forgetting_result = self.knowledge_evolution.manage_knowledge_forgetting(knowledge_base)
        self.assertIsInstance(forgetting_result, dict)
        self.assertIn("forgotten_items", forgetting_result)
        self.assertIn("retention_recommendations", forgetting_result)
        self.assertIn("memory_optimization", forgetting_result)
        
    def test_knowledge_transfer(self):
        """测试知识转移"""
        # 模拟源知识库和目标智能体
        source_knowledge = {
            "problem_solving": {"strategy": "divide_and_conquer", "effectiveness": 0.8},
            "pattern_recognition": {"method": "feature_based", "accuracy": 0.85}
        }
        
        target_agent = {
            "current_capabilities": {"learning_rate": 0.7},
            "knowledge_gaps": ["pattern_recognition"]
        }
        
        transfer_result = self.knowledge_evolution.transfer_knowledge(source_knowledge, target_agent)
        self.assertIsInstance(transfer_result, dict)
        self.assertIn("transfer_efficiency", transfer_result)
        self.assertIn("knowledge_uptake", transfer_result)
        self.assertIn("adaptation_requirements", transfer_result)


class TestEnvironmentCoevolution(unittest.TestCase):
    """环境共进化测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.coevolution = EnvironmentCoevolution()
        
    def test_environment_adaptation(self):
        """测试环境适应"""
        # 模拟环境状态
        current_environment = {
            "resources": {"abundance": 0.6, "distribution": "clustered"},
            "challenges": {"complexity": 0.7, "novelty": 0.5},
            "agent_impact": {"resource_modification": 0.3}
        }
        
        adaptation_result = self.coevolution.adapt_environment(current_environment)
        self.assertIsInstance(adaptation_result, dict)
        self.assertIn("environmental_changes", adaptation_result)
        self.assertIn("adaptation_rationale", adaptation_result)
        self.assertIn("expected_agent_impact", adaptation_result)
        
    def test_niche_formation(self):
        """测试生态位形成"""
        # 模拟生态位创建过程
        niche_conditions = {
            "resource_availability": 0.8,
            "competition_level": 0.4,
            "environmental_stability": 0.7
        }
        
        niche_result = self.coevolution.form_ecological_niche(niche_conditions)
        self.assertIsInstance(niche_result, dict)
        self.assertIn("niche_characteristics", niche_result)
        self.assertIn("niche_capacity", niche_result)
        self.assertIn("niche_evolution_potential", niche_result)
        
    def test_coevolutionary_dynamics(self):
        """测试共进化动态"""
        # 模拟智能体-环境互动
        agent_environment_interaction = {
            "agent_actions": ["resource_consumption", "habitat_modification"],
            "environmental_responses": ["resource_regeneration", "habitat_adaptation"],
            "feedback_loops": ["positive_feedback", "negative_feedback"]
        }
        
        dynamics_result = self.coevolution.simulate_coevolution_dynamics(agent_environment_interaction)
        self.assertIsInstance(dynamics_result, dict)
        self.assertIn("trajectory_analysis", dynamics_result)
        self.assertIn("stability_assessment", dynamics_result)
        self.assertIn("evolutionary_outcomes", dynamics_result)
        
    def test_ecosystem_emergence(self):
        """测试生态系统涌现"""
        # 模拟生态系统形成过程
        initial_conditions = {
            "agent_diversity": 5,
            "resource_types": 3,
            "interaction_strength": 0.6
        }
        
        emergence_result = self.coevolution.analyze_ecosystem_emergence(initial_conditions)
        self.assertIsInstance(emergence_result, dict)
        self.assertIn("ecosystem_properties", emergence_result)
        self.assertIn("emergent_phenomena", emergence_result)
        self.assertIn("ecosystem_stability", emergence_result)


class TestPopulationManager(unittest.TestCase):
    """种群管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.population_manager = PopulationManager()
        
    def test_population_initialization(self):
        """测试种群初始化"""
        # 设置初始化参数
        init_params = {
            "population_size": 20,
            "genome_length": 10,
            "diversity_targets": {"genetic": 0.8, "behavioral": 0.6}
        }
        
        population = self.population_manager.initialize_population(init_params)
        self.assertIsInstance(population, list)
        self.assertEqual(len(population), 20)
        
        for individual in population:
            self.assertIn("genome", individual)
            self.assertIn("traits", individual)
            
    def test_population_monitoring(self):
        """测试种群监控"""
        # 模拟种群状态
        test_population = [
            {
                "id": f"agent_{i}",
                "genome": np.random.rand(10),
                "fitness": np.random.rand(),
                "generation": i % 5
            }
            for i in range(15)
        ]
        
        monitoring_result = self.population_manager.monitor_population(test_population)
        self.assertIsInstance(monitoring_result, dict)
        self.assertIn("diversity_metrics", monitoring_result)
        self.assertIn("fitness_statistics", monitoring_result)
        self.assertIn("population_health", monitoring_result)
        
    def test_population_optimization(self):
        """测试种群优化"""
        # 模拟优化目标
        optimization_targets = {
            "max_fitness": 0.9,
            "min_diversity": 0.3,
            "population_size": 10
        }
        
        optimization_result = self.population_manager.optimize_population(optimization_targets)
        self.assertIsInstance(optimization_result, dict)
        self.assertIn("optimization_plan", optimization_result)
        self.assertIn("expected_outcomes", optimization_result)
        self.assertIn("resource_requirements", optimization_result)


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)
