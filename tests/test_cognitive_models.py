"""
认知模型测试模块

测试内部心智模型的各项认知能力实现。
"""

import unittest
import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cognitive_models import (
    CognitiveModel,
    MemorySystem,
    ThinkingSystem, 
    CreativitySystem,
    ObservationSystem,
    AttentionSystem,
    ImaginationSystem
)


class TestCognitiveModel(unittest.TestCase):
    """认知模型基础测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.cognitive_model = CognitiveModel()
        
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.cognitive_model, CognitiveModel)
        self.assertIsNotNone(self.cognitive_model.memory)
        self.assertIsNotNone(self.cognitive_model.thinking)
        self.assertIsNotNone(self.cognitive_model.creativity)
        
    def test_cognitive_process(self):
        """测试认知处理流程"""
        input_data = {
            "type": "observation",
            "content": "test observation",
            "timestamp": 1234567890
        }
        
        result = self.cognitive_model.process_cognitive_input(input_data)
        self.assertIsInstance(result, dict)
        self.assertIn("processed_data", result)
        self.assertIn("cognitive_state", result)


class TestMemorySystem(unittest.TestCase):
    """记忆系统测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.memory = MemorySystem()
        
    def test_short_term_memory(self):
        """测试短期记忆"""
        # 存储短期记忆
        memory_item = {
            "content": "test short-term memory",
            "timestamp": 1234567890,
            "importance": 0.5
        }
        
        memory_id = self.memory.store_short_term_memory(memory_item)
        self.assertIsNotNone(memory_id)
        
        # 检索短期记忆
        retrieved = self.memory.retrieve_short_term_memory(memory_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["content"], "test short-term memory")
        
    def test_long_term_memory(self):
        """测试长期记忆"""
        # 存储长期记忆
        memory_item = {
            "content": "test long-term memory",
            "timestamp": 1234567890,
            "importance": 0.9,
            "emotional_weight": 0.7
        }
        
        memory_id = self.memory.store_long_term_memory(memory_item)
        self.assertIsNotNone(memory_id)
        
        # 检索长期记忆
        retrieved = self.memory.retrieve_long_term_memory(memory_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["content"], "test long-term memory")
        
    def test_memory_consolidation(self):
        """测试记忆巩固"""
        # 创建短期记忆
        short_term_memories = [
            {"content": "memory 1", "timestamp": 1234567890},
            {"content": "memory 2", "timestamp": 1234567891},
            {"content": "memory 3", "timestamp": 1234567892}
        ]
        
        for mem in short_term_memories:
            self.memory.store_short_term_memory(mem)
            
        # 执行记忆巩固
        consolidation_result = self.memory.consolidate_memories()
        self.assertIsInstance(consolidation_result, dict)
        self.assertIn("consolidated_count", consolidation_result)


class TestThinkingSystem(unittest.TestCase):
    """思维系统测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.thinking = ThinkingSystem()
        
    def test_logical_reasoning(self):
        """测试逻辑推理"""
        premise1 = "All humans are mortal"
        premise2 = "Socrates is human"
        conclusion = "Socrates is mortal"
        
        result = self.thinking.logical_reasoning([premise1, premise2], conclusion)
        self.assertIsInstance(result, dict)
        self.assertIn("reasoning_valid", result)
        self.assertIn("confidence", result)
        
    def test_pattern_recognition(self):
        """测试模式识别"""
        # 测试序列模式识别
        sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        pattern_result = self.thinking.recognize_patterns(sequence)
        self.assertIsInstance(pattern_result, dict)
        self.assertIn("patterns_found", pattern_result)
        
    def test_problem_solving(self):
        """测试问题解决"""
        problem = {
            "description": "Find the shortest path from A to B",
            "constraints": ["avoid obstacles", "minimize distance"],
            "initial_state": "at point A",
            "goal_state": "at point B"
        }
        
        solution = self.thinking.solve_problem(problem)
        self.assertIsInstance(solution, dict)
        self.assertIn("solution", solution)
        self.assertIn("steps", solution)


class TestCreativitySystem(unittest.TestCase):
    """创造系统测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.creativity = CreativitySystem()
        
    def test_idea_generation(self):
        """测试创意生成"""
        context = {
            "domain": "art",
            "constraints": ["use blue color", "square format"],
            "inspiration": "nature"
        }
        
        ideas = self.creativity.generate_ideas(context)
        self.assertIsInstance(ideas, list)
        self.assertGreater(len(ideas), 0)
        
        # 检查创意质量
        for idea in ideas:
            self.assertIsInstance(idea, dict)
            self.assertIn("description", idea)
            self.assertIn("novelty_score", idea)
            
    def test_creative_combination(self):
        """测试创意组合"""
        concept1 = {"name": "bird", "properties": ["fly", "sing"]}
        concept2 = {"name": "car", "properties": ["drive", "transport"]}
        
        combination_result = self.creativity.combine_concepts(concept1, concept2)
        self.assertIsInstance(combination_result, dict)
        self.assertIn("new_concept", combination_result)
        self.assertIn("creativity_score", combination_result)
        
    def test_divergent_thinking(self):
        """测试发散思维"""
        starting_point = "design a new chair"
        directions = self.creativity.diverge_thinking(starting_point)
        
        self.assertIsInstance(directions, list)
        self.assertGreater(len(directions), 1)
        
        for direction in directions:
            self.assertIsInstance(direction, dict)
            self.assertIn("description", direction)


class TestObservationSystem(unittest.TestCase):
    """观察系统测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.observation = ObservationSystem()
        
    def test_visual_processing(self):
        """测试视觉处理"""
        # 模拟图像数据
        mock_image = np.random.rand(224, 224, 3)
        
        result = self.observation.process_visual_input(mock_image)
        self.assertIsInstance(result, dict)
        self.assertIn("detected_objects", result)
        self.assertIn("scene_description", result)
        
    def test_attention_mechanism(self):
        """测试注意机制"""
        scene_data = {
            "objects": [
                {"type": "cat", "position": [100, 100], "salience": 0.9},
                {"type": "dog", "position": [200, 200], "salience": 0.5},
                {"type": "car", "position": [300, 300], "salience": 0.3}
            ]
        }
        
        attention_result = self.observation.select_attention(scene_data)
        self.assertIsInstance(attention_result, dict)
        self.assertIn("attended_objects", attention_result)
        
    def test_scene_understanding(self):
        """测试场景理解"""
        scene_description = "A cat sitting on a table in a sunny room"
        
        understanding = self.observation.understand_scene(scene_description)
        self.assertIsInstance(understanding, dict)
        self.assertIn("objects", understanding)
        self.assertIn("relationships", understanding)
        self.assertIn("context", understanding)


class TestAttentionSystem(unittest.TestCase):
    """注意系统测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.attention = AttentionSystem()
        
    def test_selective_attention(self):
        """测试选择性注意"""
        stimuli = [
            {"type": "visual", "intensity": 0.8, "relevance": 0.9},
            {"type": "audio", "intensity": 0.6, "relevance": 0.3},
            {"type": "tactile", "intensity": 0.4, "relevance": 0.7}
        ]
        
        attention_result = self.attention.select_attention(stimuli)
        self.assertIsInstance(attention_result, dict)
        self.assertIn("attended_stimuli", attention_result)
        self.assertIn("attention_weights", attention_result)
        
    def test_attention_shifting(self):
        """测试注意转移"""
        current_attention = {"area": "visual_center", "focus": "object1"}
        new_target = {"area": "visual_periphery", "focus": "object2"}
        
        shifting_result = self.attention.shift_attention(current_attention, new_target)
        self.assertIsInstance(shifting_result, dict)
        self.assertIn("shift_success", shifting_result)
        self.assertIn("shift_cost", shifting_result)
        
    def test_attention_control(self):
        """测试注意控制"""
        goal = "focus on red objects in the scene"
        control_result = self.attention.exert_attention_control(goal)
        
        self.assertIsInstance(control_result, dict)
        self.assertIn("control_strategy", control_result)
        self.assertIn("effectiveness", control_result)


class TestImaginationSystem(unittest.TestCase):
    """想象系统测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.imagination = ImaginationSystem()
        
    def test_mental_simulation(self):
        """测试心理模拟"""
        scenario = {
            "description": "Imagine walking in a forest",
            "constraints": ["it's raining", "you have an umbrella"],
            "duration": 60  # seconds
        }
        
        simulation = self.imagination.simulate_scenario(scenario)
        self.assertIsInstance(simulation, dict)
        self.assertIn("simulated_experience", simulation)
        self.assertIn("imagery_quality", simulation)
        
    def test_future_planning(self):
        """测试未来规划"""
        current_state = {"location": "home", "resources": 50, "goal": "reach_work"}
        time_horizon = 30  # minutes
        
        plan = self.imagination.plan_future(current_state, time_horizon)
        self.assertIsInstance(plan, dict)
        self.assertIn("steps", plan)
        self.assertIn("expected_outcome", plan)
        
    def test_creative_imagination(self):
        """测试创造性想象"""
        prompt = "Imagine a creature that can fly and breathe underwater"
        constraints = ["must be fictional", "combines real animals"]
        
        creature = self.imagination.create_imaginary_entity(prompt, constraints)
        self.assertIsInstance(creature, dict)
        self.assertIn("description", creature)
        self.assertIn("characteristics", creature)
        self.assertIn("feasibility_score", creature)


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)
