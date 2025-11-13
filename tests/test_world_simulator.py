"""
世界模拟器测试模块

测试外部世界模拟器的功能，包括物理世界、社会世界和游戏世界的模拟。
"""

import unittest
import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from world_simulator import WorldSimulator, PhysicalWorld, SocialWorld, GameWorld


class TestWorldSimulator(unittest.TestCase):
    """世界模拟器基础测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.simulator = WorldSimulator()
        
    def tearDown(self):
        """测试后清理"""
        del self.simulator
        
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.simulator, WorldSimulator)
        self.assertIsNotNone(self.simulator.worlds)
        
    def test_world_registration(self):
        """测试世界注册"""
        # 测试注册新世界
        physical_world = PhysicalWorld()
        self.simulator.register_world("test_physical", physical_world)
        self.assertIn("test_physical", self.simulator.worlds)
        
    def test_world_simulation_step(self):
        """测试世界模拟步骤"""
        # 创建测试世界
        physical_world = PhysicalWorld()
        self.simulator.register_world("test", physical_world)
        
        # 执行模拟步骤
        initial_state = physical_world.get_state()
        result = self.simulator.simulate_step("test")
        
        self.assertIsInstance(result, dict)
        self.assertIn("timestamp", result)
        self.assertIn("state", result)


class TestPhysicalWorld(unittest.TestCase):
    """物理世界测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.physical_world = PhysicalWorld()
        
    def test_physics_simulation(self):
        """测试物理模拟"""
        # 测试基本物理计算
        result = self.physical_world.simulate_physics()
        self.assertIsInstance(result, dict)
        
    def test_object_interactions(self):
        """测试对象交互"""
        # 添加测试对象
        test_object = {
            "id": "test_obj",
            "position": [0, 0, 0],
            "velocity": [1, 0, 0],
            "mass": 1.0
        }
        
        self.physical_world.add_object(test_object)
        self.assertIn("test_obj", self.physical_world.objects)
        
    def test_gravity_simulation(self):
        """测试重力模拟"""
        result = self.physical_world.simulate_gravity()
        self.assertIsInstance(result, dict)
        self.assertIn("forces", result)


class TestSocialWorld(unittest.TestCase):
    """社会世界测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.social_world = SocialWorld()
        
    def test_agent_interaction(self):
        """测试智能体交互"""
        # 创建测试智能体
        agent1 = {
            "id": "agent1",
            "position": [0, 0],
            "mood": "neutral",
            "resources": 100
        }
        
        agent2 = {
            "id": "agent2", 
            "position": [10, 0],
            "mood": "happy",
            "resources": 50
        }
        
        self.social_world.add_agent(agent1)
        self.social_world.add_agent(agent2)
        
        # 测试交互
        interaction_result = self.social_world.process_interaction("agent1", "agent2")
        self.assertIsInstance(interaction_result, dict)
        
    def test_social_dynamics(self):
        """测试社会动态"""
        result = self.social_world.update_social_dynamics()
        self.assertIsInstance(result, dict)
        self.assertIn("mood_changes", result)


class TestGameWorld(unittest.TestCase):
    """游戏世界测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.game_world = GameWorld()
        
    def test_game_mechanics(self):
        """测试游戏机制"""
        result = self.game_world.apply_game_rules()
        self.assertIsInstance(result, dict)
        self.assertIn("score_changes", result)
        
    def test_player_action(self):
        """测试玩家行动"""
        # 添加测试玩家
        player = {
            "id": "player1",
            "position": [5, 5],
            "health": 100,
            "score": 0
        }
        
        self.game_world.add_player(player)
        
        # 测试行动
        action_result = self.game_world.process_player_action("player1", "move", [6, 5])
        self.assertIsInstance(action_result, dict)
        
    def test_reward_system(self):
        """测试奖励系统"""
        result = self.game_world.calculate_rewards()
        self.assertIsInstance(result, dict)
        self.assertIn("rewards", result)


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)
