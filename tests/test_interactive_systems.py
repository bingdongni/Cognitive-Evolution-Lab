"""
交互系统测试模块

测试交互行动系统的功能，包括具身智能和多模态感知。
"""

import unittest
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from interactive_systems import (
    InteractiveSystem,
    EmbodiedAgent,
    MultimodalPerception,
    ActionExecution,
    MotorControl
)


class TestInteractiveSystem(unittest.TestCase):
    """交互系统基础测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.interactive_system = InteractiveSystem()
        
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.interactive_system, InteractiveSystem)
        self.assertIsNotNone(self.interactive_system.embodied_agent)
        self.assertIsNotNone(self.interactive_system.perception)
        self.assertIsNotNone(self.interactive_system.action_execution)
        
    def test_perception_action_cycle(self):
        """测试感知-行动循环"""
        # 模拟感知输入
        sensory_input = {
            "visual": np.random.rand(224, 224, 3),
            "audio": np.random.rand(16000),
            "tactile": np.array([0.1, 0.2, 0.3])
        }
        
        # 执行感知-行动循环
        result = self.interactive_system.perception_action_cycle(sensory_input)
        self.assertIsInstance(result, dict)
        self.assertIn("perception_result", result)
        self.assertIn("action_result", result)
        self.assertIn("updated_state", result)


class TestEmbodiedAgent(unittest.TestCase):
    """具身智能体测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.agent = EmbodiedAgent()
        
    def test_agent_initialization(self):
        """测试智能体初始化"""
        self.assertIsInstance(self.agent, EmbodiedAgent)
        self.assertIsNotNone(self.agent.body_config)
        self.assertIsNotNone(self.agent.sensory_systems)
        self.assertIsNotNone(self.agent.motor_systems)
        
    def test_sensor_data_processing(self):
        """测试传感器数据处理"""
        # 模拟传感器数据
        sensor_data = {
            "camera": {"image": np.random.rand(224, 224, 3), "timestamp": 1234567890},
            "microphone": {"audio": np.random.rand(16000), "timestamp": 1234567890},
            "gyroscope": {"orientation": [0.1, 0.2, 0.3], "timestamp": 1234567890}
        }
        
        processed_data = self.agent.process_sensor_data(sensor_data)
        self.assertIsInstance(processed_data, dict)
        self.assertIn("unified_state", processed_data)
        self.assertIn("confidence_scores", processed_data)
        
    def test_body_dynamics(self):
        """测试身体动力学"""
        # 设置初始身体状态
        initial_state = {
            "position": [0, 0, 0],
            "velocity": [0, 0, 0],
            "orientation": [0, 0, 0],
            "joint_angles": [0.0] * 7  # 假设7个关节
        }
        
        # 执行动力学更新
        dynamics_result = self.agent.update_body_dynamics(initial_state, forces=[1, 0, 0])
        self.assertIsInstance(dynamics_result, dict)
        self.assertIn("new_state", dynamics_result)
        self.assertIn("acceleration", dynamics_result)
        
    def testProprioception(self):
        """测试本体感知"""
        # 设置关节状态
        joint_states = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        proprioception = self.agent.get_proprioception(joint_states)
        
        self.assertIsInstance(proprioception, dict)
        self.assertIn("joint_positions", proprioception)
        self.assertIn("joint_velocities", proprioception)
        self.assertIn("force_feedback", proprioception)


class TestMultimodalPerception(unittest.TestCase):
    """多模态感知测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.perception = MultimodalPerception()
        
    def test_visual_processing(self):
        """测试视觉处理"""
        # 模拟图像输入
        image = np.random.rand(224, 224, 3)
        
        vision_result = self.perception.process_visual_input(image)
        self.assertIsInstance(vision_result, dict)
        self.assertIn("objects_detected", vision_result)
        self.assertIn("scene_layout", vision_result)
        self.assertIn("saliency_map", vision_result)
        
    def test_auditory_processing(self):
        """测试听觉处理"""
        # 模拟音频输入
        audio_signal = np.random.rand(16000)  # 1秒音频
        
        audio_result = self.perception.process_auditory_input(audio_signal)
        self.assertIsInstance(audio_result, dict)
        self.assertIn("sound_sources", audio_result)
        self.assertIn("speech_recognition", audio_result)
        self.assertIn("environmental_sounds", audio_result)
        
    def test_tactile_processing(self):
        """测试触觉处理"""
        # 模拟触觉输入
        tactile_data = np.random.rand(100)  # 100个触觉传感器
        
        tactile_result = self.perception.process_tactile_input(tactile_data)
        self.assertIsInstance(tactile_result, dict)
        self.assertIn("contact_detected", tactile_result)
        self.assertIn("pressure_map", tactile_result)
        self.assertIn("texture_features", tactile_result)
        
    def test_sensor_fusion(self):
        """测试多传感器融合"""
        # 融合多种传感器数据
        sensor_data = {
            "visual": {"objects": [{"id": 1, "confidence": 0.9}]},
            "auditory": {"sounds": [{"id": 1, "intensity": 0.8}]},
            "tactile": {"contacts": [{"id": 1, "pressure": 0.7}]}
        }
        
        fused_result = self.perception.fuse_sensor_data(sensor_data)
        self.assertIsInstance(fused_result, dict)
        self.assertIn("unified_environment_model", fused_result)
        self.assertIn("confidence_scores", fused_result)


class TestActionExecution(unittest.TestCase):
    """行动执行测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.action_executor = ActionExecution()
        
    def test_action_planning(self):
        """测试行动规划"""
        # 设定目标
        goal = {
            "type": "reach_object",
            "target": {"position": [1.0, 0.5, 0.2], "id": "cup"},
            "constraints": ["avoid_obstacles", "minimize_time"]
        }
        
        plan = self.action_executor.plan_action(goal)
        self.assertIsInstance(plan, dict)
        self.assertIn("action_sequence", plan)
        self.assertIn("expected_duration", plan)
        self.assertIn("success_probability", plan)
        
    def test_motion_control(self):
        """测试运动控制"""
        # 设定运动目标
        motion_goal = {
            "trajectory": [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]],
            "duration": 1.0,
            "smoothness": 0.8
        }
        
        control_result = self.action_executor.execute_motion(motion_goal)
        self.assertIsInstance(control_result, dict)
        self.assertIn("control_signals", control_result)
        self.assertIn("trajectory_following", control_result)
        
    def test_grasp_control(self):
        """测试抓取控制"""
        # 设定抓取目标
        grasp_target = {
            "object_position": [0.5, 0.0, 0.3],
            "object_size": [0.1, 0.1, 0.1],
            "grasp_type": "precision"
        }
        
        grasp_result = self.action_executor.execute_grasp(grasp_target)
        self.assertIsInstance(grasp_result, dict)
        self.assertIn("gripper_config", grasp_result)
        self.assertIn("grasp_quality", grasp_result)
        self.assertIn("approach_trajectory", grasp_result)
        
    def test_adaptive_control(self):
        """测试自适应控制"""
        # 模拟控制过程中的环境变化
        initial_plan = {
            "action": "move_to_position",
            "target": [1.0, 0.0, 0.0]
        }
        
        environmental_change = {
            "obstacle_detected": True,
            "obstacle_position": [0.5, 0.0, 0.0],
            "change_time": 0.5
        }
        
        adapted_result = self.action_executor.adapt_control(initial_plan, environmental_change)
        self.assertIsInstance(adapted_result, dict)
        self.assertIn("new_plan", adapted_result)
        self.assertIn("adaptation_success", adapted_result)
        self.assertIn("performance_impact", adapted_result)


class TestMotorControl(unittest.TestCase):
    """运动控制测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.motor_control = MotorControl()
        
    def test_joint_control(self):
        """测试关节控制"""
        # 设定关节目标角度
        joint_targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        control_result = self.motor_control.control_joints(joint_targets)
        self.assertIsInstance(control_result, dict)
        self.assertIn("control_signals", control_result)
        self.assertIn("tracking_error", control_result)
        self.assertIn("stability_metrics", control_result)
        
    def test_force_control(self):
        """测试力控制"""
        # 设定力控制目标
        force_targets = {
            "end_effector_force": [1.0, 0.0, 0.0],
            "end_effector_torque": [0.0, 0.0, 0.0]
        }
        
        force_result = self.motor_control.control_forces(force_targets)
        self.assertIsInstance(force_result, dict)
        self.assertIn("force_commands", force_result)
        self.assertIn("force_feedback", force_result)
        self.assertIn("compliance_error", force_result)
        
    def test_trajectory_tracking(self):
        """测试轨迹跟踪"""
        # 设定目标轨迹
        target_trajectory = {
            "positions": [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]],
            "velocities": [[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0]],
            "accelerations": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        }
        
        tracking_result = self.motor_control.track_trajectory(target_trajectory)
        self.assertIsInstance(tracking_result, dict)
        self.assertIn("tracking_error", tracking_result)
        self.assertIn("control_effort", tracking_result)
        self.assertIn("smoothness", tracking_result)
        
    def test_impedance_control(self):
        """测试阻抗控制"""
        # 设定阻抗参数
        impedance_params = {
            "stiffness": [10.0, 10.0, 10.0],
            "damping": [0.5, 0.5, 0.5],
            "inertia": [1.0, 1.0, 1.0]
        }
        
        impedance_result = self.motor_control.set_impedance(impedance_params)
        self.assertIsInstance(impedance_result, dict)
        self.assertIn("impedance_matrix", impedance_result)
        self.assertIn("stability_check", impedance_result)


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)
