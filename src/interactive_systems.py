#!/usr/bin/env python3
"""
Cognitive Evolution Lab - 交互行动系统
作者: bingdongni

实现交互行动系统，包括：
- 具身智能（运动控制、感觉融合、平衡控制）
- 多模态感知（视觉、听觉、触觉、文本）
- 动作规划（策略生成、执行控制、安全约束）
- 环境交互（物理接触、社交互动、任务执行）
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import random
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import deque, defaultdict

# 尝试导入相关库
try:
    import cv2
    OPENCV_AV认知计算LABLE = True
except ImportError:
    OPENCV_AV认知计算LABLE = False

try:
    import pygame
    PYGAME_AV认知计算LABLE = True
except ImportError:
    PYGAME_AV认知计算LABLE = False


class ActionType(Enum):
    """动作类型枚举"""
    MOTOR_ACTION = "motor_action"      # 电机动作
    MANIPULATION = "manipulation"      # 操作动作
    COMMUNICATION = "communication"    # 交流动作
    COGNITIVE_ACTION = "cognitive_action"  # 认知动作
    SOCIAL_ACTION = "social_action"    # 社交动作


class SensorType(Enum):
    """传感器类型枚举"""
    VISION = "vision"
    AUDIO = "audio"
    TOUCH = "touch"
    PROPRIOCEPTION = "proprioception"
    TEXT = "text"


@dataclass
class SensorReading:
    """传感器读数"""
    sensor_type: SensorType
    data: Any
    timestamp: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionCommand:
    """动作命令"""
    action_type: ActionType
    parameters: Dict[str, Any]
    duration: float
    priority: int
    safety_constraints: List[str] = field(default_factory=list)


@dataclass
class BodyState:
    """身体状态"""
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    joint_angles: np.ndarray
    joint_velocities: np.ndarray
    force_sensors: Dict[str, float]
    balance_metrics: Dict[str, float]


@dataclass
class InteractionEvent:
    """交互事件"""
    event_type: str
    participants: List[str]
    intensity: float
    timestamp: float
    outcome: Dict[str, Any] = field(default_factory=dict)


class MultimodalPerception:
    """多模态感知模块"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 各模态配置
        self.vision_config = config.get('vision', {})
        self.audio_config = config.get('audio', {})
        self.touch_config = config.get('touch', {})
        self.text_config = config.get('text', {})
        
        # 初始化各模态处理器
        self.vision_processor = None
        self.audio_processor = None
        self.touch_processor = None
        self.text_processor = None
        
        # 传感器数据缓冲
        self.sensor_buffers = {
            SensorType.VISION: deque(maxlen=100),
            SensorType.AUDIO: deque(maxlen=50),
            SensorType.TOUCH: deque(maxlen=20),
            SensorType.PROPRIOCEPTION: deque(maxlen=30),
            SensorType.TEXT: deque(maxlen=10)
        }
        
        # 感知融合器
        self.perception_fusion = None
        
        self.logger.info("🔍 多模态感知系统初始化完成")
    
    async def initialize(self):
        """初始化各模态处理器"""
        self.logger.info("🔧 初始化多模态处理器...")
        
        try:
            # 初始化视觉处理
            await self._initialize_vision_processing()
            
            # 初始化听觉处理
            await self._initialize_audio_processing()
            
            # 初始化触觉处理
            await self._initialize_touch_processing()
            
            # 初始化文本处理
            await self._initialize_text_processing()
            
            # 初始化感知融合
            await self._initialize_perception_fusion()
            
            self.logger.info("✅ 多模态处理器初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 多模态处理器初始化失败: {e}")
            raise
    
    async def _initialize_vision_processing(self):
        """初始化视觉处理"""
        if not OPENCV_AV认知计算LABLE:
            self.logger.warning("⚠️ OpenCV不可用，使用简化视觉处理")
            self.vision_processor = self._simple_vision_processor
            return
        
        # 初始化视觉模型（这里使用简化的模型）
        self.vision_processor = self._opencv_vision_processor
        
        self.logger.info("✅ 视觉处理初始化完成")
    
    async def _initialize_audio_processing(self):
        """初始化听觉处理"""
        # 简化的音频处理
        self.audio_processor = self._simple_audio_processor
        self.logger.info("✅ 听觉处理初始化完成")
    
    async def _initialize_touch_processing(self):
        """初始化触觉处理"""
        # 简化的触觉处理
        self.touch_processor = self._simple_touch_processor
        self.logger.info("✅ 触觉处理初始化完成")
    
    async def _initialize_text_processing(self):
        """初始化文本处理"""
        # 简化的文本处理
        self.text_processor = self._simple_text_processor
        self.logger.info("✅ 文本处理初始化完成")
    
    async def _initialize_perception_fusion(self):
        """初始化感知融合"""
        # 简化的感知融合
        self.perception_fusion = self._simple_fusion
        self.logger.info("✅ 感知融合初始化完成")
    
    async def capture_vision(self) -> SensorReading:
        """捕获视觉信息"""
        try:
            vision_data = await self.vision_processor()
            
            reading = SensorReading(
                sensor_type=SensorType.VISION,
                data=vision_data,
                timestamp=self._get_timestamp(),
                confidence=0.9,
                metadata={'resolution': '640x480', 'fps': 30}
            )
            
            self.sensor_buffers[SensorType.VISION].append(reading)
            return reading
            
        except Exception as e:
            self.logger.error(f"视觉捕获失败: {e}")
            return self._create_fallback_reading(SensorType.VISION)
    
    async def capture_audio(self) -> SensorReading:
        """捕获听觉信息"""
        try:
            audio_data = await self.audio_processor()
            
            reading = SensorReading(
                sensor_type=SensorType.AUDIO,
                data=audio_data,
                timestamp=self._get_timestamp(),
                confidence=0.8,
                metadata={'sample_rate': 16000, 'channels': 1}
            )
            
            self.sensor_buffers[SensorType.AUDIO].append(reading)
            return reading
            
        except Exception as e:
            self.logger.error(f"音频捕获失败: {e}")
            return self._create_fallback_reading(SensorType.AUDIO)
    
    async def capture_touch(self) -> SensorReading:
        """捕获触觉信息"""
        try:
            touch_data = await self.touch_processor()
            
            reading = SensorReading(
                sensor_type=SensorType.TOUCH,
                data=touch_data,
                timestamp=self._get_timestamp(),
                confidence=0.85,
                metadata={'sensor_count': 100, 'sensitivity': 0.8}
            )
            
            self.sensor_buffers[SensorType.TOUCH].append(reading)
            return reading
            
        except Exception as e:
            self.logger.error(f"触觉捕获失败: {e}")
            return self._create_fallback_reading(SensorType.TOUCH)
    
    async def capture_proprioception(self, body_state: BodyState) -> SensorReading:
        """捕获本体感受"""
        proprioceptive_data = {
            'position': body_state.position,
            'velocity': body_state.velocity,
            'joint_angles': body_state.joint_angles,
            'joint_velocities': body_state.joint_velocities,
            'balance_metrics': body_state.balance_metrics
        }
        
        reading = SensorReading(
            sensor_type=SensorType.PROPRIOCEPTION,
            data=proprioceptive_data,
            timestamp=self._get_timestamp(),
            confidence=0.95,
            metadata={'joint_count': len(body_state.joint_angles)}
        )
        
        self.sensor_buffers[SensorType.PROPRIOCEPTION].append(reading)
        return reading
    
    async def process_text_input(self, text: str) -> SensorReading:
        """处理文本输入"""
        text_data = await self.text_processor(text)
        
        reading = SensorReading(
            sensor_type=SensorType.TEXT,
            data=text_data,
            timestamp=self._get_timestamp(),
            confidence=0.9,
            metadata={'input_length': len(text)}
        )
        
        self.sensor_buffers[SensorType.TEXT].append(reading)
        return reading
    
    async def _opencv_vision_processor(self) -> Dict[str, Any]:
        """OpenCV视觉处理器"""
        if not OPENCV_AV认知计算LABLE:
            return self._simple_vision_processor()
        
        try:
            # 模拟相机输入（实际应用中会从真实相机获取）
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # 简化的特征提取
            features = {
                'objects': self._detect_objects_simple(frame),
                'depth': np.random.uniform(0.5, 10.0, (480, 640)),
                'motion': np.random.random((480, 640, 2)) * 0.1,
                'brightness': float(np.mean(frame)),
                'contrast': float(np.std(frame))
            }
            
            return {
                'frame': frame,
                'features': features,
                'timestamp': self._get_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"OpenCV视觉处理失败: {e}")
            return self._simple_vision_processor()
    
    def _simple_vision_processor(self) -> Dict[str, Any]:
        """简化视觉处理器"""
        return {
            'simulated_view': True,
            'objects': ['object1', 'object2', 'object3'],
            'depth_map': np.random.uniform(0.5, 10.0, (100, 100)),
            'brightness': 128.0,
            'motion_vectors': np.random.random((10, 10, 2)),
            'timestamp': self._get_timestamp()
        }
    
    async def _simple_audio_processor(self) -> Dict[str, Any]:
        """简化音频处理器"""
        # 模拟音频数据
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        audio_data = np.random.normal(0, 0.1, samples)
        
        features = {
            'mfcc': np.random.random((13, 100)),
            'spectral_centroid': np.random.uniform(1000, 4000),
            'rms_energy': float(np.sqrt(np.mean(audio_data**2))),
            'zero_crossing_rate': np.random.uniform(0.1, 0.3)
        }
        
        return {
            'audio_data': audio_data,
            'features': features,
            'sample_rate': sample_rate,
            'duration': duration,
            'timestamp': self._get_timestamp()
        }
    
    async def _simple_touch_processor(self) -> Dict[str, Any]:
        """简化触觉处理器"""
        sensor_count = 100
        touch_data = {
            'pressure': np.random.uniform(0, 1, sensor_count),
            'temperature': np.random.uniform(20, 40, sensor_count),
            'vibration': np.random.uniform(0, 0.1, sensor_count),
            'texture': np.random.uniform(0, 1, sensor_count)
        }
        
        return {
            'touch_data': touch_data,
            'sensor_count': sensor_count,
            'timestamp': self._get_timestamp()
        }
    
    async def _simple_text_processor(self, text: str) -> Dict[str, Any]:
        """简化文本处理器"""
        words = text.lower().split()
        
        features = {
            'word_count': len(words),
            'sentiment_score': np.random.uniform(-1, 1),
            'keywords': words[:5],  # 前5个词作为关键词
            'language': 'chinese' if any(ord(char) > 127 for char in text) else 'english',
            'complexity': len(set(words)) / len(words) if words else 0
        }
        
        return {
            'text': text,
            'features': features,
            'timestamp': self._get_timestamp()
        }
    
    def _detect_objects_simple(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """简化物体检测"""
        # 模拟物体检测结果
        object_count = random.randint(3, 8)
        objects = []
        
        for i in range(object_count):
            obj = {
                'class': random.choice(['person', 'car', 'dog', 'cat', 'chair', 'table']),
                'bbox': [random.randint(0, 100) for _ in range(4)],
                'confidence': random.uniform(0.6, 0.95),
                'center': (random.randint(0, 640), random.randint(0, 480))
            }
            objects.append(obj)
        
        return objects
    
    def _create_fallback_reading(self, sensor_type: SensorType) -> SensorReading:
        """创建备用读数"""
        return SensorReading(
            sensor_type=sensor_type,
            data=None,
            timestamp=self._get_timestamp(),
            confidence=0.1,
            metadata={'error': 'sensor_unavailable'}
        )
    
    async def fuse_perceptions(self) -> Dict[str, Any]:
        """融合多模态感知"""
        if not self.perception_fusion:
            return {'error': 'fusion_not_available'}
        
        # 获取最新感知数据
        latest_readings = {}
        for sensor_type in SensorType:
            if self.sensor_buffers[sensor_type]:
                latest_readings[sensor_type] = self.sensor_buffers[sensor_type][-1]
        
        # 执行感知融合
        fused_perception = await self.perception_fusion(latest_readings)
        
        return fused_perception
    
    async def _simple_fusion(self, readings: Dict[SensorType, SensorReading]) -> Dict[str, Any]:
        """简化感知融合"""
        fusion_result = {
            'timestamp': self._get_timestamp(),
            'modalities_available': list(readings.keys()),
            'confidence_scores': {},
            'fused_state': {},
            'conflicts': [],
            'consensus': {}
        }
        
        # 计算融合置信度
        for sensor_type, reading in readings.items():
            fusion_result['confidence_scores'][sensor_type.value] = reading.confidence
        
        # 检测冲突
        if SensorType.VISION in readings and SensorType.TOUCH in readings:
            # 简单的冲突检测逻辑
            fusion_result['conflicts'].append({
                'type': 'vision_touch_mismatch',
                'description': '视觉和触觉信息存在差异'
            })
        
        # 计算共识
        if len(readings) > 1:
            avg_confidence = np.mean([r.confidence for r in readings.values()])
            fusion_result['consensus']['overall_confidence'] = avg_confidence
        
        # 融合状态
        fusion_result['fused_state'] = {
            'environment_type': 'mixed_modal',
            'objects_detected': len(readings.get(SensorType.VISION, {}).get('data', {}).get('objects', [])),
            'audio_level': readings.get(SensorType.AUDIO, {}).get('data', {}).get('features', {}).get('rms_energy', 0),
            'touch_pressure': np.mean(readings.get(SensorType.TOUCH, {}).get('data', {}).get('touch_data', {}).get('pressure', [0])),
            'text_input': readings.get(SensorType.TEXT, {}).get('data', {}).get('text', '')
        }
        
        return fusion_result
    
    def _get_timestamp(self) -> float:
        """获取时间戳"""
        import time
        return time.time()


class MotorController(nn.Module):
    """运动控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 动作空间
        self.action_space = config.get('action_space', {
            'linear_velocity': [-2.0, 2.0],
            'angular_velocity': [-2.0, 2.0],
            'gripper_position': [0.0, 1.0],
            'joint_positions': [0.0, 1.0]
        })
        
        # 控制器类型
        self.controller_type = config.get('type', 'pid')
        
        # PID控制器参数
        self.pid_gains = config.get('pid_gains', {
            'kp': 1.0, 'ki': 0.1, 'kd': 0.05
        })
        
        # 平衡控制器
        self.balance_controller = config.get('balance_control', True)
        
        # 安全约束
        self.safety_constraints = config.get('safety_constraints', {
            'max_velocity': 5.0,
            'max_acceleration': 2.0,
            'force_limit': 100.0,
            'torque_limit': 50.0
        })
        
        # 运动预测模型
        self.motion_predictor = self._build_motion_predictor()
    
    def _build_motion_predictor(self) -> nn.Module:
        """构建运动预测模型"""
        class MotionPredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(10, 32, 2, batch_first=True)
                self.fc = nn.Linear(32, 6)  # 预测位置、速度、加速度
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                prediction = self.fc(lstm_out[:, -1, :])
                return prediction
        
        return MotionPredictor()
    
    def forward(self, current_state: BodyState, target_state: BodyState, dt: float) -> Dict[str, Any]:
        """前向传播执行运动控制"""
        # 运动规划
        trajectory = self._plan_trajectory(current_state, target_state, dt)
        
        # 安全检查
        safe_trajectory = self._apply_safety_constraints(trajectory)
        
        # 执行控制
        control_commands = self._generate_control_commands(safe_trajectory, current_state)
        
        return {
            'trajectory': safe_trajectory,
            'control_commands': control_commands,
            'safety_status': self._check_safety_status(control_commands),
            'balance_metrics': self._calculate_balance_metrics(current_state, control_commands)
        }
    
    def _plan_trajectory(self, current_state: BodyState, target_state: BodyState, dt: float) -> Dict[str, Any]:
        """规划运动轨迹"""
        # 计算位置差
        pos_diff = target_state.position - current_state.position
        
        # 生成五次多项式轨迹
        trajectory = self._generate_quintic_trajectory(pos_diff, dt)
        
        # 添加平滑处理
        trajectory = self._smooth_trajectory(trajectory)
        
        return trajectory
    
    def _generate_quintic_trajectory(self, target_pos: np.ndarray, duration: float) -> Dict[str, np.ndarray]:
        """生成五次多项式轨迹"""
        # 时间数组
        t = np.linspace(0, duration, int(duration / 0.01))
        
        # 五次多项式系数（简化版）
        # p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        a0 = 0.0
        a1 = 0.0
        a2 = 0.0
        a3 = 10.0 / (duration**3)
        a4 = -15.0 / (duration**4)
        a5 = 6.0 / (duration**5)
        
        # 轨迹计算
        trajectory = {}
        for i, coord in enumerate(['x', 'y', 'z']):
            if i < len(target_pos):
                position = a0 + a1*t + a2*(t**2) + a3*(t**3) + a4*(t**4) + a5*(t**5)
                velocity = a1 + 2*a2*t + 3*a3*(t**2) + 4*a4*(t**3) + 5*a5*(t**4)
                acceleration = 2*a2 + 6*a3*t + 12*a4*(t**2) + 20*a5*(t**3)
                
                trajectory[coord] = {
                    'position': position * target_pos[i],
                    'velocity': velocity * target_pos[i],
                    'acceleration': acceleration * target_pos[i]
                }
        
        return trajectory
    
    def _smooth_trajectory(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """轨迹平滑处理"""
        smoothed_trajectory = {}
        
        for coord, data in trajectory.items():
            # 简单移动平均
            window_size = 5
            position = np.array(data['position'])
            velocity = np.array(data['velocity'])
            acceleration = np.array(data['acceleration'])
            
            # 移动平均
            smoothed_position = np.convolve(position, np.ones(window_size)/window_size, mode='same')
            smoothed_velocity = np.convolve(velocity, np.ones(window_size)/window_size, mode='same')
            smoothed_acceleration = np.convolve(acceleration, np.ones(window_size)/window_size, mode='same')
            
            smoothed_trajectory[coord] = {
                'position': smoothed_position,
                'velocity': smoothed_velocity,
                'acceleration': smoothed_acceleration
            }
        
        return smoothed_trajectory
    
    def _apply_safety_constraints(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """应用安全约束"""
        safe_trajectory = trajectory.copy()
        
        # 检查速度限制
        for coord, data in safe_trajectory.items():
            velocity = data['velocity']
            max_vel = self.safety_constraints['max_velocity']
            
            # 限速
            velocity = np.clip(velocity, -max_vel, max_vel)
            data['velocity'] = velocity
            
            # 重新计算加速度
            if len(velocity) > 1:
                data['acceleration'] = np.gradient(velocity)
        
        return safe_trajectory
    
    def _generate_control_commands(self, trajectory: Dict[str, Any], current_state: BodyState) -> Dict[str, Any]:
        """生成控制命令"""
        commands = {
            'motor_commands': {},
            'gripper_commands': {},
            'joint_commands': {}
        }
        
        # 为每个坐标轴生成电机命令
        for coord, data in trajectory.items():
            if coord in ['x', 'y', 'z']:
                # 简化的PID控制
                target_velocity = data['velocity'][0] if len(data['velocity']) > 0 else 0.0
                current_velocity = getattr(current_state.velocity, coord, 0.0)
                
                # PID控制律
                error = target_velocity - current_velocity
                kp = self.pid_gains['kp']
                ki = self.pid_gains['ki']
                kd = self.pid_gains['kd']
                
                # 简化的PID计算
                motor_command = kp * error
                
                commands['motor_commands'][f'{coord}_velocity'] = motor_command
        
        # 关节控制命令
        joint_count = len(current_state.joint_angles)
        for i in range(joint_count):
            commands['joint_commands'][f'joint_{i}'] = {
                'position': current_state.joint_angles[i],
                'velocity': current_state.joint_velocities[i]
            }
        
        return commands
    
    def _check_safety_status(self, commands: Dict[str, Any]) -> Dict[str, Any]:
        """检查安全状态"""
        safety_status = {
            'is_safe': True,
            'violations': [],
            'warnings': []
        }
        
        # 检查电机速度
        motor_commands = commands.get('motor_commands', {})
        for cmd_name, cmd_value in motor_commands.items():
            if 'velocity' in cmd_name:
                if abs(cmd_value) > self.safety_constraints['max_velocity']:
                    safety_status['is_safe'] = False
                    safety_status['violations'].append(f'{cmd_name} exceeds max velocity')
        
        # 检查力矩限制
        if safety_status['violations']:
            safety_status['warnings'].append('Performance degraded due to safety constraints')
        
        return safety_status
    
    def _calculate_balance_metrics(self, current_state: BodyState, commands: Dict[str, Any]) -> Dict[str, float]:
        """计算平衡指标"""
        metrics = {
            'center_of_mass_stability': 1.0,  # 简化计算
            'support_polygon_margin': 0.5,
            'angular_momentum': 0.0,
            'balance_score': 0.8
        }
        
        if self.balance_controller:
            # 增强的平衡控制
            balance_error = np.linalg.norm(current_state.velocity)
            metrics['balance_score'] = max(0.0, 1.0 - balance_error / 10.0)
            metrics['balance_error'] = balance_error
        
        return metrics


class ActionPlanner:
    """动作规划器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 规划参数
        self.horizon = config.get('horizon', 10)
        self.replanning_rate = config.get('replanning_rate', 0.1)
        self.uncertainty_handling = config.get('uncertainty_handling', 'monte_carlo')
        self.safety_constraints = config.get('safety_constraints', True)
        
        # 目标跟踪
        self.current_goals = []
        self.goal_history = []
        
        # 规划历史
        self.plan_history = deque(maxlen=100)
        
        self.logger.info("🎯 动作规划器初始化完成")
    
    async def plan_action(self, current_state: BodyState, perception_fusion: Dict[str, Any], 
                         goals: List[str]) -> Dict[str, Any]:
        """规划动作"""
        # 更新目标
        self._update_goals(goals)
        
        # 生成候选动作
        candidate_actions = await self._generate_candidate_actions(current_state, perception_fusion)
        
        # 评估动作
        action_evaluations = await self._evaluate_actions(candidate_actions, current_state, goals)
        
        # 选择最优动作
        best_action = self._select_best_action(action_evaluations)
        
        # 生成执行计划
        execution_plan = await self._generate_execution_plan(best_action, current_state)
        
        # 存储规划历史
        self.plan_history.append({
            'action': best_action,
            'evaluation': action_evaluations[best_action['type']],
            'timestamp': self._get_timestamp()
        })
        
        return execution_plan
    
    def _update_goals(self, goals: List[str]):
        """更新目标"""
        # 添加新目标
        for goal in goals:
            if goal not in [g['goal'] for g in self.current_goals]:
                self.current_goals.append({
                    'goal': goal,
                    'priority': 1.0,
                    'deadline': self._get_timestamp() + 300,  # 5分钟
                    'status': 'active'
                })
        
        # 清理过期目标
        current_time = self._get_timestamp()
        self.current_goals = [g for g in self.current_goals if g['deadline'] > current_time]
        
        # 更新目标历史
        for goal in self.current_goals:
            if goal not in self.goal_history:
                self.goal_history.append(goal)
    
    async def _generate_candidate_actions(self, current_state: BodyState, 
                                         perception_fusion: Dict[str, Any]) -> List[ActionCommand]:
        """生成候选动作"""
        candidate_actions = []
        
        # 基于当前状态生成基础动作
        base_actions = [
            ActionCommand(
                action_type=ActionType.MOTOR_ACTION,
                parameters={'move_to': 'forward', 'distance': 1.0},
                duration=2.0,
                priority=5
            ),
            ActionCommand(
                action_type=ActionType.MOTOR_ACTION,
                parameters={'rotate': 'left', 'angle': 0.5},
                duration=1.0,
                priority=4
            ),
            ActionCommand(
                action_type=ActionType.MANIPULATION,
                parameters={'grasp_object': 'target'},
                duration=3.0,
                priority=6
            ),
            ActionCommand(
                action_type=ActionType.COMMUNICATION,
                parameters={'speak': 'hello', 'gesture': True},
                duration=2.0,
                priority=3
            )
        ]
        
        candidate_actions.extend(base_actions)
        
        # 基于感知融合生成适应动作
        if perception_fusion.get('fused_state', {}).get('objects_detected', 0) > 0:
            candidate_actions.append(ActionCommand(
                action_type=ActionType.COGNITIVE_ACTION,
                parameters={'observe_objects': True, 'focus_attention': 'objects'},
                duration=1.0,
                priority=7
            ))
        
        # 基于文本输入生成交流动作
        text_input = perception_fusion.get('fused_state', {}).get('text_input', '')
        if text_input:
            candidate_actions.append(ActionCommand(
                action_type=ActionType.COMMUNICATION,
                parameters={'respond_to_text': text_input},
                duration=3.0,
                priority=8
            ))
        
        return candidate_actions
    
    async def _evaluate_actions(self, candidate_actions: List[ActionCommand], 
                               current_state: BodyState, goals: List[str]) -> Dict[str, ActionCommand]:
        """评估动作"""
        evaluations = {}
        
        for action in candidate_actions:
            score = 0.0
            
            # 目标匹配度
            goal_match = self._calculate_goal_match(action, goals)
            score += goal_match * 0.4
            
            # 安全性评估
            safety_score = self._assess_safety(action, current_state)
            score += safety_score * 0.3
            
            # 效率评估
            efficiency_score = self._assess_efficiency(action, current_state)
            score += efficiency_score * 0.2
            
            # 可行性评估
            feasibility_score = self._assess_feasibility(action, current_state)
            score += feasibility_score * 0.1
            
            evaluations[action.action_type.value] = {
                'action': action,
                'total_score': score,
                'goal_match': goal_match,
                'safety_score': safety_score,
                'efficiency_score': efficiency_score,
                'feasibility_score': feasibility_score
            }
        
        return evaluations
    
    def _calculate_goal_match(self, action: ActionCommand, goals: List[str]) -> float:
        """计算目标匹配度"""
        # 简化的目标匹配度计算
        action_keywords = []
        for param_value in action.parameters.values():
            if isinstance(param_value, str):
                action_keywords.append(param_value.lower())
        
        match_count = 0
        for goal in goals:
            goal_words = goal.lower().split()
            for word in goal_words:
                if any(keyword in word for keyword in action_keywords):
                    match_count += 1
        
        return min(1.0, match_count / max(1, len(goals)))
    
    def _assess_safety(self, action: ActionCommand, current_state: BodyState) -> float:
        """评估安全性"""
        safety_score = 1.0
        
        # 检查动作安全性
        if action.action_type == ActionType.MOTOR_ACTION:
            velocity = action.parameters.get('velocity', 0)
            if abs(velocity) > 2.0:
                safety_score *= 0.7
        
        # 检查当前位置的安全性
        pos = current_state.position
        if np.linalg.norm(pos) < 0.5:  # 接近原点
            safety_score *= 0.8
        
        return safety_score
    
    def _assess_efficiency(self, action: ActionCommand, current_state: BodyState) -> float:
        """评估效率"""
        # 基于动作持续时间和目标距离的效率评估
        duration = action.duration
        priority = action.priority
        
        # 短时间高优先级的动作更有效
        efficiency = priority / (duration + 1.0)
        return min(1.0, efficiency / 10.0)
    
    def _assess_feasibility(self, action: ActionCommand, current_state: BodyState) -> float:
        """评估可行性"""
        feasibility = 1.0
        
        # 检查当前状态是否支持该动作
        if action.action_type == ActionType.MANIPULATION:
            # 检查是否有足够的平衡性
            balance_score = current_state.balance_metrics.get('balance_score', 0.5)
            feasibility *= balance_score
        
        return feasibility
    
    def _select_best_action(self, evaluations: Dict[str, Any]) -> ActionCommand:
        """选择最优动作"""
        if not evaluations:
            return ActionCommand(
                action_type=ActionType.COGNITIVE_ACTION,
                parameters={'wait': True},
                duration=1.0,
                priority=1
            )
        
        # 按总分排序
        best_action_type = max(evaluations.keys(), 
                             key=lambda x: evaluations[x]['total_score'])
        
        return evaluations[best_action_type]['action']
    
    async def _generate_execution_plan(self, action: ActionCommand, current_state: BodyState) -> Dict[str, Any]:
        """生成执行计划"""
        execution_plan = {
            'primary_action': action,
            'execution_steps': [],
            'fallback_actions': [],
            'monitoring_points': [],
            'expected_outcome': {},
            'risk_assessment': {},
            'timestamp': self._get_timestamp()
        }
        
        # 生成执行步骤
        execution_plan['execution_steps'] = self._decompose_action(action)
        
        # 生成备用动作
        execution_plan['fallback_actions'] = self._generate_fallback_actions(action)
        
        # 设置监控点
        execution_plan['monitoring_points'] = self._set_monitoring_points(action)
        
        # 预期结果
        execution_plan['expected_outcome'] = self._predict_outcome(action, current_state)
        
        # 风险评估
        execution_plan['risk_assessment'] = self._assess_risks(action, current_state)
        
        return execution_plan
    
    def _decompose_action(self, action: ActionCommand) -> List[Dict[str, Any]]:
        """分解动作"""
        steps = []
        
        if action.action_type == ActionType.MOTOR_ACTION:
            # 分解为移动步骤
            steps = [
                {'step': 'prepare_movement', 'duration': 0.2, 'parameters': {}},
                {'step': 'execute_movement', 'duration': action.duration - 0.4, 'parameters': action.parameters},
                {'step': 'stabilize', 'duration': 0.2, 'parameters': {}}
            ]
        
        elif action.action_type == ActionType.MANIPULATION:
            # 分解为操作步骤
            steps = [
                {'step': 'approach_target', 'duration': 1.0, 'parameters': {'target': 'object'}},
                {'step': 'grasp_action', 'duration': 1.5, 'parameters': {'gripper': 'close'}},
                {'step': 'hold_position', 'duration': action.duration - 2.5, 'parameters': {}},
                {'step': 'release', 'duration': 0.5, 'parameters': {'gripper': 'open'}}
            ]
        
        elif action.action_type == ActionType.COMMUNICATION:
            # 分解为交流步骤
            steps = [
                {'step': 'prepare_communication', 'duration': 0.5, 'parameters': {}},
                {'step': 'execute_speech', 'duration': action.duration - 1.0, 'parameters': action.parameters},
                {'step': 'end_communication', 'duration': 0.5, 'parameters': {}}
            ]
        
        else:
            # 默认步骤
            steps = [
                {'step': 'start_action', 'duration': 0.1, 'parameters': action.parameters},
                {'step': 'maintain_action', 'duration': action.duration - 0.2, 'parameters': {}},
                {'step': 'end_action', 'duration': 0.1, 'parameters': {}}
            ]
        
        return steps
    
    def _generate_fallback_actions(self, action: ActionCommand) -> List[ActionCommand]:
        """生成备用动作"""
        fallback_actions = []
        
        if action.action_type == ActionType.MOTOR_ACTION:
            # 如果移动失败，改为原地等待
            fallback_actions.append(ActionCommand(
                action_type=ActionType.COGNITIVE_ACTION,
                parameters={'wait': True},
                duration=2.0,
                priority=2
            ))
        
        elif action.action_type == ActionType.MANIPULATION:
            # 如果抓取失败，改为观察
            fallback_actions.append(ActionCommand(
                action_type=ActionType.COGNITIVE_ACTION,
                parameters={'observe_environment': True},
                duration=3.0,
                priority=3
            ))
        
        return fallback_actions
    
    def _set_monitoring_points(self, action: ActionCommand) -> List[Dict[str, Any]]:
        """设置监控点"""
        monitoring_points = []
        
        # 在动作执行的25%、50%、75%设置监控点
        total_duration = action.duration
        for percentage in [0.25, 0.5, 0.75]:
            time_point = total_duration * percentage
            monitoring_points.append({
                'time': time_point,
                'checks': ['position', 'velocity', 'safety'],
                'thresholds': {
                    'position_error': 0.1,
                    'velocity_error': 0.5,
                    'safety_margin': 0.2
                }
            })
        
        return monitoring_points
    
    def _predict_outcome(self, action: ActionCommand, current_state: BodyState) -> Dict[str, Any]:
        """预测动作结果"""
        outcome = {
            'success_probability': 0.8,
            'expected_duration': action.duration,
            'resource_usage': {
                'energy': action.duration * 0.1,
                'computational': 0.2
            },
            'side_effects': []
        }
        
        # 根据动作类型预测具体结果
        if action.action_type == ActionType.MOTOR_ACTION:
            outcome['position_change'] = '预计移动1-2米'
            outcome['side_effects'] = ['消耗能量', '位置改变']
        
        elif action.action_type == ActionType.MANIPULATION:
            outcome['object_interaction'] = '预计与目标物体交互'
            outcome['side_effects'] = ['物体位移', '力的作用']
        
        return outcome
    
    def _assess_risks(self, action: ActionCommand, current_state: BodyState) -> Dict[str, Any]:
        """评估风险"""
        risk_assessment = {
            'overall_risk': 'low',
            'specific_risks': [],
            'mitigation_strategies': [],
            'risk_score': 0.2
        }
        
        specific_risks = []
        mitigation_strategies = []
        
        if action.action_type == ActionType.MOTOR_ACTION:
            risk = self._assess_movement_risk(action, current_state)
            specific_risks.extend(risk)
            mitigation_strategies.append('渐进式移动')
        
        if action.action_type == ActionType.MANIPULATION:
            risk = self._assess_manipulation_risk(action, current_state)
            specific_risks.extend(risk)
            mitigation_strategies.append('精确控制')
        
        risk_assessment['specific_risks'] = specific_risks
        risk_assessment['mitigation_strategies'] = mitigation_strategies
        
        # 计算总体风险分数
        risk_score = len(specific_risks) * 0.1
        risk_assessment['risk_score'] = min(1.0, risk_score)
        
        if risk_score > 0.7:
            risk_assessment['overall_risk'] = 'high'
        elif risk_score > 0.4:
            risk_assessment['overall_risk'] = 'medium'
        
        return risk_assessment
    
    def _assess_movement_risk(self, action: ActionCommand, current_state: BodyState) -> List[str]:
        """评估移动风险"""
        risks = []
        
        if 'velocity' in action.parameters:
            velocity = action.parameters['velocity']
            if abs(velocity) > 1.5:
                risks.append('高速移动风险')
        
        if 'distance' in action.parameters:
            distance = action.parameters['distance']
            if distance > 2.0:
                risks.append('长距离移动风险')
        
        return risks
    
    def _assess_manipulation_risk(self, action: ActionCommand, current_state: BodyState) -> List[str]:
        """评估操作风险"""
        risks = []
        
        if current_state.balance_metrics.get('balance_score', 1.0) < 0.7:
            risks.append('平衡不足风险')
        
        return risks
    
    def _get_timestamp(self) -> float:
        """获取时间戳"""
        import time
        return time.time()


class EmbodiedIntelligence:
    """
    具身智能主类
    
    整合所有交互行动功能：
    - 多模态感知
    - 运动控制
    - 动作规划
    - 环境交互
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化具身智能系统
        
        Args:
            config: 系统配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 身体模型配置
        self.body_model = config.get('body_model', 'humanoid')
        self.motor_control = config.get('motor_control', 'policy_gradient')
        self.sensory_fusion = config.get('sensory_fusion', 'kalman_filter')
        self.balance_control = config.get('balance_control', True)
        
        # 初始化组件
        self.multimodal_perception = None
        self.motor_controller = None
        self.action_planner = None
        
        # 身体状态
        self.current_body_state = self._create_initial_body_state()
        
        # 交互事件历史
        self.interaction_history = deque(maxlen=500)
        
        # 性能指标
        self.performance_metrics = {
            'actions_executed': 0,
            'successful_interactions': 0,
            'safety_violations': 0,
            'efficiency_score': 0.8
        }
        
        self.logger.info("🤖 具身智能系统初始化完成")
    
    async def initialize(self):
        """异步初始化具身智能组件"""
        self.logger.info("🔧 初始化具身智能组件...")
        
        try:
            # 初始化多模态感知
            perception_config = self.config.get('multimodal_perception', {})
            self.multimodal_perception = MultimodalPerception(perception_config)
            await self.multimodal_perception.initialize()
            
            # 初始化运动控制器
            motor_config = self.config.get('motor_control_config', {})
            self.motor_controller = MotorController(motor_config)
            
            # 初始化动作规划器
            planning_config = self.config.get('action_planning', {})
            self.action_planner = ActionPlanner(planning_config)
            
            self.logger.info("✅ 具身智能组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 具身智能组件初始化失败: {e}")
            raise
    
    def _create_initial_body_state(self) -> BodyState:
        """创建初始身体状态"""
        return BodyState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # 四元数
            joint_angles=np.random.uniform(-0.5, 0.5, 7),  # 7个关节
            joint_velocities=np.zeros(7),
            force_sensors={'left_foot': 50.0, 'right_foot': 50.0},
            balance_metrics={
                'center_of_mass': np.array([0.0, 0.0, 0.0]),
                'support_polygon_area': 0.1,
                'balance_score': 0.8,
                'angular_momentum': 0.0
            }
        )
    
    async def perceive_environment(self) -> Dict[str, Any]:
        """感知环境"""
        perception_tasks = [
            self.multimodal_perception.capture_vision(),
            self.multimodal_perception.capture_audio(),
            self.multimodal_perception.capture_touch(),
            self.multimodal_perception.capture_proprioception(self.current_body_state)
        ]
        
        # 并行执行感知任务
        vision_reading, audio_reading, touch_reading, proprioception_reading = await asyncio.gather(*perception_tasks)
        
        # 融合感知结果
        fused_perception = await self.multimodal_perception.fuse_perceptions()
        
        return {
            'individual_readings': {
                'vision': vision_reading,
                'audio': audio_reading,
                'touch': touch_reading,
                'proprioception': proprioception_reading
            },
            'fused_perception': fused_perception,
            'timestamp': self._get_timestamp()
        }
    
    async def plan_action(self, goals: List[str]) -> Dict[str, Any]:
        """规划动作"""
        # 感知环境
        perception_data = await self.perceive_environment()
        fused_perception = perception_data['fused_perception']
        
        # 规划动作
        execution_plan = await self.action_planner.plan_action(
            current_state=self.current_body_state,
            perception_fusion=fused_perception,
            goals=goals
        )
        
        return execution_plan
    
    async def execute_action(self, action_command: ActionCommand) -> Dict[str, Any]:
        """执行动作"""
        self.logger.info(f"🎬 执行动作: {action_command.action_type.value}")
        
        execution_result = {
            'success': False,
            'final_state': None,
            'performance_metrics': {},
            'interaction_events': [],
            'safety_checks': [],
            'timestamp': self._get_timestamp()
        }
        
        try:
            # 更新性能指标
            self.performance_metrics['actions_executed'] += 1
            
            # 根据动作类型执行
            if action_command.action_type == ActionType.MOTOR_ACTION:
                execution_result = await self._execute_motor_action(action_command)
            elif action_command.action_type == ActionType.MANIPULATION:
                execution_result = await self._execute_manipulation(action_command)
            elif action_command.action_type == ActionType.COMMUNICATION:
                execution_result = await self._execute_communication(action_command)
            elif action_command.action_type == ActionType.COGNITIVE_ACTION:
                execution_result = await self._execute_cognitive_action(action_command)
            elif action_command.action_type == ActionType.SOCIAL_ACTION:
                execution_result = await self._execute_social_action(action_command)
            
            # 更新身体状态
            if execution_result.get('final_state'):
                self.current_body_state = execution_result['final_state']
            
            # 记录交互事件
            if execution_result.get('interaction_events'):
                self.interaction_history.extend(execution_result['interaction_events'])
            
            # 更新成功指标
            if execution_result.get('success'):
                self.performance_metrics['successful_interactions'] += 1
            
            execution_result['timestamp'] = self._get_timestamp()
            
        except Exception as e:
            self.logger.error(f"动作执行失败: {e}")
            execution_result['error'] = str(e)
        
        return execution_result
    
    async def _execute_motor_action(self, action: ActionCommand) -> Dict[str, Any]:
        """执行电机动作"""
        # 生成目标状态
        target_state = self.current_body_state.copy()
        
        # 根据动作参数更新目标状态
        if 'move_to' in action.parameters:
            direction = action.parameters['move_to']
            distance = action.parameters.get('distance', 1.0)
            
            if direction == 'forward':
                target_state.position[1] += distance
            elif direction == 'backward':
                target_state.position[1] -= distance
            elif direction == 'left':
                target_state.position[0] -= distance
            elif direction == 'right':
                target_state.position[0] += distance
        
        elif 'rotate' in action.parameters:
            rotation = action.parameters['rotate']
            angle = action.parameters.get('angle', 0.5)
            
            # 简化旋转实现
            if rotation == 'left':
                target_state.orientation = self._rotate_orientation(
                    target_state.orientation, -angle
                )
            elif rotation == 'right':
                target_state.orientation = self._rotate_orientation(
                    target_state.orientation, angle
                )
        
        # 使用运动控制器执行
        dt = 0.01  # 时间步长
        control_output = self.motor_controller(
            current_state=self.current_body_state,
            target_state=target_state,
            dt=dt
        )
        
        # 模拟执行过程
        await asyncio.sleep(action.duration)
        
        # 更新状态
        final_state = self.current_body_state.copy()
        final_state.position = target_state.position
        final_state.orientation = target_state.orientation
        
        return {
            'success': True,
            'final_state': final_state,
            'control_output': control_output,
            'performance_metrics': {
                'trajectory_deviation': 0.1,
                'execution_time': action.duration,
                'energy_consumption': action.duration * 0.1
            },
            'interaction_events': [
                InteractionEvent(
                    event_type='movement',
                    participants=['认知主体'],
                    intensity=0.7,
                    timestamp=self._get_timestamp(),
                    outcome={'distance_moved': np.linalg.norm(target_state.position - self.current_body_state.position)}
                )
            ]
        }
    
    async def _execute_manipulation(self, action: ActionCommand) -> Dict[str, Any]:
        """执行操作动作"""
        # 模拟抓取操作
        await asyncio.sleep(action.duration)
        
        # 更新抓取器状态
        final_state = self.current_body_state.copy()
        if 'gripper' in action.parameters:
            if action.parameters['gripper'] == 'close':
                final_state.joint_angles[-1] = 1.0  # 抓取器闭合
            elif action.parameters['gripper'] == 'open':
                final_state.joint_angles[-1] = 0.0  # 抓取器张开
        
        return {
            'success': True,
            'final_state': final_state,
            'performance_metrics': {
                'manipulation_success': 0.9,
                'force_application': 5.0,
                'precision': 0.85
            },
            'interaction_events': [
                InteractionEvent(
                    event_type='manipulation',
                    participants=['认知主体', 'object'],
                    intensity=0.8,
                    timestamp=self._get_timestamp(),
                    outcome={'object_grasped': True}
                )
            ]
        }
    
    async def _execute_communication(self, action: ActionCommand) -> Dict[str, Any]:
        """执行交流动作"""
        # 模拟语音输出
        if 'speak' in action.parameters:
            text = action.parameters['speak']
            self.logger.info(f"🗣️  说话: {text}")
        
        # 模拟手势
        if action.parameters.get('gesture', False):
            await self._perform_gesture()
        
        await asyncio.sleep(action.duration)
        
        return {
            'success': True,
            'final_state': self.current_body_state,
            'performance_metrics': {
                'speech_clarity': 0.9,
                'gesture_recognizability': 0.8,
                'communication_effectiveness': 0.85
            },
            'interaction_events': [
                InteractionEvent(
                    event_type='communication',
                    participants=['认知主体', 'environment'],
                    intensity=0.6,
                    timestamp=self._get_timestamp(),
                    outcome={'message_delivered': True, 'text': action.parameters.get('speak', '')}
                )
            ]
        }
    
    async def _execute_cognitive_action(self, action: ActionCommand) -> Dict[str, Any]:
        """执行认知动作"""
        # 认知动作通常不改变身体状态
        await asyncio.sleep(action.duration)
        
        return {
            'success': True,
            'final_state': self.current_body_state,
            'performance_metrics': {
                'cognitive_load': 0.3,
                'attention_focus': 'maintained',
                'processing_time': action.duration
            },
            'interaction_events': [
                InteractionEvent(
                    event_type='cognitive_processing',
                    participants=['认知主体'],
                    intensity=0.4,
                    timestamp=self._get_timestamp(),
                    outcome={'cognitive_task_completed': True}
                )
            ]
        }
    
    async def _execute_social_action(self, action: ActionCommand) -> Dict[str, Any]:
        """执行社交动作"""
        # 模拟社交交互
        await asyncio.sleep(action.duration)
        
        return {
            'success': True,
            'final_state': self.current_body_state,
            'performance_metrics': {
                'social_engagement': 0.7,
                'empathy_response': 0.6,
                'cooperation_level': 0.8
            },
            'interaction_events': [
                InteractionEvent(
                    event_type='social_interaction',
                    participants=['认知主体', 'other_认知主体s'],
                    intensity=0.9,
                    timestamp=self._get_timestamp(),
                    outcome={'social_bond_strengthened': True}
                )
            ]
        }
    
    async def _perform_gesture(self):
        """执行手势"""
        # 模拟手势动画
        await asyncio.sleep(0.5)
    
    def _rotate_orientation(self, orientation: np.ndarray, angle: float) -> np.ndarray:
        """旋转方向（四元数）"""
        # 简化的旋转实现
        return orientation  # 实际应用中需要更复杂的四元数运算
    
    async def execute_cognitive_task(self, cognitive_state: Dict[str, Any], environment) -> Dict[str, Any]:
        """执行认知任务"""
        goals = ['complete_cognitive_task', 'maintain_focus']
        
        # 规划并执行动作
        execution_plan = await self.plan_action(goals)
        action = execution_plan['primary_action']
        
        execution_result = await self.execute_action(action)
        
        return {
            'task_success': execution_result.get('success', False),
            'performance_score': execution_result.get('performance_metrics', {}),
            'embodied_response': execution_result,
            'cognitive_integration': {
                'attention_maintained': True,
                'motor_cognitive_sync': 0.8
            }
        }
    
    async def run_interaction_loop(self, max_iterations: int = 100) -> List[Dict[str, Any]]:
        """运行交互循环"""
        interaction_log = []
        
        for iteration in range(max_iterations):
            try:
                # 感知环境
                perception = await self.perceive_environment()
                
                # 生成简单目标
                goals = [f'interaction_goal_{iteration}']
                
                # 规划动作
                execution_plan = await self.plan_action(goals)
                action = execution_plan['primary_action']
                
                # 执行动作
                execution_result = await self.execute_action(action)
                
                # 记录交互
                interaction_log.append({
                    'iteration': iteration,
                    'perception': perception,
                    'execution_plan': execution_plan,
                    'execution_result': execution_result,
                    'timestamp': self._get_timestamp()
                })
                
                # 短暂等待
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"交互循环错误 (iteration {iteration}): {e}")
                continue
        
        return interaction_log
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        # 计算成功率
        success_rate = (self.performance_metrics['successful_interactions'] / 
                       max(1, self.performance_metrics['actions_executed']))
        
        # 计算安全率
        safety_rate = 1.0 - (self.performance_metrics['safety_violations'] / 
                            max(1, self.performance_metrics['actions_executed']))
        
        return {
            **self.performance_metrics,
            'success_rate': success_rate,
            'safety_rate': safety_rate,
            'interaction_count': len(self.interaction_history),
            'current_body_state': {
                'position': self.current_body_state.position.tolist(),
                'balance_score': self.current_body_state.balance_metrics.get('balance_score', 0.8)
            }
        }
    
    def _get_timestamp(self) -> float:
        """获取时间戳"""
        import time
        return time.time()
    
    async def cleanup(self):
        """清理资源"""
        self.logger.info("🧹 清理具身智能系统资源...")
        
        # 清理交互历史
        self.interaction_history.clear()
        
        # 清理感知缓冲
        if self.multimodal_perception:
            for sensor_type in self.multimodal_perception.sensor_buffers:
                self.multimodal_perception.sensor_buffers[sensor_type].clear()
        
        self.logger.info("✅ 具身智能系统资源清理完成")