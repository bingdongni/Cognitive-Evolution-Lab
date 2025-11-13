"""
可视化系统测试模块

测试可视化界面和3D渲染功能。
"""

import unittest
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualization import (
    VisualizationSystem,
    ThreeDRenderer,
    InteractiveDashboard,
    PerformanceMonitor,
    CognitiveVisualizer
)


class TestVisualizationSystem(unittest.TestCase):
    """可视化系统基础测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.viz_system = VisualizationSystem()
        
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.viz_system, VisualizationSystem)
        self.assertIsNotNone(self.viz_system.renderer)
        self.assertIsNotNone(self.viz_system.dashboard)
        self.assertIsNotNone(self.viz_system.performance_monitor)
        
    def test_scene_rendering(self):
        """测试场景渲染"""
        # 模拟3D场景数据
        scene_data = {
            "objects": [
                {"type": "cube", "position": [0, 0, 0], "color": [1, 0, 0]},
                {"type": "sphere", "position": [1, 0, 0], "color": [0, 1, 0]}
            ],
            "camera": {"position": [2, 2, 2], "target": [0, 0, 0]},
            "lighting": {"ambient": 0.3, "directional": [0.7, 0.7, 0.7]}
        }
        
        render_result = self.viz_system.render_scene(scene_data)
        self.assertIsInstance(render_result, dict)
        self.assertIn("render_buffer", render_result)
        self.assertIn("render_time", render_result)
        self.assertIn("quality_metrics", render_result)
        
    def test_interactive_mode(self):
        """测试交互模式"""
        # 设置交互事件
        interaction_events = [
            {"type": "mouse_click", "position": [100, 150]},
            {"type": "mouse_move", "position": [110, 160]},
            {"type": "key_press", "key": "space"}
        ]
        
        interaction_result = self.viz_system.process_interactions(interaction_events)
        self.assertIsInstance(interaction_result, dict)
        self.assertIn("scene_updates", interaction_result)
        self.assertIn("user_feedback", interaction_result)


class TestThreeDRenderer(unittest.TestCase):
    """3D渲染器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.renderer = ThreeDRenderer()
        
    def test_3d_object_creation(self):
        """测试3D对象创建"""
        # 创建基本几何体
        cube = self.renderer.create_cube(size=1.0, position=[0, 0, 0])
        sphere = self.renderer.create_sphere(radius=0.5, position=[1, 0, 0])
        cylinder = self.renderer.create_cylinder(radius=0.3, height=1.0, position=[0, 1, 0])
        
        self.assertIsInstance(cube, dict)
        self.assertIn("vertices", cube)
        self.assertIn("faces", cube)
        self.assertIsInstance(sphere, dict)
        self.assertIsInstance(cylinder, dict)
        
    def test_3d_transformation(self):
        """测试3D变换"""
        # 创建测试对象
        test_object = {
            "vertices": np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
            "faces": [[0, 1, 2, 3]]
        }
        
        # 应用变换
        transformation_matrix = np.array([
            [1, 0, 0, 1],  # 平移x=1
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        transformed = self.renderer.apply_transformation(test_object, transformation_matrix)
        self.assertIsInstance(transformed, dict)
        self.assertIn("transformed_vertices", transformed)
        self.assertEqual(transformed["transformed_vertices"].shape, test_object["vertices"].shape)
        
    def test_3d_camera_control(self):
        """测试3D相机控制"""
        # 设置相机参数
        camera_params = {
            "position": [5, 5, 5],
            "target": [0, 0, 0],
            "up_vector": [0, 1, 0],
            "fov": 60
        }
        
        camera = self.renderer.setup_camera(camera_params)
        self.assertIsInstance(camera, dict)
        self.assertIn("view_matrix", camera)
        self.assertIn("projection_matrix", camera)
        
        # 测试相机移动
        movement_result = self.renderer.move_camera(camera, "orbit", angle=45, distance=6)
        self.assertIsInstance(movement_result, dict)
        self.assertIn("new_position", movement_result)
        
    def test_lighting_system(self):
        """测试光照系统"""
        # 创建光源
        light_sources = [
            {
                "type": "directional",
                "direction": [-1, -1, -1],
                "intensity": 1.0,
                "color": [1, 1, 1]
            },
            {
                "type": "point",
                "position": [2, 2, 2],
                "intensity": 0.5,
                "color": [0.8, 0.9, 1.0]
            }
        ]
        
        lighting_result = self.renderer.setup_lighting(light_sources)
        self.assertIsInstance(lighting_result, dict)
        self.assertIn("light_matrices", lighting_result)
        self.assertEqual(len(lighting_result["light_matrices"]), 2)
        
    def test_shaders(self):
        """测试着色器系统"""
        # 测试顶点着色器
        vertex_shader = """
        vec3 position;
        uniform mat4 model_view_matrix;
        uniform mat4 projection_matrix;
        void main() {
            gl_Position = projection_matrix * model_view_matrix * vec4(position, 1.0);
        }
        """
        
        fragment_shader = """
        uniform vec3 color;
        void main() {
            gl_FragColor = vec4(color, 1.0);
        }
        """
        
        shader_program = self.renderer.create_shader_program(vertex_shader, fragment_shader)
        self.assertIsInstance(shader_program, dict)
        self.assertIn("program_id", shader_program)
        self.assertIn("uniforms", shader_program)
        self.assertIn("attributes", shader_program)


class TestInteractiveDashboard(unittest.TestCase):
    """交互式仪表板测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.dashboard = InteractiveDashboard()
        
    def test_dashboard_layout(self):
        """测试仪表板布局"""
        # 定义布局配置
        layout_config = {
            "panels": [
                {"id": "cognitive_state", "position": [0, 0], "size": [0.5, 0.5]},
                {"id": "performance_metrics", "position": [0.5, 0], "size": [0.5, 0.5]},
                {"id": "evolution_progress", "position": [0, 0.5], "size": [1.0, 0.5]}
            ]
        }
        
        layout_result = self.dashboard.create_layout(layout_config)
        self.assertIsInstance(layout_result, dict)
        self.assertIn("panel_positions", layout_result)
        self.assertEqual(len(layout_result["panel_positions"]), 3)
        
    def test_real_time_updates(self):
        """测试实时更新"""
        # 模拟实时数据流
        real_time_data = [
            {"timestamp": 1234567890, "metric": "cognitive_load", "value": 0.7},
            {"timestamp": 1234567891, "metric": "performance_score", "value": 0.85},
            {"timestamp": 1234567892, "metric": "adaptation_rate", "value": 0.3}
        ]
        
        update_result = self.dashboard.update_real_time_data(real_time_data)
        self.assertIsInstance(update_result, dict)
        self.assertIn("updated_panels", update_result)
        self.assertIn("update_frequency", update_result)
        
    def test_user_interactions(self):
        """测试用户交互"""
        # 模拟用户交互事件
        user_events = [
            {"type": "button_click", "panel_id": "cognitive_state", "button_id": "reset"},
            {"type": "slider_adjust", "panel_id": "performance_metrics", "slider_id": "time_range", "value": 50},
            {"type": "dropdown_select", "panel_id": "evolution_progress", "dropdown_id": "view_mode", "value": "3d"}
        ]
        
        interaction_result = self.dashboard.handle_user_interactions(user_events)
        self.assertIsInstance(interaction_result, dict)
        self.assertIn("panel_updates", interaction_result)
        self.assertIn("visualization_changes", interaction_result)
        
    def test_data_filtering(self):
        """测试数据过滤"""
        # 模拟原始数据
        raw_data = [
            {"timestamp": 1234567890, "type": "cognitive", "value": 0.8},
            {"timestamp": 1234567891, "type": "physical", "value": 0.6},
            {"timestamp": 1234567892, "type": "cognitive", "value": 0.9},
            {"timestamp": 1234567893, "type": "social", "value": 0.7}
        ]
        
        # 应用过滤器
        filters = {
            "type": ["cognitive", "physical"],
            "timestamp_range": [1234567890, 1234567892]
        }
        
        filtered_result = self.dashboard.apply_filters(raw_data, filters)
        self.assertIsInstance(filtered_result, dict)
        self.assertIn("filtered_data", filtered_result)
        self.assertIn("filter_summary", filtered_result)
        
        # 验证过滤结果
        filtered_data = filtered_result["filtered_data"]
        self.assertEqual(len(filtered_data), 3)  # 3条数据符合条件


class TestPerformanceMonitor(unittest.TestCase):
    """性能监控器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.performance_monitor = PerformanceMonitor()
        
    def test_performance_metrics_collection(self):
        """测试性能指标收集"""
        # 收集各种性能指标
        metrics_collection = {
            "cpu_usage": 0.65,
            "memory_usage": 0.45,
            "gpu_usage": 0.78,
            "rendering_fps": 60.0,
            "processing_latency": 16.7
        }
        
        collection_result = self.performance_monitor.collect_metrics(metrics_collection)
        self.assertIsInstance(collection_result, dict)
        self.assertIn("metrics_snapshot", collection_result)
        self.assertIn("performance_health", collection_result)
        self.assertIn("resource_utilization", collection_result)
        
    def test_performance_trend_analysis(self):
        """测试性能趋势分析"""
        # 模拟历史性能数据
        historical_data = [
            {"timestamp": 1234567890, "cpu_usage": 0.5, "memory_usage": 0.4},
            {"timestamp": 1234567891, "cpu_usage": 0.6, "memory_usage": 0.45},
            {"timestamp": 1234567892, "cpu_usage": 0.7, "memory_usage": 0.5},
            {"timestamp": 1234567893, "cpu_usage": 0.65, "memory_usage": 0.48}
        ]
        
        trend_result = self.performance_monitor.analyze_trends(historical_data)
        self.assertIsInstance(trend_result, dict)
        self.assertIn("trend_analysis", trend_result)
        self.assertIn("anomaly_detection", trend_result)
        self.assertIn("predictions", trend_result)
        
    def test_performance_alerts(self):
        """测试性能警报"""
        # 设置性能阈值
        thresholds = {
            "cpu_usage": 0.8,
            "memory_usage": 0.9,
            "rendering_fps": 30.0
        }
        
        # 模拟当前性能状态
        current_performance = {
            "cpu_usage": 0.85,
            "memory_usage": 0.75,
            "rendering_fps": 45.0
        }
        
        alert_result = self.performance_monitor.check_performance_alerts(current_performance, thresholds)
        self.assertIsInstance(alert_result, dict)
        self.assertIn("active_alerts", alert_result)
        self.assertIn("alert_severity", alert_result)
        self.assertIn("recommendations", alert_result)
        
        # 验证警报触发
        self.assertGreater(len(alert_result["active_alerts"]), 0)
        
    def test_resource_optimization(self):
        """测试资源优化"""
        # 分析资源使用模式
        resource_usage_patterns = {
            "peak_hours": [9, 10, 11, 14, 15, 16],
            "bottlenecks": ["gpu_memory", "cpu_cores"],
            "optimization_potential": 0.3
        }
        
        optimization_result = self.performance_monitor.suggest_optimization(resource_usage_patterns)
        self.assertIsInstance(optimization_result, dict)
        self.assertIn("optimization_suggestions", optimization_result)
        self.assertIn("resource_allocation", optimization_result)
        self.assertIn("expected_improvements", optimization_result)


class TestCognitiveVisualizer(unittest.TestCase):
    """认知可视化器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.cognitive_viz = CognitiveVisualizer()
        
    def test_cognitive_state_visualization(self):
        """测试认知状态可视化"""
        # 模拟认知状态数据
        cognitive_state = {
            "memory_capacity": 0.7,
            "attention_focus": 0.8,
            "processing_speed": 0.6,
            "learning_rate": 0.5,
            "creative_potential": 0.9,
            "logical_reasoning": 0.75
        }
        
        visualization_result = self.cognitive_viz.visualize_cognitive_state(cognitive_state)
        self.assertIsInstance(visualization_result, dict)
        self.assertIn("visualization_type", visualization_result)
        self.assertIn("cognitive_map", visualization_result)
        self.assertIn("performance_indicators", visualization_result)
        
    def test_neural_network_visualization(self):
        """测试神经网络可视化"""
        # 模拟神经网络结构
        network_structure = {
            "layers": [
                {"type": "input", "nodes": 784, "activation": "relu"},
                {"type": "hidden", "nodes": 128, "activation": "relu"},
                {"type": "hidden", "nodes": 64, "activation": "relu"},
                {"type": "output", "nodes": 10, "activation": "softmax"}
            ],
            "connections": "dense",
            "weights_range": [-1.0, 1.0]
        }
        
        network_viz = self.cognitive_viz.visualize_neural_network(network_structure)
        self.assertIsInstance(network_viz, dict)
        self.assertIn("network_graph", network_viz)
        self.assertIn("layer_visualizations", network_viz)
        self.assertIn("connection_weights", network_viz)
        
    def test_learning_progress_visualization(self):
        """测试学习进度可视化"""
        # 模拟学习数据
        learning_data = {
            "epochs": list(range(1, 101)),
            "accuracy": [0.3 + i*0.007 + np.random.normal(0, 0.01) for i in range(100)],
            "loss": [1.0 - i*0.01 + np.random.normal(0, 0.05) for i in range(100)],
            "validation_accuracy": [0.25 + i*0.006 + np.random.normal(0, 0.015) for i in range(100)]
        }
        
        progress_viz = self.cognitive_viz.visualize_learning_progress(learning_data)
        self.assertIsInstance(progress_viz, dict)
        self.assertIn("learning_curves", progress_viz)
        self.assertIn("performance_summary", progress_viz)
        self.assertIn("convergence_analysis", progress_viz)
        
    def test_decision_tree_visualization(self):
        """测试决策树可视化"""
        # 模拟决策树结构
        decision_tree = {
            "root": {
                "feature": "feature_1",
                "threshold": 0.5,
                "left_child": {
                    "feature": "feature_2",
                    "threshold": 0.3,
                    "left_child": {"class": "A"},
                    "right_child": {"class": "B"}
                },
                "right_child": {
                    "feature": "feature_3",
                    "threshold": 0.7,
                    "left_child": {"class": "B"},
                    "right_child": {"class": "C"}
                }
            }
        }
        
        tree_viz = self.cognitive_viz.visualize_decision_tree(decision_tree)
        self.assertIsInstance(tree_viz, dict)
        self.assertIn("tree_structure", tree_viz)
        self.assertIn("decision_paths", tree_viz)
        self.assertIn("feature_importance", tree_viz)
        
    def test_evolution_process_visualization(self):
        """测试进化过程可视化"""
        # 模拟进化数据
        evolution_data = {
            "generations": list(range(1, 51)),
            "best_fitness": [0.1 + i*0.02 + np.random.normal(0, 0.005) for i in range(50)],
            "average_fitness": [0.05 + i*0.015 + np.random.normal(0, 0.008) for i in range(50)],
            "population_diversity": [0.8 - i*0.01 + np.random.normal(0, 0.02) for i in range(50)]
        }
        
        evolution_viz = self.cognitive_viz.visualize_evolution_process(evolution_data)
        self.assertIsInstance(evolution_viz, dict)
        self.assertIn("fitness_landscape", evolution_viz)
        self.assertIn("diversity_evolution", evolution_viz)
        self.assertIn("evolution_trajectory", evolution_viz)
        self.assertIn("convergence_metrics", evolution_viz)


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)
