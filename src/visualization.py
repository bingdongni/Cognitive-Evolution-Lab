#!/usr/bin/env python3
"""
Cognitive Evolution Lab - 可视化界面
作者: bingdongni

实现可视化界面，包括：
- 3D世界渲染（pygame/pyglet）
- 数据仪表板（Dash/Plotly）
- 实时监控图表
- 交互式分析工具
- 实时性能指标
"""

import asyncio
import numpy as np
import torch
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import deque, defaultdict

# 尝试导入可视化库
try:
    import pygame
    PYGAME_AV认知计算LABLE = True
except ImportError:
    PYGAME_AV认知计算LABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AV认知计算LABLE = True
except ImportError:
    PLOTLY_AV认知计算LABLE = False

try:
    import dash
    from dash import dcc, html, Input, Output, State
    DASH_AV认知计算LABLE = True
except ImportError:
    DASH_AV认知计算LABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AV认知计算LABLE = True
except ImportError:
    PIL_AV认知计算LABLE = False


class VisualizationType(Enum):
    """可视化类型枚举"""
    REAL_TIME_3D = "real_time_3d"
    DASHBOARD = "dashboard"
    PERFORMANCE_CHARTS = "performance_charts"
    BR认知计算N_ACTIVITY = "brain_activity"
    EVOLUTION_TREE = "evolution_tree"
    SOCIAL_NETWORK = "social_network"
    COGNITIVE_METRICS = "cognitive_metrics"


@dataclass
class VisualizationConfig:
    """可视化配置"""
    type: VisualizationType
    width: int = 1920
    height: int = 1080
    fps: int = 60
    refresh_rate: float = 1.0
    interactive: bool = True
    real_time: bool = True
    save_frames: bool = False


@dataclass
class RenderObject:
    """渲染对象"""
    id: str
    position: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray
    color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    visibility: bool = True
    animation_data: Dict[str, Any] = field(default_factory=dict)


class RealTimeRenderer:
    """实时3D渲染器"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 渲染状态
        self.is_running = False
        self.screen = None
        self.clock = None
        self.camera = {
            'position': np.array([0, -200, 100]),
            'target': np.array([0, 0, 0]),
            'up': np.array([0, 0, 1]),
            'fov': 60.0
        }
        
        # 渲染对象
        self.render_objects = {}
        self.world_bounds = np.array([[-100, 100], [-100, 100], [0, 50]])
        
        # 视觉效果
        self.particles = []
        self.trails = defaultdict(list)
        
        # 性能统计
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = 0
        
        self.logger.info("🎨 实时渲染器初始化完成")
    
    async def initialize(self):
        """初始化渲染器"""
        if not PYGAME_AV认知计算LABLE:
            self.logger.warning("⚠️ Pygame不可用，使用简化渲染")
            await self._initialize_simple_renderer()
            return
        
        try:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.config.width, self.config.height)
            )
            pygame.display.set_caption("Cognitive Evolution Lab - 3D World")
            
            self.clock = pygame.time.Clock()
            
            # 设置字体
            try:
                self.font = pygame.font.Font(None, 36)
                self.small_font = pygame.font.Font(None, 24)
            except:
                self.font = None
                self.small_font = None
            
            self.logger.info("✅ Pygame渲染器初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 渲染器初始化失败: {e}")
            await self._initialize_simple_renderer()
    
    async def _initialize_simple_renderer(self):
        """初始化简化渲染器"""
        self.screen = None  # 不使用真实的显示
        self.font = None
        self.logger.info("✅ 简化渲染器初始化完成")
    
    async def add_render_object(self, obj: RenderObject):
        """添加渲染对象"""
        self.render_objects[obj.id] = obj
        self.logger.debug(f"添加渲染对象: {obj.id}")
    
    async def remove_render_object(self, obj_id: str):
        """移除渲染对象"""
        if obj_id in self.render_objects:
            del self.render_objects[obj_id]
            self.logger.debug(f"移除渲染对象: {obj_id}")
    
    async def update_camera(self, target_position: np.ndarray = None, 
                          target_rotation: np.ndarray = None):
        """更新相机"""
        if target_position is not None:
            self.camera['target'] = target_position
        
        if target_rotation is not None:
            # 简化的相机旋转
            pass
    
    async def render_frame(self, world_state: Any, cognitive_state: Any = None,
                         evolution_state: Any = None):
        """渲染一帧"""
        if not self.is_running or not self.screen:
            return
        
        # 清空屏幕
        self.screen.fill((0, 0, 0))  # 黑色背景
        
        # 更新相机
        await self._update_camera_from_state(world_state)
        
        # 渲染世界对象
        await self._render_world_objects()
        
        # 渲染认知主体
        await self._render_认知主体s(world_state)
        
        # 渲染粒子效果
        await self._render_particles()
        
        # 渲染大脑活动（如果可用）
        if cognitive_state:
            await self._render_brain_activity(cognitive_state)
        
        # 渲染进化信息
        if evolution_state:
            await self._render_evolution_info(evolution_state)
        
        # 渲染UI叠加
        await self._render_ui_overlay(world_state, cognitive_state, evolution_state)
        
        # 更新显示
        pygame.display.flip()
        
        # 帧率控制
        self.clock.tick(self.config.fps)
        self.frame_count += 1
        self.fps_counter += 1
        
        # FPS计算
        current_time = pygame.time.get_ticks() / 1000.0
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    async def _update_camera_from_state(self, world_state):
        """从世界状态更新相机"""
        if hasattr(world_state, '认知主体s') and world_state.认知主体s:
            # 计算认知主体中心位置
            total_pos = np.zeros(3)
            for 认知主体 in world_state.认知主体s:
                if 'position' in 认知主体:
                    pos = np.array(认知主体['position'])
                    if len(pos) >= 2:
                        pos = np.append(pos, 0)  # 添加Z坐标
                    total_pos += pos
            
            center = total_pos / len(world_state.认知主体s)
            
            # 相机跟随中心
            self.camera['target'] = center
            self.camera['position'] = center + np.array([0, -200, 100])
    
    async def _render_world_objects(self):
        """渲染世界对象"""
        # 渲染地面网格
        await self._render_ground_grid()
        
        # 渲染其他对象
        for obj_id, obj in self.render_objects.items():
            if obj.visibility:
                await self._render_single_object(obj)
    
    async def _render_ground_grid(self):
        """渲染地面网格"""
        if not self.screen:
            return
        
        # 简化的网格渲染
        grid_color = (50, 50, 50)
        grid_size = 20
        grid_extent = 100
        
        for x in range(-grid_extent, grid_extent + 1, grid_size):
            # 垂直线
            start_pos = (x, -grid_extent)
            end_pos = (x, grid_extent)
            # pygame.draw.line(self.screen, grid_color, start_pos, end_pos, 1)
        
        for y in range(-grid_extent, grid_extent + 1, grid_size):
            # 水平线
            start_pos = (-grid_extent, y)
            end_pos = (grid_extent, y)
            # pygame.draw.line(self.screen, grid_color, start_pos, end_pos, 1)
    
    async def _render_single_object(self, obj: RenderObject):
        """渲染单个对象"""
        if not self.screen:
            return
        
        # 世界坐标到屏幕坐标转换
        screen_pos = self._world_to_screen(obj.position)
        
        # 根据对象类型渲染
        obj_type = obj.animation_data.get('type', 'cube')
        
        if obj_type == '认知主体':
            await self._render_认知主体(screen_pos, obj.color, obj.animation_data)
        elif obj_type == 'particle':
            await self._render_particle(screen_pos, obj.color)
        else:
            await self._render_generic_object(screen_pos, obj.color, obj.animation_data)
    
    def _world_to_screen(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """世界坐标转屏幕坐标"""
        # 简化的坐标转换
        screen_x = int(world_pos[0] + self.config.width // 2)
        screen_y = int(world_pos[1] + self.config.height // 2)
        
        return screen_x, screen_y
    
    async def _render_认知主体(self, screen_pos: Tuple[int, int], 
                          color: Tuple[int, int, int, int], 
                          认知主体_data: Dict[str, Any]):
        """渲染认知主体"""
        if not self.screen:
            return
        
        radius = 5
        
        # 绘制认知主体圆圈
        pygame.draw.circle(self.screen, color[:3], screen_pos, radius)
        
        # 绘制方向指示
        if 'velocity' in 认知主体_data:
            velocity = 认知主体_data['velocity']
            if len(velocity) >= 2:
                end_pos = (screen_pos[0] + int(velocity[0] * 10), 
                          screen_pos[1] + int(velocity[1] * 10))
                pygame.draw.line(self.screen, (255, 255, 0), screen_pos, end_pos, 2)
        
        # 绘制影响力范围
        if 'influence_radius' in 认知主体_data:
            radius = int(认知主体_data['influence_radius'] / 10)
            pygame.draw.circle(self.screen, color[:3], screen_pos, radius, 1)
    
    async def _render_particle(self, screen_pos: Tuple[int, int], 
                             color: Tuple[int, int, int, int]):
        """渲染粒子"""
        if not self.screen:
            return
        
        radius = 2
        pygame.draw.circle(self.screen, color[:3], screen_pos, radius)
    
    async def _render_generic_object(self, screen_pos: Tuple[int, int], 
                                   color: Tuple[int, int, int, int], 
                                   obj_data: Dict[str, Any]):
        """渲染通用对象"""
        if not self.screen:
            return
        
        size = obj_data.get('size', 10)
        
        # 简化的矩形渲染
        rect = pygame.Rect(
            screen_pos[0] - size // 2,
            screen_pos[1] - size // 2,
            size, size
        )
        
        pygame.draw.rect(self.screen, color[:3], rect)
    
    async def _render_认知主体s(self, world_state):
        """渲染认知主体"""
        if not hasattr(world_state, '认知主体s'):
            return
        
        for 认知主体 in world_state.认知主体s:
            if 'position' in 认知主体:
                认知主体_data = {
                    'type': '认知主体',
                    'velocity': 认知主体.get('velocity', [0, 0]),
                    'influence_radius': 认知主体.get('influence_radius', 20),
                    'strategy': 认知主体.get('strategy', 'neutral'),
                    'resource_level': 认知主体.get('resource_level', 0.5)
                }
                
                # 根据策略设置颜色
                strategy = 认知主体.get('strategy', 'neutral')
                color_map = {
                    'cooperative': (0, 255, 0),
                    'competitive': (255, 0, 0),
                    'neutral': (128, 128, 128)
                }
                
                color = color_map.get(strategy, (255, 255, 255))
                
                render_obj = RenderObject(
                    id=f"认知主体_{认知主体['id']}",
                    position=np.array(认知主体['position']),
                    rotation=np.array([0, 0, 0]),
                    scale=np.array([1, 1, 1]),
                    color=(*color, 255),
                    animation_data=认知主体_data
                )
                
                await self.render_object(render_obj)
    
    async def render_object(self, obj: RenderObject):
        """渲染对象"""
        await self.add_render_object(obj)
    
    async def _render_particles(self):
        """渲染粒子系统"""
        # 简化粒子渲染
        pass
    
    async def _render_brain_activity(self, cognitive_state):
        """渲染大脑活动"""
        if not self.screen or not cognitive_state:
            return
        
        # 在屏幕角落显示认知指标
        brain_text = f"认知负荷: {cognitive_state.get('cognitive_load', 0):.2f}"
        attention_text = f"注意力焦点: {cognitive_state.get('attention_focus', 'unknown')}"
        
        if self.font:
            brain_surface = self.font.render(brain_text, True, (255, 255, 255))
            attention_surface = self.font.render(attention_text, True, (255, 255, 255))
            
            self.screen.blit(brain_surface, (10, self.config.height - 80))
            self.screen.blit(attention_surface, (10, self.config.height - 40))
    
    async def _render_evolution_info(self, evolution_state):
        """渲染进化信息"""
        if not self.screen or not evolution_state:
            return
        
        # 显示进化统计
        generation = evolution_state.get('generation', 0)
        best_fitness = evolution_state.get('best_fitness', 0)
        diversity = evolution_state.get('diversity_score', 0)
        
        info_texts = [
            f"代数: {generation}",
            f"最佳适应度: {best_fitness:.3f}",
            f"多样性: {diversity:.3f}"
        ]
        
        if self.font:
            for i, text in enumerate(info_texts):
                surface = self.font.render(text, True, (0, 255, 255))
                self.screen.blit(surface, (self.config.width - 300, 10 + i * 40))
    
    async def _render_ui_overlay(self, world_state, cognitive_state, evolution_state):
        """渲染UI叠加"""
        if not self.screen:
            return
        
        # FPS显示
        fps_text = f"FPS: {self.clock.get_fps():.1f}" if self.clock else "FPS: N/A"
        
        if self.small_font:
            fps_surface = self.small_font.render(fps_text, True, (255, 255, 0))
            self.screen.blit(fps_surface, (10, 10))
        
        # 帧数计数
        frame_text = f"Frame: {self.frame_count}"
        if self.small_font:
            frame_surface = self.small_font.render(frame_text, True, (255, 255, 0))
            self.screen.blit(frame_surface, (10, 30))
        
        # 世界状态信息
        if hasattr(world_state, 'metrics'):
            metrics = world_state.metrics
            metrics_text = f"交互: {metrics.get('social_interactions', 0)}"
            if self.small_font:
                metrics_surface = self.small_font.render(metrics_text, True, (255, 255, 0))
                self.screen.blit(metrics_surface, (10, 50))
    
    async def start_rendering(self):
        """开始渲染"""
        self.is_running = True
        self.logger.info("🎬 开始渲染")
    
    async def stop_rendering(self):
        """停止渲染"""
        self.is_running = False
        self.logger.info("⏹️ 停止渲染")
    
    async def cleanup(self):
        """清理渲染器"""
        self.logger.info("🧹 清理渲染器...")
        
        if PYGAME_AV认知计算LABLE and pygame:
            pygame.quit()
        
        self.render_objects.clear()
        self.logger.info("✅ 渲染器清理完成")


class InteractiveDashboard:
    """交互式仪表板"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Dash应用
        self.app = None
        self.server = None
        
        # 数据存储
        self.real_time_data = {
            'cognitive_metrics': deque(maxlen=100),
            'evolution_metrics': deque(maxlen=100),
            'performance_metrics': deque(maxlen=100),
            'interaction_events': deque(maxlen=50)
        }
        
        # 图表配置
        self.chart_configs = {
            'cognitive_evolution': {
                'type': 'line',
                'x_axis': 'generation',
                'y_axis': 'cognitive_score',
                'title': '认知能力演化'
            },
            'fitness_landscape': {
                'type': 'scatter',
                'x_axis': 'generation',
                'y_axis': 'fitness',
                'color': 'diversity',
                'title': '适应度景观'
            },
            'social_network': {
                'type': 'network',
                'layout': 'spring',
                'title': '社交网络'
            },
            'brain_activity': {
                'type': 'heatmap',
                'title': '大脑活动热图'
            }
        }
        
        self.logger.info("📊 交互式仪表板初始化完成")
    
    async def initialize(self):
        """初始化仪表板"""
        if not DASH_AV认知计算LABLE:
            self.logger.warning("⚠️ Dash不可用，使用简化仪表板")
            await self._initialize_simple_dashboard()
            return
        
        try:
            self.app = dash.Dash(__name__)
            await self._setup_layout()
            await self._setup_callbacks()
            
            self.logger.info("✅ Dash仪表板初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 仪表板初始化失败: {e}")
            await self._initialize_simple_dashboard()
    
    async def _initialize_simple_dashboard(self):
        """初始化简化仪表板"""
        self.app = None
        self.logger.info("✅ 简化仪表板初始化完成")
    
    async def _setup_layout(self):
        """设置布局"""
        if not self.app:
            return
        
        self.app.layout = html.Div([
            # 标题
            html.H1("Cognitive Evolution Lab - 控制面板", 
                   style={'text-align': 'center', 'color': '#2c3e50'}),
            
            # 实时数据选择器
            html.Div([
                html.H3("实时监控"),
                dcc.Dropdown(
                    id='metric-selector',
                    options=[
                        {'label': '认知指标', 'value': 'cognitive'},
                        {'label': '进化指标', 'value': 'evolution'},
                        {'label': '性能指标', 'value': 'performance'},
                        {'label': '交互事件', 'value': 'interactions'}
                    ],
                    value='cognitive'
                ),
                dcc.Interval(
                    id='interval-component',
                    interval=1000,  # 每秒更新
                    n_intervals=0
                )
            ], style={'width': '100%', 'margin-bottom': '20px'}),
            
            # 主要图表区域
            html.Div([
                # 左上：认知能力演化
                html.Div([
                    dcc.Graph(id='cognitive-evolution-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                # 右上：适应度景观
                html.Div([
                    dcc.Graph(id='fitness-landscape-chart')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                # 左下：社交网络
                html.Div([
                    dcc.Graph(id='social-network-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                # 右下：大脑活动
                html.Div([
                    dcc.Graph(id='brain-activity-chart')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # 状态显示
            html.Div(id='status-display', 
                    style={'margin-top': '20px', 'padding': '10px', 
                          'background-color': '#f0f0f0', 'border-radius': '5px'})
        ])
    
    async def _setup_callbacks(self):
        """设置回调函数"""
        if not self.app:
            return
        
        @self.app.callback(
            Output('cognitive-evolution-chart', 'figure'),
            Output('fitness-landscape-chart', 'figure'),
            Output('social-network-chart', 'figure'),
            Output('brain-activity-chart', 'figure'),
            Output('status-display', 'children'),
            Input('interval-component', 'n_intervals'),
            Input('metric-selector', 'value')
        )
        async def update_charts(n, selected_metric):
            # 生成图表
            cognitive_fig = await self._generate_cognitive_evolution_chart()
            fitness_fig = await self._generate_fitness_landscape_chart()
            social_fig = await self._generate_social_network_chart()
            brain_fig = await self._generate_brain_activity_chart()
            
            # 状态信息
            status = f"当前选择: {selected_metric}, 实时数据点数: {len(self.real_time_data[selected_metric + '_metrics'])}"
            
            return cognitive_fig, fitness_fig, social_fig, brain_fig, status
    
    async def _generate_cognitive_evolution_chart(self) -> Dict[str, Any]:
        """生成认知演化图表"""
        if not PLOTLY_AV认知计算LABLE:
            return {}
        
        # 生成示例数据
        generations = list(range(100))
        memory_score = np.random.normal(0.5, 0.1, 100).cumsum() * 0.01 + 0.7
        reasoning_score = np.random.normal(0.5, 0.1, 100).cumsum() * 0.01 + 0.6
        creativity_score = np.random.normal(0.5, 0.1, 100).cumsum() * 0.01 + 0.5
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=generations,
            y=memory_score,
            mode='lines',
            name='记忆能力',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=generations,
            y=reasoning_score,
            mode='lines',
            name='推理能力',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=generations,
            y=creativity_score,
            mode='lines',
            name='创造力',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='认知能力演化',
            xaxis_title='代数',
            yaxis_title='能力分数',
            showlegend=True
        )
        
        return fig.to_dict()
    
    async def _generate_fitness_landscape_chart(self) -> Dict[str, Any]:
        """生成适应度景观图表"""
        if not PLOTLY_AV认知计算LABLE:
            return {}
        
        # 生成适应度数据
        generations = list(range(100))
        best_fitness = np.random.normal(0.8, 0.05, 100).cumsum() * 0.01 + 0.8
        avg_fitness = np.random.normal(0.6, 0.05, 100).cumsum() * 0.01 + 0.6
        diversity = np.random.uniform(0.3, 0.8, 100)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=generations,
            y=best_fitness,
            mode='lines',
            name='最佳适应度',
            line=dict(color='gold', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=generations,
            y=avg_fitness,
            mode='lines',
            name='平均适应度',
            line=dict(color='lightblue')
        ))
        
        fig.update_layout(
            title='适应度演化',
            xaxis_title='代数',
            yaxis_title='适应度',
            showlegend=True
        )
        
        return fig.to_dict()
    
    async def _generate_social_network_chart(self) -> Dict[str, Any]:
        """生成社交网络图表"""
        if not PLOTLY_AV认知计算LABLE:
            return {}
        
        # 生成网络数据
        n_nodes = 30
        node_x = np.random.uniform(-1, 1, n_nodes)
        node_y = np.random.uniform(-1, 1, n_nodes)
        
        # 生成边
        edge_x = []
        edge_y = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.random() < 0.1:  # 10%的连接概率
                    edge_x.extend([node_x[i], node_x[j], None])
                    edge_y.extend([node_y[i], node_y[j], None])
        
        fig = go.Figure()
        
        # 绘制边
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='lightgray'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # 绘制节点
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(
                size=10,
                color=np.random.uniform(0, 1, n_nodes),
                colorscale='Viridis',
                showscale=True
            ),
            text=[f'Agent {i}' for i in range(n_nodes)],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title='社交网络结构',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40)
        )
        
        return fig.to_dict()
    
    async def _generate_brain_activity_chart(self) -> Dict[str, Any]:
        """生成大脑活动图表"""
        if not PLOTLY_AV认知计算LABLE:
            return {}
        
        # 生成大脑活动数据
        brain_regions = ['前额叶', '顶叶', '颞叶', '枕叶', '小脑', '脑干']
        time_points = list(range(50))
        
        # 模拟脑电活动
        activity_data = []
        for region in brain_regions:
            region_activity = np.random.normal(0.5, 0.2, 50) + np.sin(np.array(time_points) * 0.1) * 0.3
            activity_data.append(region_activity)
        
        fig = go.Figure(data=go.Heatmap(
            z=np.array(activity_data),
            x=time_points,
            y=brain_regions,
            colorscale='RdYlBu_r'
        ))
        
        fig.update_layout(
            title='大脑活动热图',
            xaxis_title='时间点',
            yaxis_title='脑区',
            height=400
        )
        
        return fig.to_dict()
    
    async def update_cognitive_results(self, results: Dict[str, Any]):
        """更新认知结果"""
        if 'cognitive_state' in results:
            self.real_time_data['cognitive_metrics'].append({
                'timestamp': results.get('timestamp', 0),
                'attention_focus': results['cognitive_state'].get('attention_focus', 'unknown'),
                'cognitive_load': results['cognitive_state'].get('cognitive_load', 0),
                'overall_score': results.get('overall_score', 0)
            })
    
    async def update_evolution_results(self, results: Dict[str, Any]):
        """更新进化结果"""
        self.real_time_data['evolution_metrics'].append({
            'generation': results.get('evolution_data', {}).get('generation', 0),
            'best_fitness': results.get('evolutionary_fitness', 0),
            'population_size': results.get('evolution_data', {}).get('population_size', 0),
            'diversity_score': results.get('population_diversity', 0)
        })
    
    async def update_performance_metrics(self, metrics: Dict[str, Any]):
        """更新性能指标"""
        self.real_time_data['performance_metrics'].append({
            'timestamp': metrics.get('timestamp', 0),
            'fps': metrics.get('fps', 0),
            'memory_usage': metrics.get('memory_usage', 0),
            'cpu_usage': metrics.get('cpu_usage', 0)
        })
    
    async def start_server(self, port: int = 8050):
        """启动服务器"""
        if not self.app:
            self.logger.warning("⚠️ 仪表板未初始化")
            return
        
        self.logger.info(f"📊 启动仪表板服务器，端口: {port}")
        self.app.run_server(debug=False, port=port, host='0.0.0.0')
    
    async def cleanup(self):
        """清理仪表板"""
        self.logger.info("🧹 清理仪表板...")
        
        self.real_time_data.clear()
        
        if self.app:
            # Dash应用清理
            pass
        
        self.logger.info("✅ 仪表板清理完成")


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 性能数据
        self.performance_history = deque(maxlen=1000)
        self.monitoring_enabled = True
        
        # 监控指标
        self.metrics = {
            'fps': 0.0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0,
            'gpu_usage': 0.0,
            'frame_time': 0.0,
            'render_time': 0.0,
            'update_time': 0.0
        }
        
        # 警告阈值
        self.thresholds = {
            'fps_min': 30.0,
            'memory_max': 80.0,
            'cpu_max': 90.0,
            'gpu_max': 90.0,
            'frame_time_max': 0.05
        }
        
        self.logger.info("⚡ 性能监控器初始化完成")
    
    async def start_monitoring(self):
        """开始监控"""
        self.monitoring_enabled = True
        self.logger.info("⚡ 开始性能监控")
        
        # 启动监控循环
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """停止监控"""
        self.monitoring_enabled = False
        self.logger.info("⏹️ 停止性能监控")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_enabled:
            await self._update_metrics()
            await self._check_thresholds()
            await asyncio.sleep(1.0)  # 每秒更新一次
    
    async def _update_metrics(self):
        """更新指标"""
        import psutil
        
        # CPU使用率
        self.metrics['cpu_usage'] = psutil.cpu_percent(interval=0.1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        self.metrics['memory_usage'] = memory.percent
        
        # GPU使用率（如果可用）
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                self.metrics['gpu_usage'] = gpus[0].load * 100
        except:
            self.metrics['gpu_usage'] = 0.0
        
        # 记录性能历史
        self.performance_history.append({
            'timestamp': self._get_timestamp(),
            'metrics': self.metrics.copy()
        })
    
    async def _check_thresholds(self):
        """检查阈值"""
        warnings = []
        
        # FPS检查
        if self.metrics['fps'] < self.thresholds['fps_min']:
            warnings.append(f"FPS过低: {self.metrics['fps']:.1f}")
        
        # 内存检查
        if self.metrics['memory_usage'] > self.thresholds['memory_max']:
            warnings.append(f"内存使用过高: {self.metrics['memory_usage']:.1f}%")
        
        # CPU检查
        if self.metrics['cpu_usage'] > self.thresholds['cpu_max']:
            warnings.append(f"CPU使用过高: {self.metrics['cpu_usage']:.1f}%")
        
        # GPU检查
        if self.metrics['gpu_usage'] > self.thresholds['gpu_max']:
            warnings.append(f"GPU使用过高: {self.metrics['gpu_usage']:.1f}%")
        
        # 帧时间检查
        if self.metrics['frame_time'] > self.thresholds['frame_time_max']:
            warnings.append(f"帧时间过长: {self.metrics['frame_time']:.3f}s")
        
        # 记录警告
        if warnings:
            for warning in warnings:
                self.logger.warning(f"⚠️ {warning}")
    
    def update_frame_time(self, frame_time: float):
        """更新帧时间"""
        self.metrics['frame_time'] = frame_time
        self.metrics['fps'] = 1.0 / frame_time if frame_time > 0 else 0.0
    
    def update_render_time(self, render_time: float):
        """更新渲染时间"""
        self.metrics['render_time'] = render_time
    
    def update_update_time(self, update_time: float):
        """更新时间"""
        self.metrics['update_time'] = update_time
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-10:]  # 最近10个数据点
        
        summary = {
            'current_metrics': self.metrics.copy(),
            'averages': {},
            'trends': {},
            'recommendations': []
        }
        
        # 计算平均值
        for metric_name in self.metrics:
            values = [point['metrics'][metric_name] for point in recent_metrics]
            summary['averages'][metric_name] = np.mean(values)
        
        # 计算趋势
        if len(recent_metrics) >= 2:
            for metric_name in self.metrics:
                recent_values = [point['metrics'][metric_name] for point in recent_metrics[-5:]]
                older_values = [point['metrics'][metric_name] for point in recent_metrics[:5]]
                
                if len(recent_values) > 0 and len(older_values) > 0:
                    trend = np.mean(recent_values) - np.mean(older_values)
                    summary['trends'][metric_name] = trend
        
        # 生成建议
        if self.metrics['fps'] < 30:
            summary['recommendations'].append("降低渲染质量以提高帧率")
        
        if self.metrics['memory_usage'] > 80:
            summary['recommendations'].append("考虑减少种群大小或内存使用")
        
        return summary
    
    def _get_timestamp(self) -> float:
        """获取时间戳"""
        import time
        return time.time()
    
    async def cleanup(self):
        """清理监控器"""
        self.logger.info("🧹 清理性能监控器...")
        
        await self.stop_monitoring()
        self.performance_history.clear()
        
        self.logger.info("✅ 性能监控器清理完成")


class LabDashboard:
    """
    实验室仪表板主类
    
    整合所有可视化功能：
    - 3D实时渲染
    - 交互式仪表板
    - 性能监控
    - 数据分析
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化实验室仪表板
        
        Args:
            config: 可视化配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化各个组件
        self.real_time_renderer = None
        self.interactive_dashboard = None
        self.performance_monitor = None
        
        # 可视化状态
        self.is_initialized = False
        self.is_running = False
        
        # 数据同步
        self.data_sync_interval = 1.0
        
        self.logger.info("📺 实验室仪表板初始化完成")
    
    async def initialize(self):
        """异步初始化仪表板组件"""
        self.logger.info("🔧 初始化仪表板组件...")
        
        try:
            # 初始化3D渲染器
            renderer_config = VisualizationConfig(
                type=VisualizationType.REAL_TIME_3D,
                width=self.config.get('render_3d', {}).get('resolution', [1920, 1080])[0],
                height=self.config.get('render_3d', {}).get('resolution', [1920, 1080])[1],
                fps=self.config.get('render_3d', {}).get('fps', 60)
            )
            
            self.real_time_renderer = RealTimeRenderer(renderer_config)
            await self.real_time_renderer.initialize()
            
            # 初始化交互式仪表板
            dashboard_config = VisualizationConfig(
                type=VisualizationType.DASHBOARD,
                refresh_rate=self.config.get('dashboard', {}).get('refresh_rate', 1.0)
            )
            
            self.interactive_dashboard = InteractiveDashboard(dashboard_config)
            await self.interactive_dashboard.initialize()
            
            # 初始化性能监控器
            monitor_config = VisualizationConfig(
                type=VisualizationType.PERFORMANCE_CHARTS
            )
            
            self.performance_monitor = PerformanceMonitor(monitor_config)
            
            self.is_initialized = True
            self.logger.info("✅ 仪表板组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 仪表板组件初始化失败: {e}")
            raise
    
    async def start_rendering(self):
        """开始渲染"""
        if not self.is_initialized:
            await self.initialize()
        
        if self.real_time_renderer:
            await self.real_time_renderer.start_rendering()
        
        if self.performance_monitor:
            await self.performance_monitor.start_monitoring()
        
        self.is_running = True
        self.logger.info("🎬 仪表板渲染开始")
    
    async def stop_rendering(self):
        """停止渲染"""
        if self.real_time_renderer:
            await self.real_time_renderer.stop_rendering()
        
        if self.performance_monitor:
            await self.performance_monitor.stop_monitoring()
        
        self.is_running = False
        self.logger.info("⏹️ 仪表板渲染停止")
    
    async def start_server(self, port: int = 8050):
        """启动仪表板服务器"""
        if self.interactive_dashboard:
            await self.interactive_dashboard.start_server(port)
    
    async def render_frame(self, world_state=None, cognitive_state=None, evolution_state=None):
        """渲染一帧"""
        if not self.is_running or not self.real_time_renderer:
            return
        
        # 更新性能监控
        import time
        start_time = time.time()
        
        # 渲染世界
        await self.real_time_renderer.render_frame(world_state, cognitive_state, evolution_state)
        
        # 更新性能指标
        frame_time = time.time() - start_time
        if self.performance_monitor:
            self.performance_monitor.update_frame_time(frame_time)
            self.performance_monitor.update_render_time(frame_time)
    
    async def update_cognitive_results(self, results: Dict[str, Any]):
        """更新认知结果"""
        if self.interactive_dashboard:
            await self.interactive_dashboard.update_cognitive_results(results)
        
        if self.performance_monitor:
            await self.performance_monitor.update_metrics()
    
    async def update_evolution_results(self, results: Dict[str, Any]):
        """更新进化结果"""
        if self.interactive_dashboard:
            await self.interactive_dashboard.update_evolution_results(results)
        
        if self.performance_monitor:
            await self.performance_monitor.update_metrics()
    
    async def update_learning_progress(self, learning_results: Dict[str, Any]):
        """更新学习进度"""
        # 更新学习相关的可视化
        if self.interactive_dashboard:
            # 添加学习进度到实时数据
            self.interactive_dashboard.real_time_data['learning_metrics'].append({
                'timestamp': self._get_timestamp(),
                'learning_curve': learning_results.get('learning_data', {}),
                'memory_retention': learning_results.get('memory_retention', []),
                'transfer_performance': learning_results.get('transfer_performance', [])
            })
    
    async def update_integrated_results(self, integrated_results: Dict[str, Any]):
        """更新集成结果"""
        # 更新综合实验的可视化
        if self.interactive_dashboard:
            # 更新所有相关数据
            if 'baseline_cognitive' in integrated_results:
                await self.update_cognitive_results(integrated_results['baseline_cognitive'])
            
            if 'evolutionary_improvement' in integrated_results:
                await self.update_evolution_results(integrated_results['evolutionary_improvement'])
    
    async def export_visualization_data(self, output_dir: str = "./visualization_data") -> Dict[str, str]:
        """导出可视化数据"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        exported_files = {}
        
        # 导出性能数据
        if self.performance_monitor:
            performance_data = list(self.performance_monitor.performance_history)
            performance_file = output_path / "performance_data.json"
            
            with open(performance_file, 'w', encoding='utf-8') as f:
                json.dump(performance_data, f, ensure_ascii=False, indent=2)
            
            exported_files['performance'] = str(performance_file)
        
        # 导出认知数据
        if self.interactive_dashboard:
            cognitive_data = list(self.interactive_dashboard.real_time_data['cognitive_metrics'])
            cognitive_file = output_path / "cognitive_data.json"
            
            with open(cognitive_file, 'w', encoding='utf-8') as f:
                json.dump(cognitive_data, f, ensure_ascii=False, indent=2)
            
            exported_files['cognitive'] = str(cognitive_file)
        
        # 导出进化数据
        if self.interactive_dashboard:
            evolution_data = list(self.interactive_dashboard.real_time_data['evolution_metrics'])
            evolution_file = output_path / "evolution_data.json"
            
            with open(evolution_file, 'w', encoding='utf-8') as f:
                json.dump(evolution_data, f, ensure_ascii=False, indent=2)
            
            exported_files['evolution'] = str(evolution_file)
        
        self.logger.info(f"📁 可视化数据已导出到: {output_path}")
        return exported_files
    
    def get_visualization_status(self) -> Dict[str, Any]:
        """获取可视化状态"""
        status = {
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'components': {}
        }
        
        # 渲染器状态
        if self.real_time_renderer:
            status['components']['renderer'] = {
                'running': self.real_time_renderer.is_running,
                'frame_count': self.real_time_renderer.frame_count
            }
        
        # 仪表板状态
        if self.interactive_dashboard:
            status['components']['dashboard'] = {
                'server_running': self.interactive_dashboard.server is not None,
                'data_points': sum(len(data) for data in self.interactive_dashboard.real_time_data.values())
            }
        
        # 性能监控状态
        if self.performance_monitor:
            status['components']['monitor'] = {
                'monitoring': self.performance_monitor.monitoring_enabled,
                'current_metrics': self.performance_monitor.metrics,
                'performance_summary': self.performance_monitor.get_performance_summary()
            }
        
        return status
    
    def _get_timestamp(self) -> float:
        """获取时间戳"""
        import time
        return time.time()
    
    async def cleanup(self):
        """清理资源"""
        self.logger.info("🧹 清理仪表板资源...")
        
        await self.stop_rendering()
        
        if self.real_time_renderer:
            await self.real_time_renderer.cleanup()
        
        if self.interactive_dashboard:
            await self.interactive_dashboard.cleanup()
        
        if self.performance_monitor:
            await self.performance_monitor.cleanup()
        
        self.logger.info("✅ 仪表板资源清理完成")