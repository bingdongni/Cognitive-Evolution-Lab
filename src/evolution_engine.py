#!/usr/bin/env python3
"""
Cognitive Evolution Lab - 协同进化引擎
作者: bingdongni

实现协同进化引擎，包括：
- 遗传算法（变异、交叉、选择）
- 多认知主体进化（协作、竞争、通信）
- 知识进化（经验积累、规则发现、传承学习）
- 环境共演化（适应性变化、复杂度增长）
- 文化演化（社会学习、群体创新）
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import random
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import deque, defaultdict
import copy

# 尝试导入其他库
try:
    import networkx as nx
    NETWORKX_AV认知计算LABLE = True
except ImportError:
    NETWORKX_AV认知计算LABLE = False


class EvolutionType(Enum):
    """进化类型枚举"""
    SINGLE_AGENT = "single_认知主体"
    MULTI_AGENT = "multi_认知主体"
    CO_EVOLUTION = "co_evolution"
    CULTURAL = "cultural"
    KNOWLEDGE = "knowledge"
    ENVIRONMENT = "environment"


class GeneticOperator(Enum):
    """遗传算子枚举"""
    MUTATION = "mutation"
    CROSSOVER = "crossover"
    SELECTION = "selection"
    ELITISM = "elitism"


@dataclass
class Individual:
    """个体类"""
    id: str
    genome: Dict[str, Any]
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    knowledge_assets: Dict[str, Any] = field(default_factory=dict)
    traits: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Population:
    """种群类"""
    individuals: List[Individual]
    generation: int
    size: int
    diversity_score: float
    avg_fitness: float
    best_fitness: float
    avg_age: float
    diversity_trend: List[float] = field(default_factory=list)


@dataclass
class InteractionEvent:
    """交互事件"""
    认知主体1_id: str
    认知主体2_id: str
    interaction_type: str
    outcome: Dict[str, Any]
    timestamp: float
    cooperation_level: float


class GenomeEncoder:
    """基因组编码器"""
    
    def __init__(self, genome_config: Dict[str, Any]):
        self.config = genome_config
        self.encoding_scheme = self._build_encoding_scheme()
    
    def _build_encoding_scheme(self) -> Dict[str, Any]:
        """构建编码方案"""
        scheme = {
            'cognitive_parameters': {
                'learning_rate': {'type': 'float', 'range': [0.001, 0.1]},
                'memory_capacity': {'type': 'int', 'range': [100, 10000]},
                'attention_span': {'type': 'float', 'range': [1.0, 10.0]},
                'creativity_threshold': {'type': 'float', 'range': [0.1, 0.9]}
            },
            'behavioral_parameters': {
                'exploration_rate': {'type': 'float', 'range': [0.0, 1.0]},
                'cooperation_tendency': {'type': 'float', 'range': [0.0, 1.0]},
                'risk_tolerance': {'type': 'float', 'range': [0.0, 1.0]},
                'social_influence': {'type': 'float', 'range': [0.0, 1.0]}
            },
            'structural_parameters': {
                'neural_network_depth': {'type': 'int', 'range': [2, 10]},
                'neural_network_width': {'type': 'int', 'range': [16, 512]},
                'attention_heads': {'type': 'int', 'range': [1, 16]},
                'memory_layers': {'type': 'int', 'range': [1, 5]}
            }
        }
        return scheme
    
    def encode_individual(self, individual: Individual) -> torch.Tensor:
        """将个体编码为张量"""
        # 提取基因参数
        genes = []
        
        # 认知参数
        cognitive = individual.genome.get('cognitive_parameters', {})
        for param_name, param_config in self.encoding_scheme['cognitive_parameters'].items():
            value = cognitive.get(param_name, param_config['range'][0])
            # 标准化到[0,1]
            normalized = (value - param_config['range'][0]) / (param_config['range'][1] - param_config['range'][0])
            genes.append(normalized)
        
        # 行为参数
        behavioral = individual.genome.get('behavioral_parameters', {})
        for param_name, param_config in self.encoding_scheme['behavioral_parameters'].items():
            value = behavioral.get(param_name, param_config['range'][0])
            normalized = (value - param_config['range'][0]) / (param_config['range'][1] - param_config['range'][0])
            genes.append(normalized)
        
        # 结构参数
        structural = individual.genome.get('structural_parameters', {})
        for param_name, param_config in self.encoding_scheme['structural_parameters'].items():
            value = structural.get(param_name, param_config['range'][0])
            normalized = (value - param_config['range'][0]) / (param_config['range'][1] - param_config['range'][0])
            genes.append(normalized)
        
        return torch.tensor(genes, dtype=torch.float32)
    
    def decode_individual(self, encoded_genome: torch.Tensor) -> Dict[str, Any]:
        """从张量解码个体"""
        genome = {
            'cognitive_parameters': {},
            'behavioral_parameters': {},
            'structural_parameters': {}
        }
        
        gene_index = 0
        
        # 解码认知参数
        for param_name, param_config in self.encoding_scheme['cognitive_parameters'].items():
            if gene_index < len(encoded_genome):
                normalized = encoded_genome[gene_index].item()
                value = param_config['range'][0] + normalized * (param_config['range'][1] - param_config['range'][0])
                if param_config['type'] == 'int':
                    value = int(round(value))
                genome['cognitive_parameters'][param_name] = value
                gene_index += 1
        
        # 解码行为参数
        for param_name, param_config in self.encoding_scheme['behavioral_parameters'].items():
            if gene_index < len(encoded_genome):
                normalized = encoded_genome[gene_index].item()
                value = param_config['range'][0] + normalized * (param_config['range'][1] - param_config['range'][0])
                genome['behavioral_parameters'][param_name] = value
                gene_index += 1
        
        # 解码结构参数
        for param_name, param_config in self.encoding_scheme['structural_parameters'].items():
            if gene_index < len(encoded_genome):
                normalized = encoded_genome[gene_index].item()
                value = param_config['range'][0] + normalized * (param_config['range'][1] - param_config['range'][0])
                if param_config['type'] == 'int':
                    value = int(round(value))
                genome['structural_parameters'][param_name] = value
                gene_index += 1
        
        return genome


class GeneticOperators:
    """遗传算子集合"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 遗传参数
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.elitism_rate = config.get('elitism_rate', 0.1)
        self.diversity_threshold = config.get('diversity_threshold', 0.1)
    
    def mutate(self, individual: Individual, mutation_strength: float = 0.1) -> Individual:
        """变异操作"""
        mutated = copy.deepcopy(individual)
        mutated.id = f"{individual.id}_mut_{random.randint(1000, 9999)}"
        
        # 认知参数变异
        cognitive_params = mutated.genome.get('cognitive_parameters', {})
        for param_name, value in cognitive_params.items():
            if random.random() < self.mutation_rate:
                # 高斯变异
                mutation = np.random.normal(0, mutation_strength)
                new_value = value + mutation
                # 限制范围
                new_value = max(0.001, min(0.1, new_value))
                cognitive_params[param_name] = new_value
        
        # 行为参数变异
        behavioral_params = mutated.genome.get('behavioral_parameters', {})
        for param_name, value in behavioral_params.items():
            if random.random() < self.mutation_rate:
                mutation = np.random.normal(0, mutation_strength)
                new_value = value + mutation
                new_value = max(0.0, min(1.0, new_value))
                behavioral_params[param_name] = new_value
        
        # 结构参数变异
        structural_params = mutated.genome.get('structural_parameters', {})
        for param_name, value in structural_params.items():
            if random.random() < self.mutation_rate:
                if param_name in ['neural_network_depth', 'neural_network_width', 'attention_heads', 'memory_layers']:
                    # 整数参数变异
                    mutation = np.random.choice([-1, 1])
                    new_value = max(1, value + mutation)
                    structural_params[param_name] = new_value
        
        mutated.genome['cognitive_parameters'] = cognitive_params
        mutated.genome['behavioral_parameters'] = behavioral_params
        mutated.genome['structural_parameters'] = structural_params
        
        # 重置适应度
        mutated.fitness = 0.0
        mutated.parents = [individual.id]
        
        return mutated
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作"""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        child1.id = f"cross_{parent1.id}_{parent2.id}_1"
        child2.id = f"cross_{parent1.id}_{parent2.id}_2"
        
        # 基因组交叉
        self._crossover_parameters(child1.genome, child2.genome, parent1.genome, parent2.genome)
        
        # 重置适应度
        child1.fitness = 0.0
        child2.fitness = 0.0
        
        # 设置父母
        child1.parents = [parent1.id, parent2.id]
        child2.parents = [parent1.id, parent2.id]
        
        return child1, child2
    
    def _crossover_parameters(self, child1_genome: Dict, child2_genome: Dict, 
                             parent1_genome: Dict, parent2_genome: Dict):
        """交叉参数"""
        # 认知参数交叉
        cognitive1 = child1_genome.get('cognitive_parameters', {})
        cognitive2 = child2_genome.get('cognitive_parameters', {})
        
        p1_cognitive = parent1_genome.get('cognitive_parameters', {})
        p2_cognitive = parent2_genome.get('cognitive_parameters', {})
        
        for param_name in p1_cognitive.keys():
            if random.random() < 0.5:
                cognitive1[param_name] = p1_cognitive[param_name]
                cognitive2[param_name] = p2_cognitive[param_name]
            else:
                cognitive1[param_name] = p2_cognitive[param_name]
                cognitive2[param_name] = p1_cognitive[param_name]
        
        # 行为参数交叉
        behavioral1 = child1_genome.get('behavioral_parameters', {})
        behavioral2 = child2_genome.get('behavioral_parameters', {})
        
        p1_behavioral = parent1_genome.get('behavioral_parameters', {})
        p2_behavioral = parent2_genome.get('behavioral_parameters', {})
        
        for param_name in p1_behavioral.keys():
            if random.random() < 0.5:
                behavioral1[param_name] = p1_behavioral[param_name]
                behavioral2[param_name] = p2_behavioral[param_name]
            else:
                behavioral1[param_name] = p2_behavioral[param_name]
                behavioral2[param_name] = p1_behavioral[param_name]
        
        # 结构参数交叉
        structural1 = child1_genome.get('structural_parameters', {})
        structural2 = child2_genome.get('structural_parameters', {})
        
        p1_structural = parent1_genome.get('structural_parameters', {})
        p2_structural = parent2_genome.get('structural_parameters', {})
        
        for param_name in p1_structural.keys():
            if random.random() < 0.5:
                structural1[param_name] = p1_structural[param_name]
                structural2[param_name] = p2_structural[param_name]
            else:
                structural1[param_name] = p2_structural[param_name]
                structural2[param_name] = p1_structural[param_name]
        
        child1_genome['cognitive_parameters'] = cognitive1
        child1_genome['behavioral_parameters'] = behavioral1
        child1_genome['structural_parameters'] = structural1
        
        child2_genome['cognitive_parameters'] = cognitive2
        child2_genome['behavioral_parameters'] = behavioral2
        child2_genome['structural_parameters'] = structural2
    
    def select_tournament(self, population: List[Individual], tournament_size: int = 3) -> Individual:
        """锦标赛选择"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda ind: ind.fitness)
    
    def select_roulette(self, population: List[Individual]) -> Individual:
        """轮盘赌选择"""
        total_fitness = sum(ind.fitness for ind in population if ind.fitness > 0)
        if total_fitness == 0:
            return random.choice(population)
        
        # 构建轮盘
        selection_probs = []
        cumulative_probs = []
        
        cumulative = 0.0
        for ind in population:
            prob = ind.fitness / total_fitness if ind.fitness > 0 else 0.001
            selection_probs.append(prob)
            cumulative += prob
            cumulative_probs.append(cumulative)
        
        # 选择
        rand = random.random()
        for i, cum_prob in enumerate(cumulative_probs):
            if rand <= cum_prob:
                return population[i]
        
        return population[-1]
    
    def calculate_diversity(self, population: List[Individual]) -> float:
        """计算种群多样性"""
        if len(population) < 2:
            return 1.0
        
        # 计算基因距离
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = self._calculate_genome_distance(population[i], population[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_genome_distance(self, ind1: Individual, ind2: Individual) -> float:
        """计算基因组距离"""
        # 简化的距离计算
        distance = 0.0
        count = 0
        
        # 认知参数距离
        cog1 = ind1.genome.get('cognitive_parameters', {})
        cog2 = ind2.genome.get('cognitive_parameters', {})
        for param in cog1:
            if param in cog2:
                distance += abs(cog1[param] - cog2[param])
                count += 1
        
        # 行为参数距离
        beh1 = ind1.genome.get('behavioral_parameters', {})
        beh2 = ind2.genome.get('behavioral_parameters', {})
        for param in beh1:
            if param in beh2:
                distance += abs(beh1[param] - beh2[param])
                count += 1
        
        return distance / max(1, count) if count > 0 else 0.0


class MultiAgentEvolution:
    """多认知主体进化管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 多认知主体参数
        self.cooperation_threshold = config.get('cooperation_threshold', 0.7)
        self.competition_weight = config.get('competition_weight', 0.3)
        self.communication_enabled = config.get('communication_enabled', True)
        self.coalition_formation = config.get('coalition_formation', True)
        
        # 交互网络
        self.interaction_network = None
        if NETWORKX_AV认知计算LABLE:
            self.interaction_network = nx.Graph()
        
        # 社交结构
        self.social_groups = defaultdict(list)
        self.leadership_structure = {}
        
        self.logger.info("🤝 多认知主体进化管理器初始化完成")
    
    def create_认知主体_network(self, population: List[Individual]) -> Dict[str, Any]:
        """创建认知主体网络"""
        network_info = {
            'nodes': [],
            'edges': [],
            'groups': [],
            'centrality_scores': {}
        }
        
        # 添加节点
        for individual in population:
            network_info['nodes'].append({
                'id': individual.id,
                'fitness': individual.fitness,
                'generation': individual.generation,
                'traits': individual.traits
            })
        
        # 计算连接（基于相似性和交互历史）
        for i, ind1 in enumerate(population):
            for j, ind2 in enumerate(population[i+1:], i+1):
                # 计算连接强度
                connection_strength = self._calculate_connection_strength(ind1, ind2)
                
                if connection_strength > 0.3:  # 连接阈值
                    network_info['edges'].append({
                        'source': ind1.id,
                        'target': ind2.id,
                        'weight': connection_strength
                    })
                    
                    if self.interaction_network:
                        self.interaction_network.add_edge(ind1.id, ind2.id, weight=connection_strength)
        
        # 检测群体结构
        if self.interaction_network and NETWORKX_AV认知计算LABLE:
            communities = list(nx.community.greedy_modularity_communities(self.interaction_network))
            network_info['groups'] = [
                {'members': [node for node in community], 'size': len(community)}
                for community in communities
            ]
        
        return network_info
    
    def _calculate_connection_strength(self, ind1: Individual, ind2: Individual) -> float:
        """计算连接强度"""
        # 基于行为相似性
        beh1 = ind1.genome.get('behavioral_parameters', {})
        beh2 = ind2.genome.get('behavioral_parameters', {})
        
        if beh1 and beh2:
            beh_similarity = 1.0 - abs(beh1.get('cooperation_tendency', 0.5) - 
                                     beh2.get('cooperation_tendency', 0.5))
        else:
            beh_similarity = 0.5
        
        # 基于适应度差距（相似适应度的个体更容易连接）
        fitness_similarity = 1.0 - abs(ind1.fitness - ind2.fitness)
        
        # 基于世代差距（相近世代的个体更容易连接）
        generation_similarity = 1.0 - abs(ind1.generation - ind2.generation) / 10.0
        
        # 综合连接强度
        connection_strength = (beh_similarity * 0.4 + 
                             fitness_similarity * 0.3 + 
                             generation_similarity * 0.3)
        
        return min(1.0, max(0.0, connection_strength))
    
    def simulate_interactions(self, population: List[Individual], 
                            environment: Any) -> List[InteractionEvent]:
        """模拟认知主体交互"""
        interaction_events = []
        
        # 随机选择交互对
        interaction_pairs = self._select_interaction_pairs(population)
        
        for 认知主体1_id, 认知主体2_id in interaction_pairs:
            ind1 = next(ind for ind in population if ind.id == 认知主体1_id)
            ind2 = next(ind for ind in population if ind.id == 认知主体2_id)
            
            # 执行交互
            event = self._execute_interaction(ind1, ind2, environment)
            if event:
                interaction_events.append(event)
                
                # 更新个体的知识资产
                self._update_knowledge_assets(ind1, ind2, event)
        
        return interaction_events
    
    def _select_interaction_pairs(self, population: List[Individual]) -> List[Tuple[str, str]]:
        """选择交互对"""
        pairs = []
        population_size = len(population)
        
        # 基于网络结构选择交互
        if self.interaction_network and NETWORKX_AV认知计算LABLE:
            # 从已有连接中选择
            edges = list(self.interaction_network.edges())
            if edges:
                selected_edges = random.sample(edges, min(len(edges), population_size // 4))
                pairs = [(edge[0], edge[1]) for edge in selected_edges]
        
        # 如果网络连接不足，随机生成连接
        while len(pairs) < population_size // 4:
            ind1 = random.choice(population)
            ind2 = random.choice(population)
            if ind1.id != ind2.id:
                pairs.append((ind1.id, ind2.id))
        
        return pairs
    
    def _execute_interaction(self, ind1: Individual, ind2: Individual, 
                           environment: Any) -> Optional[InteractionEvent]:
        """执行交互"""
        # 获取交互参数
        coop1 = ind1.genome.get('behavioral_parameters', {}).get('cooperation_tendency', 0.5)
        coop2 = ind2.genome.get('behavioral_parameters', {}).get('cooperation_tendency', 0.5)
        
        # 计算合作水平
        cooperation_level = (coop1 + coop2) / 2.0
        
        # 执行交互
        if random.random() < cooperation_level:
            interaction_type = "cooperation"
            outcome = self._cooperation_outcome(ind1, ind2, environment)
        else:
            interaction_type = "competition"
            outcome = self._competition_outcome(ind1, ind2, environment)
        
        # 计算结果质量
        quality = self._calculate_interaction_quality(outcome, cooperation_level)
        
        return InteractionEvent(
            认知主体1_id=ind1.id,
            认知主体2_id=ind2.id,
            interaction_type=interaction_type,
            outcome=outcome,
            timestamp=self._get_timestamp(),
            cooperation_level=cooperation_level
        )
    
    def _cooperation_outcome(self, ind1: Individual, ind2: Individual, 
                           environment: Any) -> Dict[str, Any]:
        """合作结果"""
        # 合作产生的收益
        base_gain = (ind1.fitness + ind2.fitness) / 2.0
        cooperation_bonus = random.uniform(0.1, 0.3)
        
        # 知识共享
        knowledge_sharing = {
            'shared_by_ind1': random.random() < 0.7,
            'shared_by_ind2': random.random() < 0.7,
            'knowledge_transfer': random.uniform(0.1, 0.8)
        }
        
        return {
            'type': 'cooperation',
            'joint_benefit': base_gain * (1 + cooperation_bonus),
            'individual_gain': base_gain * 0.8,
            'knowledge_sharing': knowledge_sharing,
            'trust_level': random.uniform(0.6, 1.0)
        }
    
    def _competition_outcome(self, ind1: Individual, ind2: Individual, 
                           environment: Any) -> Dict[str, Any]:
        """竞争结果"""
        # 竞争产生的收益
        ind1_fitness = ind1.fitness
        ind2_fitness = ind2.fitness
        
        if ind1_fitness > ind2_fitness:
            winner = ind1
            loser = ind2
            winner_gain = 0.2
            loser_loss = -0.1
        elif ind2_fitness > ind1_fitness:
            winner = ind2
            loser = ind1
            winner_gain = 0.2
            loser_loss = -0.1
        else:
            # 平局
            return {
                'type': 'competition',
                'result': 'draw',
                'mutual_learning': random.uniform(0.1, 0.5)
            }
        
        return {
            'type': 'competition',
            'result': f'{winner.id}_wins',
            'winner_gain': winner_gain,
            'loser_loss': loser_loss,
            'learning_opportunity': random.uniform(0.1, 0.6),
            'conflict_level': random.uniform(0.3, 0.8)
        }
    
    def _calculate_interaction_quality(self, outcome: Dict[str, Any], 
                                     cooperation_level: float) -> float:
        """计算交互质量"""
        base_quality = 0.5
        
        if outcome['type'] == 'cooperation':
            quality = base_quality + 0.3 * cooperation_level
        elif outcome['type'] == 'competition':
            quality = base_quality - 0.2 * (1 - cooperation_level)
        else:
            quality = base_quality
        
        return min(1.0, max(0.0, quality))
    
    def _update_knowledge_assets(self, ind1: Individual, ind2: Individual, 
                               event: InteractionEvent):
        """更新知识资产"""
        if event.interaction_type == 'cooperation':
            # 合作促进知识共享
            if random.random() < 0.6:
                # 模拟知识转移
                knowledge_transfer = random.uniform(0.1, 0.5)
                ind1.knowledge_assets[f'learned_from_{ind2.id}'] = knowledge_transfer
                ind2.knowledge_assets[f'learned_from_{ind1.id}'] = knowledge_transfer
        
        elif event.interaction_type == 'competition':
            # 竞争促进学习
            winner = next(ind for ind in [ind1, ind2] 
                         if f'{ind.id}_wins' in event.outcome.get('result', ''))
            
            learning_opportunity = event.outcome.get('learning_opportunity', 0.1)
            winner.knowledge_assets[f'competitive_advantage'] = max(
                winner.knowledge_assets.get('competitive_advantage', 0),
                learning_opportunity
            )
    
    def evolve_social_structure(self, population: List[Individual]) -> Dict[str, Any]:
        """演化社会结构"""
        # 创建或更新认知主体网络
        network_info = self.create_认知主体_network(population)
        
        # 检测领导结构
        leadership = self._detect_leadership_structure(network_info)
        
        # 形成联盟
        coalitions = self._form_coalitions(population, network_info)
        
        # 更新社会群体
        for coalition_id, members in coalitions.items():
            self.social_groups[coalition_id] = members
        
        return {
            'network_info': network_info,
            'leadership': leadership,
            'coalitions': coalitions,
            'social_groups': dict(self.social_groups)
        }
    
    def _detect_leadership_structure(self, network_info: Dict[str, Any]) -> Dict[str, Any]:
        """检测领导结构"""
        leadership = {
            'leaders': [],
            'followers': [],
            'influence_network': []
        }
        
        # 简单的领导检测：适应度最高的个体
        nodes = network_info.get('nodes', [])
        if nodes:
            leaders = sorted(nodes, key=lambda x: x['fitness'], reverse=True)[:3]
            leadership['leaders'] = [node['id'] for node in leaders]
            
            # 其他个体为跟随者
            leader_ids = set(leadership['leaders'])
            followers = [node for node in nodes if node['id'] not in leader_ids]
            leadership['followers'] = [node['id'] for node in followers]
        
        return leadership
    
    def _form_coalitions(self, population: List[Individual], 
                        network_info: Dict[str, Any]) -> Dict[str, str]:
        """形成联盟"""
        coalitions = {}
        
        # 基于适应度和相似性形成联盟
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # 简单的联盟形成：每3-5个个体组成一个联盟
        coalition_size = random.randint(3, 5)
        
        for i in range(0, len(sorted_population), coalition_size):
            coalition_members = [ind.id for ind in sorted_population[i:i+coalition_size]]
            coalition_id = f"coalition_{i//coalition_size}"
            
            for member_id in coalition_members:
                coalitions[member_id] = coalition_id
        
        return coalitions


class EvolutionEngine:
    """
    协同进化引擎主类
    
    整合所有进化功能：
    - 遗传算法操作
    - 多认知主体进化
    - 知识进化
    - 环境共演化
    - 文化演化
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化进化引擎
        
        Args:
            config: 进化配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 进化参数
        self.population_size = config.get('population_size', 100)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.elitism_rate = config.get('elitism_rate', 0.1)
        
        # 进化类型
        self.evolution_type = EvolutionType(config.get('evolution_type', 'multi_认知主体'))
        
        # 初始化组件
        self.genome_encoder = None
        self.genetic_operators = None
        self.multi_认知主体_evolution = None
        
        # 进化状态
        self.current_population = Population(
            individuals=[],
            generation=0,
            size=0,
            diversity_score=0.0,
            avg_fitness=0.0,
            best_fitness=0.0,
            avg_age=0.0
        )
        
        # 进化历史
        self.evolution_history = deque(maxlen=1000)
        
        # 知识库
        self.knowledge_base = {
            'discovered_rules': [],
            'successful_strategies': [],
            'social_patterns': [],
            'environmental_adaptations': []
        }
        
        # 环境信息
        self.environment_info = {
            'complexity_level': 1.0,
            'challenge_types': [],
            'adaptive_pressures': []
        }
        
        self.logger.info("🧬 协同进化引擎初始化完成")
    
    async def initialize(self):
        """异步初始化进化引擎"""
        self.logger.info("🔧 初始化进化引擎组件...")
        
        try:
            # 初始化基因组编码器
            genome_config = self.config.get('genome_config', {})
            self.genome_encoder = GenomeEncoder(genome_config)
            
            # 初始化遗传算子
            genetic_config = self.config.get('genetic_config', {})
            self.genetic_operators = GeneticOperators(genetic_config)
            
            # 初始化多认知主体进化
            multi_认知主体_config = self.config.get('multi_认知主体_config', {})
            self.multi_认知主体_evolution = MultiAgentEvolution(multi_认知主体_config)
            
            self.logger.info("✅ 进化引擎组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 进化引擎组件初始化失败: {e}")
            raise
    
    async def initialize_population(self, environment=None, 
                                  experiment_type: str = "multi_认知主体") -> List[Individual]:
        """初始化种群"""
        self.logger.info(f"👥 初始化种群，大小: {self.population_size}")
        
        individuals = []
        
        for i in range(self.population_size):
            # 创建随机基因组
            genome = self._generate_random_genome()
            
            # 创建个体
            individual = Individual(
                id=f"ind_{i}_{random.randint(1000, 9999)}",
                genome=genome,
                fitness=0.0,
                age=0,
                generation=0,
                traits=self._generate_traits(genome)
            )
            
            individuals.append(individual)
        
        # 更新种群状态
        self.current_population = Population(
            individuals=individuals,
            generation=0,
            size=len(individuals),
            diversity_score=self.genetic_operators.calculate_diversity(individuals),
            avg_fitness=0.0,
            best_fitness=0.0,
            avg_age=0.0
        )
        
        self.logger.info(f"✅ 种群初始化完成，包含{len(individuals)}个个体")
        return individuals
    
    def _generate_random_genome(self) -> Dict[str, Any]:
        """生成随机基因组"""
        genome = {
            'cognitive_parameters': {
                'learning_rate': random.uniform(0.01, 0.05),
                'memory_capacity': random.randint(500, 2000),
                'attention_span': random.uniform(2.0, 8.0),
                'creativity_threshold': random.uniform(0.3, 0.8)
            },
            'behavioral_parameters': {
                'exploration_rate': random.uniform(0.1, 0.9),
                'cooperation_tendency': random.uniform(0.0, 1.0),
                'risk_tolerance': random.uniform(0.1, 0.8),
                'social_influence': random.uniform(0.0, 1.0)
            },
            'structural_parameters': {
                'neural_network_depth': random.randint(3, 7),
                'neural_network_width': random.randint(64, 256),
                'attention_heads': random.randint(4, 12),
                'memory_layers': random.randint(1, 3)
            }
        }
        
        return genome
    
    def _generate_traits(self, genome: Dict[str, Any]) -> Dict[str, float]:
        """从基因组生成特征"""
        traits = {
            'exploration_tendency': genome['behavioral_parameters']['exploration_rate'],
            'cooperation_level': genome['behavioral_parameters']['cooperation_tendency'],
            'cognitive_flexibility': 1.0 / genome['cognitive_parameters']['learning_rate'],
            'memory_efficiency': min(1.0, genome['cognitive_parameters']['memory_capacity'] / 1000.0),
            'attention_capacity': genome['cognitive_parameters']['attention_span'] / 10.0,
            'creativity_potential': genome['cognitive_parameters']['creativity_threshold']
        }
        
        return traits
    
    async def evolve(self, population: List[Individual], environment=None, 
                    generations: int = 100) -> Dict[str, Any]:
        """进化过程"""
        self.logger.info(f"🧬 开始进化过程，代数: {generations}")
        
        evolution_results = {
            'generations': [],
            'best_fitness_history': [],
            'diversity_history': [],
            'population_history': [],
            'knowledge_evolution': [],
            'social_evolution': [],
            'environmental_changes': []
        }
        
        current_population = population
        
        for generation in range(generations):
            self.logger.info(f"📈 第{generation + 1}代进化")
            
            # 评估适应度
            fitness_scores = await self._evaluate_fitness(current_population, environment)
            
            # 更新个体适应度
            for individual, fitness in zip(current_population, fitness_scores):
                individual.fitness = fitness
                individual.generation = generation
            
            # 计算种群统计
            population_stats = self._calculate_population_stats(current_population)
            evolution_results['generations'].append(population_stats)
            
            # 记录历史数据
            evolution_results['best_fitness_history'].append(population_stats['best_fitness'])
            evolution_results['diversity_history'].append(population_stats['diversity_score'])
            
            # 多认知主体进化
            if self.evolution_type in [EvolutionType.MULTI_AGENT, EvolutionType.CO_EVOLUTION]:
                social_evolution = await self._evolve_social_structure(current_population, environment)
                evolution_results['social_evolution'].append(social_evolution)
            
            # 知识进化
            knowledge_evolution = await self._evolve_knowledge(current_population)
            evolution_results['knowledge_evolution'].append(knowledge_evolution)
            
            # 环境共演化
            if generation % 10 == 0:  # 每10代更新一次环境
                environmental_changes = await self._co_evolve_environment(current_population, generation)
                evolution_results['environmental_changes'].append(environmental_changes)
            
            # 产生下一代
            if generation < generations - 1:  # 最后一代不需要产生下一代
                current_population = await self._generate_next_generation(current_population)
            
            # 更新当前种群
            self.current_population = Population(
                individuals=current_population,
                generation=generation,
                size=len(current_population),
                diversity_score=self.genetic_operators.calculate_diversity(current_population),
                avg_fitness=population_stats['avg_fitness'],
                best_fitness=population_stats['best_fitness'],
                avg_age=population_stats['avg_age'],
                diversity_trend=evolution_results['diversity_history']
            )
        
        # 生成最终结果
        final_results = self._compile_final_results(evolution_results, current_population)
        
        self.logger.info(f"✅ 进化过程完成，最终最佳适应度: {final_results['final_fitness']:.4f}")
        return final_results
    
    async def _evaluate_fitness(self, population: List[Individual], environment=None) -> List[float]:
        """评估适应度"""
        fitness_scores = []
        
        # 为每个个体评估适应度
        for individual in population:
            # 简化的适应度评估
            cognitive_score = self._evaluate_cognitive_fitness(individual)
            behavioral_score = self._evaluate_behavioral_fitness(individual)
            structural_score = self._evaluate_structural_fitness(individual)
            knowledge_score = self._evaluate_knowledge_fitness(individual)
            
            # 综合适应度
            total_fitness = (cognitive_score * 0.3 + 
                           behavioral_score * 0.3 + 
                           structural_score * 0.2 + 
                           knowledge_score * 0.2)
            
            fitness_scores.append(total_fitness)
        
        return fitness_scores
    
    def _evaluate_cognitive_fitness(self, individual: Individual) -> float:
        """评估认知适应度"""
        cognitive_params = individual.genome.get('cognitive_parameters', {})
        
        # 学习率适中性（太高或太低都不好）
        learning_rate = cognitive_params.get('learning_rate', 0.01)
        learning_score = 1.0 - abs(learning_rate - 0.03) / 0.03
        
        # 记忆容量效率
        memory_capacity = cognitive_params.get('memory_capacity', 1000)
        memory_score = min(1.0, memory_capacity / 1500.0)
        
        # 注意力范围平衡
        attention_span = cognitive_params.get('attention_span', 5.0)
        attention_score = 1.0 - abs(attention_span - 5.0) / 5.0
        
        # 创造力阈值适中性
        creativity = cognitive_params.get('creativity_threshold', 0.5)
        creativity_score = 1.0 - abs(creativity - 0.6) / 0.6
        
        return (learning_score + memory_score + attention_score + creativity_score) / 4.0
    
    def _evaluate_behavioral_fitness(self, individual: Individual) -> float:
        """评估行为适应度"""
        behavioral_params = individual.genome.get('behavioral_parameters', {})
        
        # 探索率平衡
        exploration = behavioral_params.get('exploration_rate', 0.5)
        exploration_score = 4 * exploration * (1 - exploration)  # 二次函数，最大值在0.5
        
        # 合作倾向适中性
        cooperation = behavioral_params.get('cooperation_tendency', 0.5)
        cooperation_score = 1.0 - abs(cooperation - 0.7) / 0.7
        
        # 风险容忍度平衡
        risk_tolerance = behavioral_params.get('risk_tolerance', 0.5)
        risk_score = 4 * risk_tolerance * (1 - risk_tolerance)
        
        # 社会影响力适中
        social_influence = behavioral_params.get('social_influence', 0.5)
        influence_score = 1.0 - abs(social_influence - 0.6) / 0.6
        
        return (exploration_score + cooperation_score + risk_score + influence_score) / 4.0
    
    def _evaluate_structural_fitness(self, individual: Individual) -> float:
        """评估结构适应度"""
        structural_params = individual.genome.get('structural_parameters', {})
        
        # 网络深度适中
        depth = structural_params.get('neural_network_depth', 5)
        depth_score = 1.0 - abs(depth - 5) / 5
        
        # 网络宽度适中性
        width = structural_params.get('neural_network_width', 128)
        width_score = 1.0 - abs(width - 128) / 128
        
        # 注意力头数平衡
        heads = structural_params.get('attention_heads', 8)
        heads_score = 1.0 - abs(heads - 8) / 8
        
        # 记忆层数适中
        memory_layers = structural_params.get('memory_layers', 2)
        memory_score = 1.0 - abs(memory_layers - 2) / 2
        
        return (depth_score + width_score + heads_score + memory_score) / 4.0
    
    def _evaluate_knowledge_fitness(self, individual: Individual) -> float:
        """评估知识适应度"""
        knowledge_score = 0.5  # 基础分数
        
        # 知识资产数量
        num_assets = len(individual.knowledge_assets)
        asset_score = min(1.0, num_assets / 10.0)
        
        # 知识多样性
        if individual.knowledge_assets:
            diversity_score = len(set(individual.knowledge_assets.values())) / len(individual.knowledge_assets)
        else:
            diversity_score = 0.0
        
        # 社会学习能力
        social_learning = 0.0
        for key, value in individual.knowledge_assets.items():
            if 'learned_from' in key:
                social_learning += value
        
        social_score = min(1.0, social_learning / 5.0)
        
        return (knowledge_score + asset_score + diversity_score + social_score) / 4.0
    
    def _calculate_population_stats(self, population: List[Individual]) -> Dict[str, Any]:
        """计算种群统计"""
        if not population:
            return {}
        
        fitnesses = [ind.fitness for ind in population]
        ages = [ind.age for ind in population]
        
        stats = {
            'generation': population[0].generation if population else 0,
            'population_size': len(population),
            'avg_fitness': np.mean(fitnesses),
            'best_fitness': max(fitnesses),
            'worst_fitness': min(fitnesses),
            'fitness_std': np.std(fitnesses),
            'avg_age': np.mean(ages),
            'diversity_score': self.genetic_operators.calculate_diversity(population),
            'genetic_variance': self._calculate_genetic_variance(population)
        }
        
        return stats
    
    def _calculate_genetic_variance(self, population: List[Individual]) -> float:
        """计算基因方差"""
        if len(population) < 2:
            return 0.0
        
        variances = []
        
        # 认知参数方差
        cognitive_params = ['learning_rate', 'memory_capacity', 'attention_span', 'creativity_threshold']
        for param in cognitive_params:
            values = []
            for ind in population:
                cog_params = ind.genome.get('cognitive_parameters', {})
                if param in cog_params:
                    values.append(cog_params[param])
            
            if values:
                variances.append(np.var(values))
        
        # 行为参数方差
        behavioral_params = ['exploration_rate', 'cooperation_tendency', 'risk_tolerance', 'social_influence']
        for param in behavioral_params:
            values = []
            for ind in population:
                beh_params = ind.genome.get('behavioral_parameters', {})
                if param in beh_params:
                    values.append(beh_params[param])
            
            if values:
                variances.append(np.var(values))
        
        return np.mean(variances) if variances else 0.0
    
    async def _evolve_social_structure(self, population: List[Individual], 
                                     environment=None) -> Dict[str, Any]:
        """演化社会结构"""
        if not self.multi_认知主体_evolution:
            return {}
        
        # 演化社会结构
        social_evolution = self.multi_认知主体_evolution.evolve_social_structure(population)
        
        # 模拟认知主体交互
        interaction_events = self.multi_认知主体_evolution.simulate_interactions(population, environment)
        
        # 更新社交结构信息
        social_evolution['interaction_events'] = [
            {
                '认知主体1_id': event.认知主体1_id,
                '认知主体2_id': event.认知主体2_id,
                'type': event.interaction_type,
                'cooperation_level': event.cooperation_level,
                'outcome_quality': self.multi_认知主体_evolution._calculate_interaction_quality(
                    event.outcome, event.cooperation_level
                )
            }
            for event in interaction_events
        ]
        
        return social_evolution
    
    async def _evolve_knowledge(self, population: List[Individual]) -> Dict[str, Any]:
        """知识进化"""
        knowledge_evolution = {
            'new_discoveries': [],
            'knowledge_transfer_events': [],
            'rule_formation': [],
            'collective_intelligence': {}
        }
        
        # 检测新发现
        for individual in population:
            if individual.knowledge_assets:
                for key, value in individual.knowledge_assets.items():
                    if value > 0.7:  # 高价值知识
                        discovery = {
                            'discovered_by': individual.id,
                            'knowledge_type': key,
                            'quality_score': value,
                            'generation': individual.generation
                        }
                        knowledge_evolution['new_discoveries'].append(discovery)
        
        # 知识转移事件
        for individual in population:
            for key in individual.knowledge_assets:
                if 'learned_from' in key:
                    transfer_event = {
                        'learner': individual.id,
                        'teacher': key.replace('learned_from_', ''),
                        'knowledge_quality': individual.knowledge_assets[key],
                        'generation': individual.generation
                    }
                    knowledge_evolution['knowledge_transfer_events'].append(transfer_event)
        
        # 规则形成
        if len(knowledge_evolution['new_discoveries']) > 5:
            # 简化的规则形成逻辑
            common_patterns = self._extract_common_patterns(population)
            knowledge_evolution['rule_formation'] = common_patterns
        
        # 集体智能
        knowledge_evolution['collective_intelligence'] = {
            'population_knowledge': len(set(key for ind in population for key in ind.knowledge_assets)),
            'knowledge_diversity': self._calculate_knowledge_diversity(population),
            'learning_velocity': len(knowledge_evolution['knowledge_transfer_events']) / len(population)
        }
        
        return knowledge_evolution
    
    def _extract_common_patterns(self, population: List[Individual]) -> List[Dict[str, Any]]:
        """提取共同模式"""
        patterns = []
        
        # 简化的模式提取：查找普遍存在的知识资产
        asset_counts = defaultdict(int)
        
        for individual in population:
            for asset_key in individual.knowledge_assets:
                asset_counts[asset_key] += 1
        
        # 识别高频模式
        for asset_key, count in asset_counts.items():
            if count > len(population) * 0.3:  # 30%以上的个体拥有
                patterns.append({
                    'pattern_type': asset_key,
                    'prevalence': count / len(population),
                    'strength': count
                })
        
        return patterns
    
    def _calculate_knowledge_diversity(self, population: List[Individual]) -> float:
        """计算知识多样性"""
        all_assets = set()
        individual_assets = []
        
        for individual in population:
            assets = set(individual.knowledge_assets.keys())
            individual_assets.append(len(assets))
            all_assets.update(assets)
        
        if not all_assets:
            return 0.0
        
        # 计算基于个体知识重叠的多样性
        if len(individual_assets) > 0:
            avg_individual_knowledge = np.mean(individual_assets)
            total_knowledge = len(all_assets)
            diversity = total_knowledge / (avg_individual_knowledge + 1)
            return min(1.0, diversity / 10.0)
        
        return 0.0
    
    async def _co_evolve_environment(self, population: List[Individual], generation: int) -> Dict[str, Any]:
        """环境共演化"""
        # 计算种群适应度分布
        fitnesses = [ind.fitness for ind in population]
        avg_fitness = np.mean(fitnesses)
        fitness_variance = np.var(fitnesses)
        
        # 基于种群表现调整环境复杂度
        adaptation_rate = 0.01
        complexity_change = (avg_fitness - 0.5) * adaptation_rate
        
        self.environment_info['complexity_level'] = max(0.1, 
            self.environment_info['complexity_level'] + complexity_change)
        
        # 动态调整挑战类型
        if generation % 20 == 0:  # 每20代更新挑战
            self.environment_info['challenge_types'] = self._generate_environmental_challenges(
                self.environment_info['complexity_level']
            )
        
        return {
            'complexity_level': self.environment_info['complexity_level'],
            'challenge_types': self.environment_info['challenge_types'],
            'adaptive_pressure': self._calculate_adaptive_pressure(fitness_variance),
            'environmental_fitness': avg_fitness
        }
    
    def _generate_environmental_challenges(self, complexity_level: float) -> List[str]:
        """生成环境挑战"""
        base_challenges = ['resource_scarcity', 'social_competition', 'cognitive_demand']
        
        # 根据复杂度添加更多挑战
        if complexity_level > 0.5:
            base_challenges.extend(['temporal_pressure', 'uncertainty_handling'])
        
        if complexity_level > 0.7:
            base_challenges.extend(['multi_objective', 'dynamic_environment'])
        
        if complexity_level > 0.9:
            base_challenges.extend(['adversarial_conditions', 'extreme_variability'])
        
        return base_challenges
    
    def _calculate_adaptive_pressure(self, fitness_variance: float) -> float:
        """计算适应压力"""
        # 基于适应度方差的适应压力
        # 方差越大，适应压力越大
        return min(1.0, fitness_variance * 4)
    
    async def _generate_next_generation(self, population: List[Individual]) -> List[Individual]:
        """生成下一代"""
        next_generation = []
        
        # 精英保留
        elite_count = int(self.population_size * self.elitism_rate)
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        for i in range(elite_count):
            elite = copy.deepcopy(sorted_population[i])
            elite.id = f"elite_{i}_{random.randint(1000, 9999)}"
            elite.age += 1
            next_generation.append(elite)
        
        # 遗传操作生成剩余个体
        while len(next_generation) < self.population_size:
            # 选择父母
            if random.random() < self.crossover_rate:
                # 交叉
                parent1 = self.genetic_operators.select_tournament(population)
                parent2 = self.genetic_operators.select_tournament(population)
                
                child1, child2 = self.genetic_operators.crossover(parent1, parent2)
                
                # 变异
                if random.random() < self.mutation_rate:
                    child1 = self.genetic_operators.mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.genetic_operators.mutate(child2)
                
                next_generation.extend([child1, child2])
            else:
                # 直接复制和变异
                parent = self.genetic_operators.select_tournament(population)
                child = self.genetic_operators.mutate(parent)
                next_generation.append(child)
        
        # 确保种群大小正确
        next_generation = next_generation[:self.population_size]
        
        # 更新年龄
        for individual in next_generation:
            individual.age += 1
            individual.generation += 1
        
        return next_generation
    
    def _compile_final_results(self, evolution_results: Dict[str, Any], 
                             final_population: List[Individual]) -> Dict[str, Any]:
        """编译最终结果"""
        # 找到最佳个体
        best_individual = max(final_population, key=lambda x: x.fitness)
        
        # 计算多样性分数
        diversity_score = self.genetic_operators.calculate_diversity(final_population)
        
        # 计算最终适应度
        final_fitness = best_individual.fitness
        
        # 统计知识资产
        total_knowledge_assets = sum(len(ind.knowledge_assets) for ind in final_population)
        
        # 计算社会网络信息
        if self.multi_认知主体_evolution:
            social_structure = self.multi_认知主体_evolution.create_认知主体_network(final_population)
        else:
            social_structure = {}
        
        final_results = {
            'final_generation': evolution_results['generations'][-1] if evolution_results['generations'] else {},
            'best_individual': {
                'id': best_individual.id,
                'fitness': best_individual.fitness,
                'genome': best_individual.genome,
                'traits': best_individual.traits,
                'knowledge_assets': best_individual.knowledge_assets,
                'generation': best_individual.generation
            },
            'final_fitness': final_fitness,
            'diversity_score': diversity_score,
            'population_size': len(final_population),
            'total_generations': len(evolution_results['generations']),
            'knowledge_summary': {
                'total_assets': total_knowledge_assets,
                'avg_assets_per_individual': total_knowledge_assets / len(final_population) if final_population else 0,
                'unique_asset_types': len(set(key for ind in final_population for key in ind.knowledge_assets))
            },
            'evolution_metrics': {
                'best_fitness_history': evolution_results['best_fitness_history'],
                'diversity_history': evolution_results['diversity_history'],
                'fitness_improvement': evolution_results['best_fitness_history'][-1] - evolution_results['best_fitness_history'][0] if evolution_results['best_fitness_history'] else 0,
                'diversity_retention': diversity_score
            },
            'social_evolution': evolution_results['social_evolution'][-5:] if evolution_results['social_evolution'] else [],
            'knowledge_evolution': evolution_results['knowledge_evolution'][-5:] if evolution_results['knowledge_evolution'] else [],
            'environmental_adaptation': evolution_results['environmental_changes'][-3:] if evolution_results['environmental_changes'] else [],
            'social_structure': social_structure
        }
        
        return final_results
    
    def get_evolution_metrics(self) -> Dict[str, Any]:
        """获取进化指标"""
        if not self.current_population.individuals:
            return {}
        
        population = self.current_population
        
        return {
            'current_generation': population.generation,
            'population_size': population.size,
            'avg_fitness': population.avg_fitness,
            'best_fitness': population.best_fitness,
            'diversity_score': population.diversity_score,
            'avg_age': population.avg_age,
            'fitness_distribution': {
                'min': min(ind.fitness for ind in population.individuals),
                'max': max(ind.fitness for ind in population.individuals),
                'median': np.median([ind.fitness for ind in population.individuals])
            },
            'trait_distribution': self._analyze_trait_distribution(population.individuals),
            'knowledge_distribution': self._analyze_knowledge_distribution(population.individuals)
        }
    
    def _analyze_trait_distribution(self, population: List[Individual]) -> Dict[str, float]:
        """分析特征分布"""
        traits = ['exploration_tendency', 'cooperation_level', 'cognitive_flexibility', 
                 'memory_efficiency', 'attention_capacity', 'creativity_potential']
        
        distribution = {}
        for trait in traits:
            values = [ind.traits.get(trait, 0.5) for ind in population]
            distribution[trait] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': min(values),
                'max': max(values)
            }
        
        return distribution
    
    def _analyze_knowledge_distribution(self, population: List[Individual]) -> Dict[str, Any]:
        """分析知识分布"""
        knowledge_types = defaultdict(int)
        total_assets = 0
        
        for individual in population:
            for asset_key in individual.knowledge_assets:
                knowledge_types[asset_key] += 1
                total_assets += 1
        
        return {
            'total_assets': total_assets,
            'unique_types': len(knowledge_types),
            'type_distribution': dict(knowledge_types),
            'avg_assets_per_individual': total_assets / len(population) if population else 0
        }
    
    def _get_timestamp(self) -> float:
        """获取时间戳"""
        import time
        return time.time()
    
    async def cleanup(self):
        """清理资源"""
        self.logger.info("🧹 清理进化引擎资源...")
        
        # 清空进化历史
        self.evolution_history.clear()
        
        # 清空当前种群
        self.current_population = Population(
            individuals=[],
            generation=0,
            size=0,
            diversity_score=0.0,
            avg_fitness=0.0,
            best_fitness=0.0,
            avg_age=0.0
        )
        
        self.logger.info("✅ 进化引擎资源清理完成")