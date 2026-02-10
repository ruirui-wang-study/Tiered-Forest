import os
import pickle
import networkx as nx
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re

class MetaQAGraphEngine:
    """
    MetaQA 知识图谱引擎
    负责加载 kb.txt，构建 NetworkX 图，提供邻居检索功能
    """
    def __init__(self, kb_path: str, cache_dir: str = "data/processed"):
        self.kb_path = kb_path
        self.cache_dir = cache_dir
        self.graph = None
        self.entity_index = None  # 实体名称 -> 节点 ID 的映射
        
        # 加载或构建图谱
        self.load_or_build_graph()
    
    def load_or_build_graph(self):
        """加载缓存的图谱，如果不存在则构建"""
        graph_cache = os.path.join(self.cache_dir, "graph.pkl")
        index_cache = os.path.join(self.cache_dir, "entity_index.pkl")
        
        if os.path.exists(graph_cache) and os.path.exists(index_cache):
            print(f"Loading cached graph from {graph_cache}...")
            with open(graph_cache, 'rb') as f:
                self.graph = pickle.load(f)
            with open(index_cache, 'rb') as f:
                self.entity_index = pickle.load(f)
            print(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        else:
            print(f"Building graph from {self.kb_path}...")
            self.build_graph()
            self.save_graph(graph_cache, index_cache)
    
    def build_graph(self):
        """从 kb.txt 构建 NetworkX 图"""
        # 使用 MultiDiGraph 支持多条边（不同关系）
        self.graph = nx.MultiDiGraph()
        self.entity_index = defaultdict(set)
        
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # 解析三元组: subject|relation|object
                parts = line.split('|')
                if len(parts) != 3:
                    continue
                
                subject, relation, obj = parts
                subject = self.normalize_entity(subject)
                obj = self.normalize_entity(obj)
                relation = relation.strip()
                
                # 添加边（MultiDiGraph 支持多条边）
                self.graph.add_edge(subject, obj, relation=relation)
                
                # 更新实体索引
                self.entity_index[subject].add(subject)
                self.entity_index[obj].add(obj)
                
                if line_num % 10000 == 0:
                    print(f"Processed {line_num} triples...")
        
        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def save_graph(self, graph_path: str, index_path: str):
        """保存图谱到缓存"""
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        with open(graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
        with open(index_path, 'wb') as f:
            pickle.dump(dict(self.entity_index), f)
        print(f"Graph cached to {graph_path}")
    
    def normalize_entity(self, entity: str) -> str:
        """归一化实体名称（小写、去空格）"""
        return entity.strip().lower()
    
    def find_entity(self, entity_name: str) -> Optional[str]:
        """
        查找实体（支持模糊匹配）
        
        Args:
            entity_name: 实体名称
            
        Returns:
            标准化后的实体名称，如果不存在则返回 None
        """
        normalized = self.normalize_entity(entity_name)
        
        # 精确匹配
        if normalized in self.graph.nodes:
            return normalized
        
        # 模糊匹配（包含关系）
        for node in self.graph.nodes:
            if normalized in node or node in normalized:
                return node
        
        return None
    
    def get_neighbors(self, entity: str, relation: Optional[str] = None, 
                     direction: str = "out") -> List[Tuple[str, str]]:
        """
        获取实体的邻居节点
        
        Args:
            entity: 实体名称
            relation: 关系类型（可选，用于过滤）
            direction: "out" (出边), "in" (入边), "both" (双向)
            
        Returns:
            [(neighbor, relation), ...] 列表
        """
        entity = self.normalize_entity(entity)
        
        if entity not in self.graph.nodes:
            return []
        
        neighbors = []
        
        # 出边
        if direction in ["out", "both"]:
            for neighbor in self.graph.successors(entity):
                # MultiDiGraph: 遍历所有边
                for key, edge_data in self.graph[entity][neighbor].items():
                    rel = edge_data.get('relation', 'unknown')
                    if relation is None or rel == relation:
                        neighbors.append((neighbor, rel))
        
        # 入边
        if direction in ["in", "both"]:
            for neighbor in self.graph.predecessors(entity):
                # MultiDiGraph: 遍历所有边
                for key, edge_data in self.graph[neighbor][entity].items():
                    rel = edge_data.get('relation', 'unknown')
                    if relation is None or rel == relation:
                        neighbors.append((neighbor, rel))
        
        return neighbors
    
    def get_path(self, start: str, end: str, max_hops: int = 3) -> Optional[List[str]]:
        """
        查找两个实体之间的最短路径
        
        Args:
            start: 起始实体
            end: 目标实体
            max_hops: 最大跳数
            
        Returns:
            路径节点列表，如果不存在则返回 None
        """
        start = self.normalize_entity(start)
        end = self.normalize_entity(end)
        
        if start not in self.graph.nodes or end not in self.graph.nodes:
            return None
        
        try:
            path = nx.shortest_path(self.graph, start, end)
            if len(path) - 1 <= max_hops:
                return path
        except nx.NetworkXNoPath:
            pass
        
        return None
    
    def query_simple(self, question: str) -> Optional[str]:
        """
        简单图谱查询（用于 Tier 1）
        处理简单的模板问题，如 "who directed X?"
        
        Args:
            question: 问题文本
            
        Returns:
            答案，如果无法回答则返回 None
        """
        question_lower = question.lower()
        
        # 模板1: "who directed X?"
        match = re.search(r"who directed (.+?)\?", question_lower)
        if match:
            movie = match.group(1).strip()
            entity = self.find_entity(movie)
            if entity:
                neighbors = self.get_neighbors(entity, relation="directed_by", direction="out")
                if neighbors:
                    return neighbors[0][0]  # 返回第一个导演
        
        # 模板2: "who acted in X?"
        match = re.search(r"who (?:acted in|starred in) (.+?)\?", question_lower)
        if match:
            movie = match.group(1).strip()
            entity = self.find_entity(movie)
            if entity:
                neighbors = self.get_neighbors(entity, relation="starred_actors", direction="out")
                if neighbors:
                    # 返回所有演员（逗号分隔）
                    actors = [n[0] for n in neighbors[:3]]  # 最多返回3个
                    return ", ".join(actors)
        
        return None