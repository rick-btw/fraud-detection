"""
Synthetic Data Generator for creating test graphs with known fraud patterns.
"""

import numpy as np
import random
from typing import List, Tuple, Set
from src.sparse_graph import SparseGraph


class SyntheticDataGenerator:
    """
    Generate synthetic graphs with embedded fraud patterns for testing.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator with random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_scale_free_graph(self, num_nodes: int, num_edges: int,
                                   fraud_cluster_size: int = 10,
                                   fraud_cluster_density: float = 0.8) -> Tuple[SparseGraph, Set[int]]:
        """
        Generate a scale-free graph with an embedded fraud cluster.
        
        Args:
            num_nodes: Total number of nodes
            num_edges: Total number of edges
            fraud_cluster_size: Size of the fraud cluster
            fraud_cluster_density: Edge density within fraud cluster (0-1)
        
        Returns:
            Tuple of (SparseGraph, Set of fraud node IDs)
        """
        graph = SparseGraph(num_nodes=num_nodes)
        
        # Create fraud cluster (seed set)
        fraud_nodes = set(random.sample(range(num_nodes), fraud_cluster_size))
        
        # Generate edges
        edges = []
        edges_added = 0
        
        # Add dense edges within fraud cluster
        fraud_list = list(fraud_nodes)
        num_fraud_edges = int(fraud_cluster_size * (fraud_cluster_size - 1) * fraud_cluster_density)
        fraud_edges_added = 0
        
        for i in range(len(fraud_list)):
            for j in range(i + 1, len(fraud_list)):
                if fraud_edges_added < num_fraud_edges and random.random() < fraud_cluster_density:
                    edges.append((fraud_list[i], fraud_list[j]))
                    edges.append((fraud_list[j], fraud_list[i]))  # Bidirectional
                    fraud_edges_added += 2
                    edges_added += 2
        
        # Add connections from fraud cluster to other nodes (guilt by association)
        num_connections = min(100, num_edges - edges_added)
        for _ in range(num_connections):
            fraud_node = random.choice(fraud_list)
            other_node = random.choice([n for n in range(num_nodes) if n not in fraud_nodes])
            edges.append((fraud_node, other_node))
            edges_added += 1
        
        # Fill remaining edges randomly (scale-free like)
        remaining_edges = num_edges - edges_added
        for _ in range(remaining_edges):
            source = random.randint(0, num_nodes - 1)
            target = random.randint(0, num_nodes - 1)
            if source != target:
                edges.append((source, target))
                edges_added += 1
        
        # Load edges into graph
        graph.load_from_edge_list(edges)
        
        return graph, fraud_nodes
    
    def generate_erdos_renyi_graph(self, num_nodes: int, edge_probability: float,
                                    fraud_cluster_size: int = 10) -> Tuple[SparseGraph, Set[int]]:
        """
        Generate an Erdos-Renyi random graph with fraud cluster.
        
        Args:
            num_nodes: Total number of nodes
            edge_probability: Probability of edge between any two nodes
            fraud_cluster_size: Size of fraud cluster
        
        Returns:
            Tuple of (SparseGraph, Set of fraud node IDs)
        """
        graph = SparseGraph(num_nodes=num_nodes)
        
        # Create fraud cluster
        fraud_nodes = set(random.sample(range(num_nodes), fraud_cluster_size))
        
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and random.random() < edge_probability:
                    edges.append((i, j))
        
        # Increase density within fraud cluster
        fraud_list = list(fraud_nodes)
        for i in range(len(fraud_list)):
            for j in range(i + 1, len(fraud_list)):
                if random.random() < 0.9:  # High density
                    edges.append((fraud_list[i], fraud_list[j]))
                    edges.append((fraud_list[j], fraud_list[i]))
        
        graph.load_from_edge_list(edges)
        return graph, fraud_nodes
    
    def save_edge_list(self, graph: SparseGraph, filename: str):
        """
        Save graph as edge list to file.
        
        Args:
            graph: SparseGraph instance
            filename: Output filename
        """
        edges = []
        for source in graph.adjacency_dict:
            for target in graph.adjacency_dict[source]:
                edges.append((source, target))
        
        with open(filename, 'w') as f:
            for source, target in edges:
                f.write(f"{source}\t{target}\n")
    
    def load_edge_list(self, filename: str) -> SparseGraph:
        """
        Load graph from edge list file.
        
        Args:
            filename: Input filename
        
        Returns:
            SparseGraph instance
        """
        graph = SparseGraph()
        edges = []
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        source = int(parts[0])
                        target = int(parts[1])
                        edges.append((source, target))
        
        graph.load_from_edge_list(edges)
        return graph
