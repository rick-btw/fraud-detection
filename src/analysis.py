"""
Analysis and Evaluation Module for Fraud Detection System.
Implements scalability tests, parameter sensitivity, and Precision@K metrics.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Set
from src.sparse_graph import SparseGraph
from src.pagerank import CustomPageRank, MonteCarloApprox
from src.data_generator import SyntheticDataGenerator


class FraudDetectionAnalyzer:
    """
    Comprehensive analysis toolkit for evaluating fraud detection performance.
    """
    
    def __init__(self):
        """Initialize analyzer."""
        self.results = {}
    
    def scalability_test(self, node_sizes: List[int], edges_per_node: int = 5,
                         alpha: float = 0.15, seed_size: int = 10) -> Dict:
        """
        Measure runtime vs. graph size.
        
        Args:
            node_sizes: List of graph sizes to test (e.g., [1000, 5000, 10000])
            edges_per_node: Average number of edges per node
            alpha: Teleportation probability
            seed_size: Size of seed set
        
        Returns:
            Dictionary with node_sizes, runtimes, and iterations
        """
        generator = SyntheticDataGenerator()
        results = {
            'node_sizes': [],
            'runtimes': [],
            'iterations': [],
            'num_edges': []
        }
        
        for num_nodes in node_sizes:
            print(f"Testing scalability with {num_nodes} nodes...")
            num_edges = num_nodes * edges_per_node
            
            # Generate graph
            graph, fraud_nodes = generator.generate_scale_free_graph(
                num_nodes=num_nodes,
                num_edges=num_edges,
                fraud_cluster_size=seed_size
            )
            
            # Select seed set (subset of fraud nodes)
            seed_set = set(list(fraud_nodes)[:seed_size])
            
            # Run PPR
            ppr = CustomPageRank(graph, alpha=alpha)
            start_time = time.time()
            scores = ppr.compute(seed_set)
            runtime = time.time() - start_time
            
            results['node_sizes'].append(num_nodes)
            results['runtimes'].append(runtime)
            results['iterations'].append(ppr.get_iterations())
            results['num_edges'].append(graph.get_num_edges())
        
        self.results['scalability'] = results
        return results
    
    def parameter_sensitivity(self, graph: SparseGraph, seed_set: Set[int],
                             alpha_values: List[float] = [0.1, 0.15, 0.3]) -> Dict:
        """
        Test how changing alpha affects score distribution.
        
        Args:
            graph: SparseGraph instance
            seed_set: Set of seed nodes
            alpha_values: List of alpha values to test
        
        Returns:
            Dictionary with alpha values and their score statistics
        """
        results = {
            'alpha_values': [],
            'mean_scores': [],
            'std_scores': [],
            'max_scores': [],
            'top_k_scores': []
        }
        
        for alpha in alpha_values:
            print(f"Testing alpha = {alpha}...")
            ppr = CustomPageRank(graph, alpha=alpha)
            scores = ppr.compute(seed_set)
            
            results['alpha_values'].append(alpha)
            results['mean_scores'].append(np.mean(scores))
            results['std_scores'].append(np.std(scores))
            results['max_scores'].append(np.max(scores))
            # Top 10 scores
            top_k = np.sort(scores)[-10:][::-1]
            results['top_k_scores'].append(top_k.tolist())
        
        self.results['parameter_sensitivity'] = results
        return results
    
    def precision_at_k(self, scores: np.ndarray, ground_truth: Set[int],
                       k_values: List[int] = [10, 20, 50, 100]) -> Dict:
        """
        Compute Precision@K metric.
        
        Args:
            scores: Rank vector from PPR
            ground_truth: Set of actual fraudster node IDs
            k_values: List of K values to evaluate
        
        Returns:
            Dictionary with K values and their precision scores
        """
        # Get top K nodes by score
        top_indices = np.argsort(scores)[::-1]  # Descending order
        
        results = {
            'k_values': [],
            'precision': [],
            'recall': []
        }
        
        for k in k_values:
            top_k_nodes = set(top_indices[:k])
            true_positives = len(top_k_nodes & ground_truth)
            precision = true_positives / k if k > 0 else 0.0
            recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0.0
            
            results['k_values'].append(k)
            results['precision'].append(precision)
            results['recall'].append(recall)
        
        self.results['precision_at_k'] = results
        return results
    
    def compare_methods(self, graph: SparseGraph, seed_set: Set[int],
                       ground_truth: Set[int], alpha: float = 0.15) -> Dict:
        """
        Compare Power Iteration vs Monte Carlo approximation.
        
        Args:
            graph: SparseGraph instance
            seed_set: Set of seed nodes
            ground_truth: Set of actual fraud nodes
            alpha: Teleportation probability
        
        Returns:
            Dictionary with comparison metrics
        """
        # Power Iteration
        print("Running Power Iteration...")
        ppr = CustomPageRank(graph, alpha=alpha)
        start_time = time.time()
        scores_pi = ppr.compute(seed_set)
        time_pi = time.time() - start_time
        
        # Monte Carlo
        print("Running Monte Carlo approximation...")
        mc = MonteCarloApprox(graph, alpha=alpha)
        start_time = time.time()
        scores_mc = mc.compute(seed_set, num_walks=10000)
        time_mc = time.time() - start_time
        
        # Compare scores using correlation
        correlation = np.corrcoef(scores_pi, scores_mc)[0, 1]
        
        # Precision@K comparison
        precision_pi = self.precision_at_k(scores_pi, ground_truth, k_values=[20])
        precision_mc = self.precision_at_k(scores_mc, ground_truth, k_values=[20])
        
        results = {
            'power_iteration': {
                'runtime': time_pi,
                'iterations': ppr.get_iterations(),
                'precision@20': precision_pi['precision'][0]
            },
            'monte_carlo': {
                'runtime': time_mc,
                'precision@20': precision_mc['precision'][0]
            },
            'correlation': correlation,
            'scores_pi': scores_pi,
            'scores_mc': scores_mc
        }
        
        self.results['method_comparison'] = results
        return results
