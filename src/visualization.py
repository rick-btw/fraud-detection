"""
Visualization Module for Fraud Detection System.
Creates plots for convergence, scalability, and graph visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Set
import os


class FraudDetectionVisualizer:
    """
    Visualization toolkit for fraud detection analysis.
    """
    
    def __init__(self, output_dir: str = 'results'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                plt.style.use('default')
    
    def plot_convergence(self, convergence_history: List[float], 
                        title: str = "PPR Convergence", 
                        filename: str = "convergence.png"):
        """
        Plot convergence error vs iterations.
        
        Args:
            convergence_history: List of error values per iteration
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(10, 6))
        iterations = range(1, len(convergence_history) + 1)
        plt.semilogy(iterations, convergence_history, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('L1 Norm Error (log scale)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved convergence plot to {filepath}")
    
    def plot_scalability(self, node_sizes: List[int], runtimes: List[float],
                        title: str = "Scalability Analysis",
                        filename: str = "scalability.png"):
        """
        Plot runtime vs graph size.
        
        Args:
            node_sizes: List of graph sizes
            runtimes: List of corresponding runtimes
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(10, 6))
        plt.plot(node_sizes, runtimes, 'g-o', linewidth=2, markersize=8)
        plt.xlabel('Number of Nodes', fontsize=12)
        plt.ylabel('Runtime (seconds)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(node_sizes) > 1:
            z = np.polyfit(node_sizes, runtimes, 1)
            p = np.poly1d(z)
            plt.plot(node_sizes, p(node_sizes), "r--", alpha=0.5, label=f'Trend: y={z[0]:.2e}x+{z[1]:.2e}')
            plt.legend()
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved scalability plot to {filepath}")
    
    def plot_parameter_sensitivity(self, alpha_values: List[float],
                                  mean_scores: List[float],
                                  std_scores: List[float],
                                  title: str = "Parameter Sensitivity (Alpha)",
                                  filename: str = "parameter_sensitivity.png"):
        """
        Plot how alpha affects score distribution.
        
        Args:
            alpha_values: List of alpha values tested
            mean_scores: Mean scores for each alpha
            std_scores: Standard deviation of scores for each alpha
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(10, 6))
        plt.errorbar(alpha_values, mean_scores, yerr=std_scores, 
                    fmt='o-', linewidth=2, markersize=8, capsize=5)
        plt.xlabel('Alpha (Teleportation Probability)', fontsize=12)
        plt.ylabel('Mean Suspicion Score', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved parameter sensitivity plot to {filepath}")
    
    def plot_precision_at_k(self, k_values: List[int], precision: List[float],
                           title: str = "Precision@K Evaluation",
                           filename: str = "precision_at_k.png"):
        """
        Plot Precision@K curve.
        
        Args:
            k_values: List of K values
            precision: List of precision scores
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, precision, 'm-o', linewidth=2, markersize=8)
        plt.xlabel('K (Top K nodes)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.1])
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Precision@K plot to {filepath}")
    
    def plot_score_distribution(self, scores: np.ndarray,
                               title: str = "Suspicion Score Distribution",
                               filename: str = "score_distribution.png"):
        """
        Plot histogram of suspicion scores.
        
        Args:
            scores: Rank vector
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Suspicion Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved score distribution plot to {filepath}")
    
    def plot_method_comparison(self, scores_pi: np.ndarray, scores_mc: np.ndarray,
                              title: str = "Power Iteration vs Monte Carlo",
                              filename: str = "method_comparison.png"):
        """
        Scatter plot comparing two methods.
        
        Args:
            scores_pi: Scores from Power Iteration
            scores_mc: Scores from Monte Carlo
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(10, 8))
        plt.scatter(scores_pi, scores_mc, alpha=0.5, s=20)
        
        # Add diagonal line
        max_score = max(np.max(scores_pi), np.max(scores_mc))
        plt.plot([0, max_score], [0, max_score], 'r--', linewidth=2, label='Perfect Agreement')
        
        plt.xlabel('Power Iteration Scores', fontsize=12)
        plt.ylabel('Monte Carlo Scores', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved method comparison plot to {filepath}")
    
    def plot_graph_visualization(self, graph, seed_set: Set[int], scores: np.ndarray,
                                top_k: int = 50, title: str = "Fraud Detection Graph",
                                filename: str = "graph_visualization.png"):
        """
        Visualize graph with seed nodes and high-suspicion nodes.
        Note: For large graphs, this samples a subgraph.
        
        Args:
            graph: SparseGraph instance
            seed_set: Set of seed nodes (shown in red)
            scores: Rank vector
            top_k: Number of top nodes to highlight
            title: Plot title
            filename: Output filename
        """
        try:
            import networkx as nx
        except ImportError:
            print("NetworkX not available. Skipping graph visualization.")
            return
        
        # For large graphs, sample a subgraph
        num_nodes = graph.get_num_nodes()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        if num_nodes > 1000:
            # Sample nodes: all seeds + top K + random sample
            sample_nodes = set(seed_set) | set(top_indices[:top_k])
            # Add some random nodes for context
            remaining = set(range(num_nodes)) - sample_nodes
            if len(remaining) > 0:
                random_sample = np.random.choice(list(remaining), 
                                               size=min(200, len(remaining)), 
                                               replace=False)
                sample_nodes = sample_nodes | set(random_sample)
        else:
            sample_nodes = set(range(num_nodes))
        
        # Build NetworkX graph
        G = nx.DiGraph()
        sample_list = list(sample_nodes)
        
        # Add edges within sample
        for source in sample_list:
            neighbors = graph.get_neighbors(source)
            for target in neighbors:
                if target in sample_nodes:
                    G.add_edge(source, target)
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        # Node colors
        node_colors = []
        for node in G.nodes():
            if node in seed_set:
                node_colors.append('red')  # Seed nodes
            elif node in top_indices[:top_k]:
                node_colors.append('orange')  # High suspicion
            else:
                node_colors.append('lightblue')  # Normal
        
        # Plot
        plt.figure(figsize=(14, 10))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=100, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.2, arrows=True, 
                              arrowsize=10, edge_color='gray')
        
        # Legend
        red_patch = mpatches.Patch(color='red', label='Seed Nodes (Known Fraudsters)')
        orange_patch = mpatches.Patch(color='orange', label='High Suspicion Nodes')
        blue_patch = mpatches.Patch(color='lightblue', label='Other Nodes')
        plt.legend(handles=[red_patch, orange_patch, blue_patch], loc='upper right')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved graph visualization to {filepath}")
