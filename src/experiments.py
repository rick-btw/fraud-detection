"""
Main Experiment Script for Fraud Detection System.
Runs comprehensive evaluations and generates all plots.
"""

import numpy as np
from src.sparse_graph import SparseGraph
from src.pagerank import CustomPageRank, MonteCarloApprox
from src.data_generator import SyntheticDataGenerator
from src.analysis import FraudDetectionAnalyzer
from src.visualization import FraudDetectionVisualizer


def run_all_experiments():
    """Run comprehensive set of experiments."""
    print("=" * 60)
    print("Fraud Detection System - Comprehensive Experiments")
    print("=" * 60)
    
    # Initialize components
    generator = SyntheticDataGenerator(seed=42)
    analyzer = FraudDetectionAnalyzer()
    visualizer = FraudDetectionVisualizer(output_dir='results')
    
    # ========== Experiment 1: Generate Test Graph ==========
    print("\n[Experiment 1] Generating synthetic test graph...")
    num_nodes = 1000
    num_edges = 5000
    fraud_cluster_size = 20
    
    graph, fraud_nodes = generator.generate_scale_free_graph(
        num_nodes=num_nodes,
        num_edges=num_edges,
        fraud_cluster_size=fraud_cluster_size,
        fraud_cluster_density=0.7
    )
    
    print(f"Generated graph: {graph.get_num_nodes()} nodes, {graph.get_num_edges()} edges")
    print(f"Fraud cluster size: {len(fraud_nodes)} nodes")
    
    # Select seed set (subset of fraud nodes)
    seed_set = set(list(fraud_nodes)[:10])
    print(f"Seed set size: {len(seed_set)} nodes")
    
    # ========== Experiment 2: Run PPR and Plot Convergence ==========
    print("\n[Experiment 2] Running Personalized PageRank...")
    ppr = CustomPageRank(graph, alpha=0.15, epsilon=1e-6)
    scores = ppr.compute(seed_set)
    
    print(f"Converged in {ppr.get_iterations()} iterations")
    print(f"Top 10 suspicious nodes: {np.argsort(scores)[-10:][::-1]}")
    print(f"Max suspicion score: {np.max(scores):.6f}")
    
    # Plot convergence
    convergence_history = ppr.get_convergence_history()
    visualizer.plot_convergence(convergence_history)
    
    # ========== Experiment 3: Scalability Test ==========
    print("\n[Experiment 3] Running scalability test...")
    node_sizes = [500, 1000, 2000, 5000]
    scalability_results = analyzer.scalability_test(
        node_sizes=node_sizes,
        edges_per_node=5,
        alpha=0.15,
        seed_size=10
    )
    
    visualizer.plot_scalability(
        scalability_results['node_sizes'],
        scalability_results['runtimes']
    )
    
    # ========== Experiment 4: Parameter Sensitivity ==========
    print("\n[Experiment 4] Testing parameter sensitivity...")
    alpha_values = [0.1, 0.15, 0.2, 0.3, 0.5]
    sensitivity_results = analyzer.parameter_sensitivity(
        graph=graph,
        seed_set=seed_set,
        alpha_values=alpha_values
    )
    
    visualizer.plot_parameter_sensitivity(
        sensitivity_results['alpha_values'],
        sensitivity_results['mean_scores'],
        sensitivity_results['std_scores']
    )
    
    # ========== Experiment 5: Precision@K ==========
    print("\n[Experiment 5] Computing Precision@K...")
    precision_results = analyzer.precision_at_k(
        scores=scores,
        ground_truth=fraud_nodes,
        k_values=[10, 20, 50, 100, 200]
    )
    
    print("Precision@K results:")
    for k, prec in zip(precision_results['k_values'], precision_results['precision']):
        print(f"  Precision@{k}: {prec:.4f}")
    
    visualizer.plot_precision_at_k(
        precision_results['k_values'],
        precision_results['precision']
    )
    
    # ========== Experiment 6: Score Distribution ==========
    print("\n[Experiment 6] Plotting score distribution...")
    visualizer.plot_score_distribution(scores)
    
    # ========== Experiment 7: Method Comparison ==========
    print("\n[Experiment 7] Comparing Power Iteration vs Monte Carlo...")
    comparison_results = analyzer.compare_methods(
        graph=graph,
        seed_set=seed_set,
        ground_truth=fraud_nodes,
        alpha=0.15
    )
    
    print(f"Power Iteration runtime: {comparison_results['power_iteration']['runtime']:.4f}s")
    print(f"Monte Carlo runtime: {comparison_results['monte_carlo']['runtime']:.4f}s")
    print(f"Correlation: {comparison_results['correlation']:.4f}")
    
    visualizer.plot_method_comparison(
        comparison_results['scores_pi'],
        comparison_results['scores_mc']
    )
    
    # ========== Experiment 8: Graph Visualization ==========
    print("\n[Experiment 8] Creating graph visualization...")
    visualizer.plot_graph_visualization(
        graph=graph,
        seed_set=seed_set,
        scores=scores,
        top_k=30
    )
    
    print("\n" + "=" * 60)
    print("All experiments completed successfully!")
    print(f"Results saved to 'results/' directory")
    print("=" * 60)


if __name__ == "__main__":
    run_all_experiments()
