"""
Experiments using the Bitcoin OTC trust network dataset.
Generates convergence and runtime plots using real-world data.
Dataset: https://snap.stanford.edu/data/soc-sign-bitcoinotc.html
"""

import numpy as np
import time
import os
from src.sparse_graph import SparseGraph
from src.pagerank import CustomPageRank
from src.visualization import FraudDetectionVisualizer


def run_bitcoin_experiments(csv_path: str = 'data/soc-sign-bitcoinotc.csv'):
    """
    Run experiments on Bitcoin OTC dataset and generate convergence and runtime plots.
    
    Args:
        csv_path: Path to the Bitcoin OTC CSV file (supports .csv and .csv.gz)
    """
    print("=" * 60)
    print("Bitcoin OTC Trust Network - Fraud Detection Experiments")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = FraudDetectionVisualizer(output_dir='results')
    
    # ========== Load Bitcoin OTC Dataset ==========
    print("\n[Step 1] Loading Bitcoin OTC dataset...")
    
    if not os.path.exists(csv_path):
        # Try .gz version
        csv_path_gz = csv_path + '.gz'
        if os.path.exists(csv_path_gz):
            csv_path = csv_path_gz
            print(f"Found compressed file: {csv_path}")
        else:
            print(f"Error: Dataset file not found: {csv_path}")
            print("Please download from: https://snap.stanford.edu/data/soc-sign-bitcoinotc.html")
            print("Expected location: data/soc-sign-bitcoinotc.csv or data/soc-sign-bitcoinotc.csv.gz")
            return
    
    print(f"Loading from: {csv_path}")
    
    # Load graph with negative ratings (distrust edges) for fraud detection
    graph = SparseGraph.load_from_bitcoin_otc_csv(csv_path, use_negative_ratings=True)
    
    print(f"Loaded graph: {graph.get_num_nodes()} nodes, {graph.get_num_edges()} edges")
    print(f"Dangling nodes: {len(graph.get_dangling_nodes())}")
    
    if graph.get_num_nodes() == 0 or graph.get_num_edges() == 0:
        print("Error: Graph is empty. Please check the dataset file.")
        return
    
    # ========== Select Seed Set ==========
    print("\n[Step 2] Selecting seed set (known fraudsters)...")
    
    # Strategy: Select nodes with highest in-degree of negative ratings
    # (nodes that are distrusted by many others)
    in_degree_negative = {}
    for source in graph.adjacency_dict:
        for target in graph.adjacency_dict[source]:
            in_degree_negative[target] = in_degree_negative.get(target, 0) + 1
    
    # Select top 20 most distrusted nodes as seed set
    seed_size = min(20, len(in_degree_negative))
    sorted_nodes = sorted(in_degree_negative.items(), key=lambda x: x[1], reverse=True)
    seed_set = set([node for node, degree in sorted_nodes[:seed_size]])
    
    print(f"Seed set size: {len(seed_set)} nodes")
    print(f"Seed nodes (most distrusted): {sorted(list(seed_set))[:10]}...")
    
    # ========== Experiment 1: Convergence Analysis ==========
    print("\n[Experiment 1] Running Personalized PageRank for convergence analysis...")
    
    ppr = CustomPageRank(graph, alpha=0.15, epsilon=1e-6)
    start_time = time.time()
    scores = ppr.compute(seed_set, max_iterations=1000)
    runtime = time.time() - start_time
    
    print(f"Converged in {ppr.get_iterations()} iterations")
    print(f"Runtime: {runtime:.4f} seconds")
    print(f"Max suspicion score: {np.max(scores):.6f}")
    print(f"Top 10 suspicious nodes: {np.argsort(scores)[-10:][::-1]}")
    
    # Plot convergence
    convergence_history = ppr.get_convergence_history()
    visualizer.plot_convergence(
        convergence_history,
        title="Bitcoin OTC Network - PPR Convergence",
        filename="bitcoin_convergence.png"
    )
    
    # ========== Experiment 2: Runtime Analysis ==========
    print("\n[Experiment 2] Running runtime analysis with different graph sizes...")
    
    # Create subgraphs of different sizes for scalability testing
    node_sizes = []
    runtimes = []
    iterations_list = []
    
    # Get all nodes that actually have edges
    nodes_with_edges = set()
    for source in graph.adjacency_dict:
        nodes_with_edges.add(source)
        for target in graph.adjacency_dict[source]:
            nodes_with_edges.add(target)
    
    all_nodes = sorted(list(nodes_with_edges))
    
    # Sample different sizes (up to full graph)
    max_size = min(len(all_nodes), 3000)  # Limit for reasonable runtime
    test_sizes = [min(500, len(all_nodes)), min(1000, len(all_nodes)), 
                  min(2000, len(all_nodes)), min(max_size, len(all_nodes))]
    test_sizes = sorted(list(set([s for s in test_sizes if s > 0])))
    
    for size in test_sizes:
        print(f"  Testing with {size} nodes...")
        
        # Create subgraph by sampling nodes
        np.random.seed(42)
        sampled_nodes = set(np.random.choice(all_nodes, size=size, replace=False))
        
        # Build subgraph with remapped node IDs (0 to size-1)
        subgraph = SparseGraph(num_nodes=size)
        node_mapping = {old_node: new_node for new_node, old_node in enumerate(sorted(sampled_nodes))}
        
        edges_added = 0
        for source in graph.adjacency_dict:
            if source in sampled_nodes:
                for target in graph.adjacency_dict[source]:
                    if target in sampled_nodes:
                        new_source = node_mapping[source]
                        new_target = node_mapping[target]
                        subgraph.add_edge(new_source, new_target)
                        edges_added += 1
        
        if edges_added == 0:
            print(f"    Warning: No edges in subgraph, skipping...")
            continue
        
        # Map seed set to subgraph
        subgraph_seed_set = set()
        for seed in seed_set:
            if seed in sampled_nodes:
                subgraph_seed_set.add(node_mapping[seed])
        
        if len(subgraph_seed_set) == 0:
            # If no seeds in subgraph, use nodes with highest in-degree
            subgraph_in_degree = {}
            for s in subgraph.adjacency_dict:
                for t in subgraph.adjacency_dict[s]:
                    subgraph_in_degree[t] = subgraph_in_degree.get(t, 0) + 1
            if subgraph_in_degree:
                top_node = max(subgraph_in_degree.items(), key=lambda x: x[1])[0]
                subgraph_seed_set = {top_node}
            else:
                # Use first node as seed
                subgraph_seed_set = {0}
        
        # Run PPR on subgraph
        subgraph_ppr = CustomPageRank(subgraph, alpha=0.15, epsilon=1e-6)
        start_time = time.time()
        subgraph_scores = subgraph_ppr.compute(subgraph_seed_set, max_iterations=1000)
        subgraph_runtime = time.time() - start_time
        
        node_sizes.append(size)
        runtimes.append(subgraph_runtime)
        iterations_list.append(subgraph_ppr.get_iterations())
        
        print(f"    Runtime: {subgraph_runtime:.4f}s, Iterations: {subgraph_ppr.get_iterations()}, Edges: {edges_added}")
    
    # Plot runtime scalability
    if len(node_sizes) > 0:
        visualizer.plot_scalability(
            node_sizes,
            runtimes,
            title="Bitcoin OTC Network - Runtime Scalability",
            filename="bitcoin_runtime.png"
        )
        
        print("\nRuntime Analysis Results:")
        for size, rt, it in zip(node_sizes, runtimes, iterations_list):
            print(f"  {size} nodes: {rt:.4f}s ({it} iterations)")
    
    print("\n" + "=" * 60)
    print("Bitcoin OTC experiments completed!")
    print("Generated plots:")
    print("  - results/bitcoin_convergence.png")
    print("  - results/bitcoin_runtime.png")
    print("=" * 60)


if __name__ == "__main__":
    # Default path - user can modify or download the dataset
    csv_path = 'data/soc-sign-bitcoinotc.csv'
    
    # Check if file exists, if not provide instructions
    if not os.path.exists(csv_path) and not os.path.exists(csv_path + '.gz'):
        print("=" * 60)
        print("Dataset not found!")
        print("=" * 60)
        print("\nPlease download the Bitcoin OTC dataset:")
        print("1. Visit: https://snap.stanford.edu/data/soc-sign-bitcoinotc.html")
        print("2. Download: soc-sign-bitcoinotc.csv.gz")
        print("3. Extract or place in: data/soc-sign-bitcoinotc.csv.gz")
        print("   (or extract to: data/soc-sign-bitcoinotc.csv)")
        print("\nAlternatively, run:")
        print("  cd data && wget https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz")
        print("=" * 60)
    else:
        run_bitcoin_experiments(csv_path)
