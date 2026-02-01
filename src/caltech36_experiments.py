"""
Experiments using the Caltech36 Facebook social network dataset.
Generates convergence and runtime plots using real-world data.
Dataset: https://networkrepository.com/socfb-Caltech36.php
"""

import numpy as np
import time
import os
from src.sparse_graph import SparseGraph
from src.pagerank import CustomPageRank
from src.visualization import FraudDetectionVisualizer


def run_caltech36_experiments(edgelist_path: str = 'data/socfb-Caltech36.edges'):
    """
    Run experiments on Caltech36 Facebook dataset and generate convergence and runtime plots.
    
    Args:
        edgelist_path: Path to the Caltech36 edge list file (supports .txt, .edgelist, .edges, or .gz)
    """
    print("=" * 60)
    print("Caltech36 Facebook Network - Fraud Detection Experiments")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = FraudDetectionVisualizer(output_dir='results')
    
    # ========== Load Caltech36 Dataset ==========
    print("\n[Step 1] Loading Caltech36 Facebook dataset...")
    
    # Try different possible file extensions
    possible_paths = [
        edgelist_path,
        edgelist_path + '.txt',
        edgelist_path + '.edgelist',
        edgelist_path + '.edges',
        edgelist_path + '.gz',
        edgelist_path + '.txt.gz',
        edgelist_path + '.edgelist.gz',
    ]
    
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break
    
    if not found_path:
        print(f"Error: Dataset file not found. Tried:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease download the Caltech36 dataset:")
        print("1. Visit: https://networkrepository.com/socfb-Caltech36.php")
        print("2. Download the edge list file")
        print("3. Place in: data/socfb-Caltech36.edges (or .txt, .edgelist)")
        print("\nAlternatively, run:")
        print("  cd data && wget https://nrvis.com/download/data/social/socfb-Caltech36.zip")
        print("  unzip socfb-Caltech36.zip")
        print("=" * 60)
        return
    
    print(f"Loading from: {found_path}")
    
    # Load graph (Facebook networks are typically undirected)
    graph = SparseGraph.load_from_edgelist_file(found_path, directed=False, has_weights=False)
    
    print(f"Loaded graph: {graph.get_num_nodes()} nodes, {graph.get_num_edges()} edges")
    print(f"Dangling nodes: {len(graph.get_dangling_nodes())}")
    
    if graph.get_num_nodes() == 0 or graph.get_num_edges() == 0:
        print("Error: Graph is empty. Please check the dataset file.")
        return
    
    # ========== Select Seed Set ==========
    print("\n[Step 2] Selecting seed set (known fraudsters)...")
    
    # Strategy: Select nodes with highest degree (most connected nodes)
    # In social networks, highly connected nodes might be suspicious (e.g., fake accounts)
    node_degree = {}
    for source in graph.adjacency_dict:
        node_degree[source] = len(graph.get_neighbors(source))
        # Also count in-degree for nodes that appear as targets
        for target in graph.adjacency_dict[source]:
            if target not in node_degree:
                node_degree[target] = 0
            node_degree[target] += 1
    
    # Select top 20 most connected nodes as seed set
    seed_size = min(20, len(node_degree))
    sorted_nodes = sorted(node_degree.items(), key=lambda x: x[1], reverse=True)
    seed_set = set([node for node, degree in sorted_nodes[:seed_size]])
    
    print(f"Seed set size: {len(seed_set)} nodes")
    print(f"Seed nodes (highest degree): {sorted(list(seed_set))[:10]}...")
    print(f"Average degree of seed nodes: {np.mean([node_degree[n] for n in seed_set]):.2f}")
    
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
        title="Caltech36 Facebook Network - PPR Convergence",
        filename="caltech36_convergence.png"
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
    max_size = min(len(all_nodes), 2000)  # Limit for reasonable runtime
    test_sizes = [min(200, len(all_nodes)), min(500, len(all_nodes)), 
                  min(1000, len(all_nodes)), min(max_size, len(all_nodes))]
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
            title="Caltech36 Facebook Network - Runtime Scalability",
            filename="caltech36_runtime.png"
        )
        
        print("\nRuntime Analysis Results:")
        for size, rt, it in zip(node_sizes, runtimes, iterations_list):
            print(f"  {size} nodes: {rt:.4f}s ({it} iterations)")
    
    print("\n" + "=" * 60)
    print("Caltech36 Facebook experiments completed!")
    print("Generated plots:")
    print("  - results/caltech36_convergence.png")
    print("  - results/caltech36_runtime.png")
    print("=" * 60)


if __name__ == "__main__":
    # Default path - user can modify or download the dataset
    edgelist_path = 'data/socfb-Caltech36.edges'
    
    # Check if file exists, if not provide instructions
    run_caltech36_experiments(edgelist_path)
