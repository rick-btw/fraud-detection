"""
Basic test script to verify the implementation works correctly.
Run this after installing dependencies: pip install -r requirements.txt
"""

import numpy as np
from src.sparse_graph import SparseGraph
from src.pagerank import CustomPageRank
from src.data_generator import SyntheticDataGenerator


def test_basic_functionality():
    """Test basic PPR functionality."""
    print("Testing basic functionality...")
    
    # Create a simple graph manually
    graph = SparseGraph(num_nodes=5)
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(4, 0)
    
    print(f"Graph: {graph.get_num_nodes()} nodes, {graph.get_num_edges()} edges")
    
    # Test PPR
    seed_set = {0}
    ppr = CustomPageRank(graph, alpha=0.15, epsilon=1e-6)
    scores = ppr.compute(seed_set)
    
    print(f"Converged in {ppr.get_iterations()} iterations")
    print(f"Scores: {scores}")
    print(f"Sum of scores: {np.sum(scores):.6f} (should be ~1.0)")
    
    # Test with synthetic data
    print("\nTesting synthetic data generation...")
    generator = SyntheticDataGenerator(seed=42)
    graph2, fraud_nodes = generator.generate_scale_free_graph(
        num_nodes=100,
        num_edges=200,
        fraud_cluster_size=10
    )
    
    print(f"Generated graph: {graph2.get_num_nodes()} nodes, {graph2.get_num_edges()} edges")
    print(f"Fraud cluster: {len(fraud_nodes)} nodes")
    
    seed_set2 = set(list(fraud_nodes)[:5])
    ppr2 = CustomPageRank(graph2, alpha=0.15)
    scores2 = ppr2.compute(seed_set2)
    
    top_5 = np.argsort(scores2)[-5:][::-1]
    print(f"Top 5 suspicious nodes: {top_5}")
    print(f"Converged in {ppr2.get_iterations()} iterations")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_basic_functionality()
