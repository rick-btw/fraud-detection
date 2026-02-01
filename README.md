# Fraud Detection System using Graph Analysis and Personalized PageRank

A comprehensive implementation of a fraud detection system based on "Guilt by Association" using Personalized PageRank (PPR) from scratch. This project implements the Power Iteration method for PPR computation with efficient sparse matrix representations.

## Project Overview

This system detects fraudulent entities by propagating suspicion scores from a seed set of known fraudsters through a network graph. The implementation focuses on:

- **Algorithmic Correctness**: Rigorous mathematical implementation of PPR
- **Memory Efficiency**: Sparse matrix representations (CSR format) for O(V+E) complexity
- **Edge Case Handling**: Dangling nodes and disconnected components
- **Comprehensive Analysis**: Scalability tests, parameter sensitivity, and evaluation metrics

## Features

- **Custom PageRank Implementation**: Power Iteration method with convergence criterion
- **Sparse Graph Engine**: Efficient CSR-based graph representation
- **Monte Carlo Approximation**: Alternative random-walk based method (bonus feature)
- **Comprehensive Experiments**: Scalability, parameter sensitivity, Precision@K
- **Visualization**: Convergence plots, scalability analysis, graph visualizations

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Dependencies

Install required packages:

```bash
pip install numpy scipy matplotlib networkx
```

Or install from requirements file (if provided):

```bash
pip install -r requirements.txt
```

## Project Structure

```
FraudDetectionC01/
├── src/
│   ├── __init__.py
│   ├── sparse_graph.py          # Sparse graph data structure (CSR)
│   ├── pagerank.py              # PPR implementation (Power Iteration + Monte Carlo)
│   ├── data_generator.py        # Synthetic data generator
│   ├── analysis.py              # Analysis and evaluation tools
│   ├── visualization.py         # Plotting and visualization
│   └── experiments.py           # Main experiment script
├── data/                        # Dataset directory
├── results/                     # Output plots and results
├── README.md                    # This file
└── REPORT_DRAFT.md              # Detailed report structure
```

## Usage

### Quick Start

Run all experiments:

```bash
python -m src.experiments
```

This will:
1. Generate a synthetic test graph
2. Run Personalized PageRank
3. Perform scalability tests
4. Test parameter sensitivity
5. Compute Precision@K metrics
6. Generate all visualizations

### Basic Usage

```python
from src.sparse_graph import SparseGraph
from src.pagerank import CustomPageRank
from src.data_generator import SyntheticDataGenerator

# Generate synthetic graph
generator = SyntheticDataGenerator(seed=42)
graph, fraud_nodes = generator.generate_scale_free_graph(
    num_nodes=1000,
    num_edges=5000,
    fraud_cluster_size=20
)

# Select seed set (known fraudsters)
seed_set = set(list(fraud_nodes)[:10])

# Run PPR
ppr = CustomPageRank(graph, alpha=0.15, epsilon=1e-6)
scores = ppr.compute(seed_set)

# Get top suspicious nodes
top_k = 10
top_indices = np.argsort(scores)[-top_k:][::-1]
print(f"Top {top_k} suspicious nodes: {top_indices}")
```

### Loading Custom Data

```python
from src.sparse_graph import SparseGraph
from src.data_generator import SyntheticDataGenerator

# Load from edge list file
generator = SyntheticDataGenerator()
graph = generator.load_edge_list('data/my_graph.edgelist')

# Load Bitcoin OTC dataset (real-world fraud detection data)
# Download from: https://snap.stanford.edu/data/soc-sign-bitcoinotc.html
graph = SparseGraph.load_from_bitcoin_otc_csv('data/soc-sign-bitcoinotc.csv.gz', 
                                              use_negative_ratings=True)

# Or build manually
graph = SparseGraph(num_nodes=1000)
graph.add_edge(0, 1)
graph.add_edge(1, 2)
# ... add more edges
```

### Running Bitcoin OTC Experiments

The project includes a script to run experiments on the real-world Bitcoin OTC trust network dataset:

```bash
# First, download the dataset
cd data
wget https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz

# Then run the experiments
python -m src.bitcoin_experiments
```

This will generate:
- `results/bitcoin_convergence.png`: Convergence analysis on real data
- `results/bitcoin_runtime.png`: Runtime scalability on real data

### Running Caltech36 Facebook Experiments

The project includes a script to run experiments on the Caltech36 Facebook social network dataset:

```bash
# First, download the dataset
./download_caltech36_dataset.sh

# Or manually:
cd data
wget https://nrvis.com/download/data/social/socfb-Caltech36.zip
unzip socfb-Caltech36.zip

# Then run the experiments
python -m src.caltech36_experiments
```

This will generate:
- `results/caltech36_convergence.png`: Convergence analysis on real data
- `results/caltech36_runtime.png`: Runtime scalability on real data

### Running Individual Experiments

```python
from src.analysis import FraudDetectionAnalyzer
from src.visualization import FraudDetectionVisualizer

analyzer = FraudDetectionAnalyzer()
visualizer = FraudDetectionVisualizer()

# Scalability test
results = analyzer.scalability_test(
    node_sizes=[1000, 5000, 10000],
    edges_per_node=5
)

# Parameter sensitivity
sensitivity = analyzer.parameter_sensitivity(
    graph=graph,
    seed_set=seed_set,
    alpha_values=[0.1, 0.15, 0.3]
)

# Generate plots
visualizer.plot_scalability(results['node_sizes'], results['runtimes'])
visualizer.plot_parameter_sensitivity(
    sensitivity['alpha_values'],
    sensitivity['mean_scores'],
    sensitivity['std_scores']
)
```

## Mathematical Background

### Personalized PageRank Formula

The system implements the Power Iteration method:

$$r^{(t+1)} = (1 - \alpha) \cdot r^{(t)}M + \alpha \cdot p$$

Where:
- $r$: Rank vector (suspicion scores)
- $M$: Row-normalized transition matrix
- $\alpha$: Teleportation probability (damping factor), default 0.15
- $p$: Personalized teleportation vector (non-zero only for seed set)

### Convergence Criterion

The algorithm converges when:

$$||r^{(t+1)} - r^{(t)}||_1 < \epsilon$$

Where $\epsilon = 10^{-6}$ by default.

## Key Implementation Details

### Sparse Matrix Representation

- Uses **CSR (Compressed Sparse Row)** format for O(V+E) space complexity
- Efficient matrix-vector multiplication for Power Iteration
- Handles graphs with millions of nodes efficiently

### Dangling Node Handling

Three strategies implemented:
1. **Teleport to Seeds**: Redistribute mass to seed set (default)
2. **Uniform**: Redistribute uniformly to all nodes
3. **Self-loop**: Add self-loop for dangling nodes

### Disconnected Components

The algorithm handles disconnected graphs by:
- Ensuring all nodes are reachable through teleportation
- Maintaining probability mass conservation
- No crashes on isolated components

## Output

All plots and results are saved to the `results/` directory:

- `convergence.png`: Convergence error vs iterations
- `scalability.png`: Runtime vs graph size
- `parameter_sensitivity.png`: Effect of alpha on scores
- `precision_at_k.png`: Precision@K evaluation
- `score_distribution.png`: Histogram of suspicion scores
- `method_comparison.png`: Power Iteration vs Monte Carlo
- `graph_visualization.png`: Graph plot with seed/high-suspicion nodes

## Performance

- **Time Complexity**: O(k(V+E)) where k is iterations until convergence
- **Space Complexity**: O(V+E) using sparse matrices
- **Typical Convergence**: 20-50 iterations for graphs with 1K-10K nodes

## Limitations

1. **Cold Start Problem**: Requires initial seed set of known fraudsters
2. **Parameter Tuning**: Alpha value needs tuning for different graph structures
3. **Scalability**: Very large graphs (>1M nodes) may require distributed computing
4. **Graph Structure**: Assumes fraudsters form connected clusters

## Future Improvements

- Distributed computation for very large graphs
- Adaptive alpha selection
- Incremental updates for dynamic graphs
- Additional graph metrics (betweenness, closeness centrality)

## License

This project is for educational purposes (Data Structures course).

## Author

Senior Data Scientist and Algorithm Engineer

## References

- PageRank algorithm (Brin & Page, 1998)
- Personalized PageRank (Haveliwala, 2002)
- Sparse matrix representations (CSR format)
