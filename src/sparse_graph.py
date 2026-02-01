"""
Sparse Graph Data Structure for efficient representation of large graphs.
Implements CSR (Compressed Sparse Row) format for O(V+E) space complexity.
"""

import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class SparseGraph:
    """
    Efficient sparse graph representation using CSR format.
    Handles edge lists and converts to row-normalized transition matrix.
    """
    
    def __init__(self, num_nodes: int = None):
        """
        Initialize an empty graph.
        
        Args:
            num_nodes: Optional pre-specified number of nodes
        """
        self.num_nodes = num_nodes
        self.adjacency_dict: Dict[int, List[int]] = defaultdict(list)
        self.out_degree: Dict[int, int] = defaultdict(int)
        self.in_degree: Dict[int, int] = defaultdict(int)
        self._transition_matrix = None
        self._dangling_nodes: Set[int] = set()
        
    def add_edge(self, source: int, target: int, weight: float = 1.0):
        """
        Add a directed edge from source to target.
        
        Args:
            source: Source node ID
            target: Target node ID
            weight: Edge weight (default 1.0)
        """
        if target not in self.adjacency_dict[source]:
            self.adjacency_dict[source].append(target)
            self.out_degree[source] += weight
            self.in_degree[target] += weight
            
        # Update num_nodes if needed
        max_node = max(source, target)
        if self.num_nodes is None:
            self.num_nodes = max_node + 1
        else:
            self.num_nodes = max(self.num_nodes, max_node + 1)
    
    def load_from_edge_list(self, edge_list: List[Tuple[int, int]]):
        """
        Load graph from a list of (source, target) tuples.
        
        Args:
            edge_list: List of (source, target) edge tuples
        """
        for source, target in edge_list:
            self.add_edge(source, target)
        self._update_dangling_nodes()
    
    def _update_dangling_nodes(self):
        """Identify nodes with no outgoing edges (dangling nodes)."""
        all_nodes = set()
        for source in self.adjacency_dict:
            all_nodes.add(source)
            for target in self.adjacency_dict[source]:
                all_nodes.add(target)
        
        self._dangling_nodes = all_nodes - set(self.adjacency_dict.keys())
        # Also check nodes with zero out-degree
        for node in all_nodes:
            if self.out_degree[node] == 0:
                self._dangling_nodes.add(node)
    
    def get_transition_matrix(self, handle_dangling: str = 'teleport_to_seeds') -> sparse.csr_matrix:
        """
        Build row-normalized transition matrix in CSR format.
        
        Args:
            handle_dangling: Strategy for dangling nodes:
                - 'teleport_to_seeds': Redistribute to seed set (requires seed set)
                - 'uniform': Redistribute uniformly to all nodes
                - 'self_loop': Add self-loop
        
        Returns:
            scipy.sparse.csr_matrix: Row-normalized transition matrix
        """
        if self._transition_matrix is not None:
            return self._transition_matrix
        
        if self.num_nodes is None:
            raise ValueError("Graph is empty. Add edges first.")
        
        # Build sparse matrix in COO format first
        rows = []
        cols = []
        data = []
        
        for source in self.adjacency_dict:
            if self.out_degree[source] > 0:
                # Normalize by out-degree
                for target in self.adjacency_dict[source]:
                    rows.append(source)
                    cols.append(target)
                    data.append(1.0 / self.out_degree[source])
        
        # Handle dangling nodes
        if handle_dangling == 'uniform':
            # Each dangling node teleports uniformly to all nodes
            uniform_prob = 1.0 / self.num_nodes
            for node in self._dangling_nodes:
                for target in range(self.num_nodes):
                    rows.append(node)
                    cols.append(target)
                    data.append(uniform_prob)
        elif handle_dangling == 'self_loop':
            # Add self-loop for dangling nodes
            for node in self._dangling_nodes:
                rows.append(node)
                cols.append(node)
                data.append(1.0)
        
        # Convert to CSR format
        if rows:
            self._transition_matrix = sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(self.num_nodes, self.num_nodes)
            )
        else:
            # Empty graph - create zero matrix
            self._transition_matrix = sparse.csr_matrix(
                (self.num_nodes, self.num_nodes)
            )
        
        return self._transition_matrix
    
    def get_dangling_nodes(self) -> Set[int]:
        """Return set of dangling nodes (nodes with no outgoing edges)."""
        return self._dangling_nodes.copy()
    
    def get_num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return self.num_nodes if self.num_nodes else 0
    
    def get_num_edges(self) -> int:
        """Return the number of edges in the graph."""
        return sum(len(targets) for targets in self.adjacency_dict.values())
    
    def get_neighbors(self, node: int) -> List[int]:
        """Get list of neighbors (outgoing edges) for a node."""
        return self.adjacency_dict.get(node, []).copy()
    
    def has_edge(self, source: int, target: int) -> bool:
        """Check if an edge exists from source to target."""
        return target in self.adjacency_dict.get(source, [])
    
    @staticmethod
    def load_from_bitcoin_otc_csv(csv_path: str, use_negative_ratings: bool = True) -> 'SparseGraph':
        """
        Load graph from Bitcoin OTC trust network CSV file.
        
        The Bitcoin OTC dataset contains trust ratings from -10 (total distrust) to +10 (total trust).
        For fraud detection, we can focus on negative ratings (distrust) as indicators of potential fraud.
        
        Args:
            csv_path: Path to the CSV file (can be .csv or .csv.gz)
            use_negative_ratings: If True, only include edges with negative ratings (distrust).
                                 If False, include all edges regardless of rating.
        
        Returns:
            SparseGraph instance loaded from the CSV file
        
        Dataset format: SOURCE, TARGET, RATING, TIME
        Reference: https://snap.stanford.edu/data/soc-sign-bitcoinotc.html
        """
        import csv
        import gzip
        
        graph = SparseGraph()
        edges = []
        node_set = set()
        
        # Handle both .csv and .csv.gz files
        if csv_path.endswith('.gz'):
            file_handle = gzip.open(csv_path, 'rt', encoding='utf-8')
        else:
            file_handle = open(csv_path, 'r', encoding='utf-8')
        
        try:
            reader = csv.reader(file_handle)
            for row in reader:
                if len(row) < 3:
                    continue
                
                try:
                    source = int(row[0])
                    target = int(row[1])
                    rating = int(row[2])
                    
                    # Filter based on rating if needed
                    if use_negative_ratings:
                        # Only include negative ratings (distrust) for fraud detection
                        if rating < 0:
                            edges.append((source, target))
                            node_set.add(source)
                            node_set.add(target)
                    else:
                        # Include all edges
                        edges.append((source, target))
                        node_set.add(source)
                        node_set.add(target)
                except (ValueError, IndexError):
                    # Skip invalid rows
                    continue
        
        finally:
            file_handle.close()
        
        # Set num_nodes to max node ID + 1
        if node_set:
            graph.num_nodes = max(node_set) + 1
        else:
            graph.num_nodes = 0
        
        # Load edges
        graph.load_from_edge_list(edges)
        
        return graph
    
    @staticmethod
    def load_from_edgelist_file(edgelist_path: str, directed: bool = True, 
                                 has_weights: bool = False) -> 'SparseGraph':
        """
        Load graph from edge list file (common format for Network Repository datasets).
        
        Supports standard edge list formats:
        - source target
        - source target weight
        - source target weight (with comments starting with #)
        
        Args:
            edgelist_path: Path to the edge list file (can be .txt, .edgelist, or .gz)
            directed: If True, treat edges as directed. If False, add both directions.
            has_weights: If True, expect weight column (will be ignored, edges treated as unweighted)
        
        Returns:
            SparseGraph instance loaded from the edge list file
        
        Reference: https://networkrepository.com/
        """
        import gzip
        
        graph = SparseGraph()
        edges = []
        node_set = set()
        
        # Handle both compressed and uncompressed files
        if edgelist_path.endswith('.gz'):
            file_handle = gzip.open(edgelist_path, 'rt', encoding='utf-8')
        else:
            file_handle = open(edgelist_path, 'r', encoding='utf-8')
        
        try:
            for line in file_handle:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse line (space or tab separated)
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                try:
                    source = int(parts[0])
                    target = int(parts[1])
                    
                    # Add edge(s)
                    edges.append((source, target))
                    node_set.add(source)
                    node_set.add(target)
                    
                    # If undirected, add reverse edge
                    if not directed:
                        edges.append((target, source))
                        node_set.add(target)
                        node_set.add(source)
                
                except (ValueError, IndexError):
                    # Skip invalid lines
                    continue
        
        finally:
            file_handle.close()
        
        # Set num_nodes to max node ID + 1
        if node_set:
            graph.num_nodes = max(node_set) + 1
        else:
            graph.num_nodes = 0
        
        # Load edges
        graph.load_from_edge_list(edges)
        
        return graph
