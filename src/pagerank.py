"""
Personalized PageRank (PPR) Implementation using Power Iteration.
Implements the mathematical formula: r^(t+1) = (1-α) * r^(t)M + α * p
"""

import numpy as np
from scipy import sparse
from typing import Set, List, Dict
import time


class CustomPageRank:
    """
    Personalized PageRank implementation using Power Iteration method.
    Detects "Guilt by Association" by propagating suspicion scores from seed nodes.
    """
    
    def __init__(self, graph, alpha: float = 0.15, epsilon: float = 1e-6):
        """
        Initialize PPR algorithm.
        
        Args:
            graph: SparseGraph instance
            alpha: Teleportation probability (damping factor), default 0.15
            epsilon: Convergence threshold for L1 norm, default 1e-6
        """
        self.graph = graph
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_nodes = graph.get_num_nodes()
        self.convergence_history = []
        self.iterations = 0
        
    def compute(self, seed_set: Set[int], handle_dangling: str = 'teleport_to_seeds',
                max_iterations: int = 1000) -> np.ndarray:
        """
        Compute Personalized PageRank scores using Power Iteration.
        
        Args:
            seed_set: Set of known fraudster node IDs (seed nodes)
            handle_dangling: Strategy for handling dangling nodes
            max_iterations: Maximum number of iterations to prevent infinite loops
        
        Returns:
            numpy.ndarray: Rank vector with suspicion scores for each node
        """
        if not seed_set:
            raise ValueError("Seed set cannot be empty")
        
        # Validate seed nodes
        valid_seeds = {s for s in seed_set if 0 <= s < self.num_nodes}
        if not valid_seeds:
            raise ValueError("No valid seed nodes in the specified range")
        
        # Build personalized teleportation vector
        p = np.zeros(self.num_nodes)
        for seed in valid_seeds:
            p[seed] = 1.0 / len(valid_seeds)
        
        # Get transition matrix
        M = self.graph.get_transition_matrix(handle_dangling=handle_dangling)
        
        # Handle dangling nodes in transition matrix if needed
        if handle_dangling == 'teleport_to_seeds':
            M = self._handle_dangling_teleport(M, valid_seeds)
        
        # Initialize rank vector uniformly
        r = np.ones(self.num_nodes) / self.num_nodes
        
        # Power Iteration: r^(t+1) = (1-α) * r^(t)M + α * p
        self.convergence_history = []
        self.iterations = 0
        
        for iteration in range(max_iterations):
            # Store previous rank vector
            r_prev = r.copy()
            
            # Matrix-vector multiplication: r^(t)M
            rM = r_prev @ M
            
            # Update: r^(t+1) = (1-α) * r^(t)M + α * p
            r = (1 - self.alpha) * rM + self.alpha * p
            
            # Compute L1 norm of difference
            error = np.linalg.norm(r - r_prev, ord=1)
            self.convergence_history.append(error)
            self.iterations = iteration + 1
            
            # Check convergence
            if error < self.epsilon:
                break
        
        return r
    
    def _handle_dangling_teleport(self, M: sparse.csr_matrix, seed_set: Set[int]) -> sparse.csr_matrix:
        """
        Handle dangling nodes by redistributing their mass to seed set.
        
        Args:
            M: Transition matrix
            seed_set: Set of seed nodes
        
        Returns:
            Modified transition matrix
        """
        dangling_nodes = self.graph.get_dangling_nodes()
        if not dangling_nodes:
            return M
        
        # Convert to COO format for easier modification
        M_coo = M.tocoo()
        rows = list(M_coo.row)
        cols = list(M_coo.col)
        data = list(M_coo.data)
        
        # For each dangling node, add edges to all seed nodes
        seed_list = list(seed_set)
        seed_prob = 1.0 / len(seed_list) if seed_list else 0
        
        for node in dangling_nodes:
            # Check if node already has outgoing edges
            node_has_edges = any(r == node for r in rows)
            
            if not node_has_edges:
                # Add teleportation to all seed nodes
                for seed in seed_list:
                    rows.append(node)
                    cols.append(seed)
                    data.append(seed_prob)
        
        # Reconstruct CSR matrix
        if rows:
            M_modified = sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(self.num_nodes, self.num_nodes)
            )
        else:
            M_modified = M
        
        return M_modified
    
    def get_convergence_history(self) -> List[float]:
        """Return the convergence error history."""
        return self.convergence_history.copy()
    
    def get_iterations(self) -> int:
        """Return the number of iterations until convergence."""
        return self.iterations


class MonteCarloApprox:
    """
    Monte Carlo approximation for Personalized PageRank.
    Uses random walks starting from seed nodes to approximate PPR scores.
    """
    
    def __init__(self, graph, alpha: float = 0.15):
        """
        Initialize Monte Carlo PPR.
        
        Args:
            graph: SparseGraph instance
            alpha: Teleportation probability (damping factor)
        """
        self.graph = graph
        self.alpha = alpha
        self.num_nodes = graph.get_num_nodes()
        np.random.seed(42)  # For reproducibility
    
    def compute(self, seed_set: Set[int], num_walks: int = 10000,
                max_walk_length: int = 100) -> np.ndarray:
        """
        Compute PPR scores using Monte Carlo random walks.
        
        Args:
            seed_set: Set of seed node IDs
            num_walks: Number of random walks to perform
            max_walk_length: Maximum length of each walk
        
        Returns:
            numpy.ndarray: Approximate rank vector
        """
        if not seed_set:
            raise ValueError("Seed set cannot be empty")
        
        valid_seeds = {s for s in seed_set if 0 <= s < self.num_nodes}
        if not valid_seeds:
            raise ValueError("No valid seed nodes")
        
        # Initialize visit counts
        visit_counts = np.zeros(self.num_nodes)
        seed_list = list(valid_seeds)
        
        # Perform random walks
        for _ in range(num_walks):
            # Start from a random seed
            current = np.random.choice(seed_list)
            visit_counts[current] += 1
            
            walk_length = 0
            while walk_length < max_walk_length:
                # With probability alpha, teleport back to seed
                if np.random.random() < self.alpha:
                    current = np.random.choice(seed_list)
                    visit_counts[current] += 1
                    walk_length += 1
                    continue
                
                # Otherwise, follow an outgoing edge
                neighbors = self.graph.get_neighbors(current)
                if not neighbors:
                    # Dangling node - teleport to seed
                    current = np.random.choice(seed_list)
                    visit_counts[current] += 1
                else:
                    # Random neighbor
                    current = np.random.choice(neighbors)
                    visit_counts[current] += 1
                
                walk_length += 1
        
        # Normalize to get probability distribution
        rank_vector = visit_counts / np.sum(visit_counts)
        return rank_vector
