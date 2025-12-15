"""
measures.py

This module provides network analysis tools for computing various geometric and topological
measures on networks. It includes methods for calculating effective resistance, 
graph mass, and different types of curvatures (Forman, Ollivier-Ricci).

Key Features:
- Effective resistance computation using sparse matrices
- Graph mass calculation
- Multiple curvature measures (Forman, Ollivier-Ricci)
- Node-based curvature aggregation
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from typing import Union, Tuple, Optional, List

class Network_Measures:
    """
    A class containing methods for computing various geometric and topological 
    measures on networks.
    
    This class provides tools for analyzing network structure through different
    mathematical lenses, including effective resistance, graph mass, and various
    curvature measures.
    """
    
    def __init__(self):
        """Initialize the Network_Measures class."""
        pass

    def compute_graph_mass(self, edge_list: np.ndarray) -> float:
        """
        Compute the total mass of a graph based on edge weights.
        
        The mass is computed as the sum of reciprocal edge weights.
        
        Parameters
        ----------
        edge_list : np.ndarray
            Array of shape (m, 3) where each row is [node1, node2, weight]
            
        Returns
        -------
        float
            The total mass of the graph
            
        Examples
        --------
        >>> edges = np.array([[0, 1, 2.0], [1, 2, 3.0]])
        >>> measures = Network_Measures()
        >>> mass = measures.compute_graph_mass(edges)
        """
        weights = 1 / np.array(edge_list[:,2])
        return np.sum(weights)

    def compute_effective_resistance(self, edge_list: np.ndarray) -> float:
        """
        Compute the Total Effective Resistance of a graph using sparse matrices.
        
        This method uses efficient sparse matrix operations to compute the 
        effective resistance, which measures the overall connectivity of the network.
        
        Parameters
        ----------
        edge_list : np.ndarray
            Array of shape (m, 3) where each row contains [node1, node2, weight]
            Node indices should be 0-based and contiguous
            
        Returns
        -------
        float
            The Total Effective Resistance of the graph
            
        Raises
        ------
        ValueError
            If the graph is not connected or edge_list format is invalid
        MemoryError
            If the graph is too large for dense matrix operations
            
        Notes
        -----
        The effective resistance is computed using the pseudoinverse of the 
        graph Laplacian. For very large graphs, this can be computationally intensive.
        """
        if edge_list.ndim != 2 or edge_list.shape[1] != 3:
            raise ValueError("edge_list must be a 2D array with shape (m, 3)")

        # Extract nodes and compute weights (inverse of edge length)
        node1 = edge_list[:, 0].astype(int)
        node2 = edge_list[:, 1].astype(int)
        weights = 1 / edge_list[:, 2]
        num_nodes = max(node1.max(), node2.max()) + 1

        # Construct sparse adjacency matrix
        adj = sp.coo_matrix(
            (weights, (node1, node2)),
            shape=(num_nodes, num_nodes)
        )
        
        # Make adjacency matrix symmetric
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj.tocsr()

        # Compute degree matrix and Laplacian
        degrees = np.array(adj.sum(axis=1)).flatten()
        laplacian = sp.diags(degrees) - adj

        # Verify graph connectivity using breadth-first search instead of eigenvalues
        G = nx.from_scipy_sparse_array(adj)
        if not nx.is_connected(G):
            raise ValueError("Graph is not connected")

        # Compute pseudoinverse and effective resistance
        try:
            L_dense = laplacian.toarray()
            eigenvals, eigenvecs = np.linalg.eigh(L_dense)
            non_zero_eigenvals = eigenvals[1:]  # Exclude zero eigenvalue
            trace_L_pseudo = np.sum(1.0 / non_zero_eigenvals)
            R_total = num_nodes * trace_L_pseudo
            return R_total
            
        except MemoryError:
            raise MemoryError("Graph too large for dense matrix operations")
        except Exception as e:
            raise RuntimeError(f"Error in computation: {e}")

    def forman_ricci_curvatures(self, edge_array: np.ndarray) -> List[float]:
        """
        Compute Forman-Ricci curvature for each edge in the graph.
        
        Parameters
        ----------
        edge_array : np.ndarray
            Array of shape (m, 3) containing [node1, node2, weight] for each edge
            
        Returns
        -------
        List[float]
            List of Forman-Ricci curvatures for each edge
        """
        edges = edge_array
        node_edges = {}
        
        # Build edge adjacency dictionary
        for idx, edge in enumerate(edges):
            v1, v2, _ = edge
            if v1 not in node_edges:
                node_edges[v1] = []
            if v2 not in node_edges:
                node_edges[v2] = []
            node_edges[v1].append(idx)
            node_edges[v2].append(idx)

        # Compute curvatures
        curvatures = []
        for idx, edge in enumerate(edges):
            v1, v2, w_e = edge
            w_e = float(w_e)
            
            # Compute contributions from adjacent edges
            sum_v1 = sum(1.0 / (w_e * float(edges[e_idx][2])) ** 0.5 
                        for e_idx in node_edges[v1] if e_idx != idx)
            sum_v2 = sum(1.0 / (w_e * float(edges[e_idx][2])) ** 0.5 
                        for e_idx in node_edges[v2] if e_idx != idx)

            curvature = 2 - w_e * (sum_v1 + sum_v2)
            curvatures.append(curvature)

        return curvatures

    def node_curvatures(self, edge_array: np.ndarray) -> np.ndarray:
        """
        Compute curvature values for each node by averaging incident edge curvatures.
        
        Parameters
        ----------
        edge_array : np.ndarray
            Array of shape (m, 3) containing [node1, node2, weight] for each edge
            
        Returns
        -------
        np.ndarray
            Array of node curvatures indexed by node ID
        """
        # Find maximum node index
        max_node = int(max(edge_array[:, :2].max(), 0))
        curvature_array = np.zeros(max_node + 1, dtype=np.float64)
        
        # Build node-edge adjacency dictionary
        node_edges = {}
        for idx, edge in enumerate(edge_array):
            v1, v2, _ = map(int, edge[:2])
            for v in (v1, v2):
                if v not in node_edges:
                    node_edges[v] = []
                node_edges[v].append(idx)

        # Compute edge curvatures
        edge_curvatures = self.forman_ricci_curvatures(edge_array)

        # Average edge curvatures for each node
        for node, incident_edges in node_edges.items():
            if incident_edges:  # Skip isolated nodes
                node_curv = np.mean([edge_curvatures[e_idx] for e_idx in incident_edges])
                curvature_array[node] = node_curv

        return curvature_array

    def ollivier_ricci_curvature(
        self, 
        edge_list: np.ndarray, 
        alpha: float = 0
    ) -> np.ndarray:
        """
        Compute Ollivier-Ricci curvature for edges using the GraphRicciCurvature package.
        
        Parameters
        ----------
        edge_list : np.ndarray
            Array of shape (m, 3) containing [node1, node2, weight] for each edge
        alpha : float, optional
            Convex combination parameter for the computation (default: 0)
            
        Returns
        -------
        np.ndarray
            Array of shape (m, 3) containing [node1, node2, curvature] for each edge
        """
        # Create NetworkX graph
        G = nx.Graph()
        for edge in edge_list:
            n1, n2, length = int(edge[0]), int(edge[1]), edge[2]
            G.add_edge(n1, n2, weight=1/length)
        
        # Compute Ollivier-Ricci curvature
        orc = OllivierRicci(G, alpha=alpha, verbose="ERROR")
        orc.compute_ricci_curvature()
        
        # Extract curvatures
        result = np.zeros_like(edge_list)
        result[:, :2] = edge_list[:, :2]
        
        for i, (n1, n2, _) in enumerate(edge_list):
            result[i, 2] = orc.G[int(n1)][int(n2)]["ricciCurvature"]
        
        return result

    def analyze_graph_structure(self, edges: np.ndarray, n_nodes: int, graph_type: str = "") -> None:
        """
        Analyze the structural properties of the graph that might affect mixing time.
        """
        # Convert to NetworkX graph for analysis
        G = nx.Graph()
        for u, v, w in edges:
            G.add_edge(int(u), int(v), weight=float(w))
        
        print(f"\nGraph Structure Analysis for {graph_type}:")
        print(f"Basic Properties:")
        print(f"  Nodes: {n_nodes}")
        print(f"  Edges: {len(edges)}")
        print(f"  Average degree: {2*len(edges)/n_nodes:.2f}")
        
        # Connectivity analysis
        components = list(nx.connected_components(G))
        print(f"\nConnectivity:")
        print(f"  Number of components: {len(components)}")
        print(f"  Largest component size: {len(max(components, key=len))}")
        
        if len(components) == 1:
            # Only compute these if graph is connected
            print(f"\nPath Properties:")
            diameter = nx.diameter(G)
            avg_path = nx.average_shortest_path_length(G, weight='weight')
            print(f"  Diameter: {diameter}")
            print(f"  Average path length: {avg_path:.4f}")
            
            # Clustering and structure
            clustering = nx.average_clustering(G)
            print(f"\nClustering Properties:")
            print(f"  Average clustering coefficient: {clustering:.4f}")
            
            # Degree distribution stats
            degrees = [d for _, d in G.degree()]
            print(f"\nDegree Distribution:")
            print(f"  Min degree: {min(degrees)}")
            print(f"  Max degree: {max(degrees)}")
            print(f"  Std dev of degrees: {np.std(degrees):.4f}")
        
        return G

    def compute_mixing_time_sparse_weighted(
        self, 
        edges: np.ndarray, 
        n_nodes: int, 
        epsilon: float = 1e-3,
        max_iter: int = 2000,
        debug: bool = True
    ) -> Optional[int]:
        """Enhanced version with detailed diagnostics"""
        if debug:
            G = self.analyze_graph_structure(edges, n_nodes)
        
        # Build sparse adjacency structure with weights
        row, col, data = [], [], []
        deg = np.zeros(n_nodes, dtype=np.float64)
        
        for u, v, length in edges:
            u, v = int(u), int(v)
            weight = 1.0 / length
            
            row.extend([u, v])
            col.extend([v, u])
            data.extend([weight, weight])
            deg[u] += weight
            deg[v] += weight

        # Create sparse matrices
        adjacency_sparse = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        adjacency_csr = adjacency_sparse.tocsr()
        
        if debug:
            print("\nTransition Matrix Properties:")
            print(f"  Max transition probability: {max(data)/min(deg[deg>0]):.6f}")
            print(f"  Min transition probability: {min(data)/max(deg):.6f}")
            print(f"  Probability ratio (max/min): {(max(data)*max(deg))/(min(data)*min(deg[deg>0])):.6f}")

        # Compute stationary distribution
        total_weight = deg.sum()
        pi = deg / total_weight
        
        if debug:
            print("\nStationary Distribution Properties:")
            print(f"  Max probability: {np.max(pi):.6f}")
            print(f"  Min probability: {np.min(pi[pi>0]):.6f}")
            print(f"  Number of zero probabilities: {np.sum(pi==0)}")

        # Power iteration with enhanced diagnostics
        p = np.ones(n_nodes) / n_nodes
        last_distance = float('inf')
        best_distance = float('inf')
        stall_count = 0
        
        convergence_history = []
        
        for t in range(max_iter):
            p_next = np.zeros(n_nodes, dtype=np.float64)
            
            for i in range(n_nodes):
                if deg[i] > 0:
                    start, end = adjacency_csr.indptr[i:i+2]
                    for idx in range(start, end):
                        j = adjacency_csr.indices[idx]
                        w_ij = adjacency_csr.data[idx]
                        p_next[j] += p[i] * w_ij * (1.0/deg[i])
            
            # Normalize to prevent numerical drift
            p_next = p_next / np.sum(p_next)
            
            distance = 0.5 * np.sum(np.abs(p_next - pi))
            convergence_history.append(distance)
            
            # Track best distance and report progress
            best_distance = min(best_distance, distance)
            
            if debug and (t % 100 == 0 or t < 10):
                print(f"  Iteration {t}: distance = {distance:.6f}")
                if t > 0:
                    improvement = (convergence_history[-2] - distance) / convergence_history[-2]
                    print(f"    Improvement: {improvement:.2%}")
            
            if distance < epsilon:
                if debug:
                    print(f"\nConverged after {t+1} iterations")
                    print(f"Final distance: {distance:.6f}")
                return t + 1
            
            if distance >= last_distance:
                stall_count += 1
            else:
                stall_count = 0
            
            if stall_count > 50 and best_distance < epsilon * 10:
                if debug:
                    print(f"\nEarly convergence at {t+1} iterations")
                    print(f"Best distance achieved: {best_distance:.6f}")
                return t + 1
            
            last_distance = distance
            p = p_next
        
        if best_distance < epsilon * 10:
            if debug:
                print(f"\nPartial convergence after {max_iter} iterations")
                print(f"Best distance achieved: {best_distance:.6f}")
            return max_iter
        
        if debug:
            print(f"\nFailed to converge after {max_iter} iterations")
            print(f"Final distance: {distance:.6f}")
            print(f"Best distance achieved: {best_distance:.6f}")
            print("\nConvergence behavior:")
            print(f"  Initial distance: {convergence_history[0]:.6f}")
            print(f"  Final distance: {convergence_history[-1]:.6f}")
            print(f"  Best distance: {best_distance:.6f}")
            print(f"  Number of stalls: {stall_count}")
        
        return None

    def compute_mixing_time_sparse_unweighted(
        self, 
        edges: np.ndarray, 
        n_nodes: int, 
        epsilon: float = 1e-4, 
        max_iter: int = 1000
    ) -> Optional[int]:
        """
        Computes approximate mixing time for an undirected, unweighted graph using
        a sparse random-walk transition matrix.

        Parameters
        ----------
        edges : np.ndarray
            Array of shape (m, 2) or (m, 3) containing [node1, node2] or 
            [node1, node2, length] for each edge (length is ignored)
        n_nodes : int
            Number of nodes in the graph
        epsilon : float, optional
            Convergence threshold for total variation distance (default: 1e-4)
        max_iter : int, optional
            Maximum number of iterations (default: 1000)

        Returns
        -------
        Optional[int]
            Number of steps until mixing, or None if not converged
        """
        # Build sparse adjacency structure with unit weights
        row, col, data = [], [], []
        deg = np.zeros(n_nodes, dtype=np.float64)
        
        for edge in edges:
            u, v = int(edge[0]), int(edge[1])
            
            # Add both directions for undirected graph
            row.extend([u, v])
            col.extend([v, u])
            data.extend([1.0, 1.0])
            deg[u] += 1
            deg[v] += 1

        # Create sparse matrices and compute transition probabilities
        adjacency_sparse = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        adjacency_csr = adjacency_sparse.tocsr()
        inv_deg = np.zeros(n_nodes, dtype=np.float64)
        inv_deg[deg > 0] = 1.0 / deg[deg > 0]

        # Compute stationary distribution
        total_degree = deg.sum()
        pi = deg / total_degree

        # Power iteration
        p = np.ones(n_nodes) / n_nodes
        for t in range(max_iter):
            p_next = np.zeros(n_nodes, dtype=np.float64)
            
            for i in range(n_nodes):
                if deg[i] > 0:
                    start, end = adjacency_csr.indptr[i:i+2]
                    for idx in range(start, end):
                        j = adjacency_csr.indices[idx]
                        p_next[j] += p[i] / deg[i]
            
            p = p_next
            
            if 0.5 * np.sum(np.abs(p - pi)) < epsilon:
                return t + 1

        return None