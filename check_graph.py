"""
check_graph.py

Sanity check script to verify that the "Primal Graph" conversion is correct.
Generates a small SR system and visualizes both the raw geometry (ligaments)
and the network graph (nodes at intersections) side by side.

This helps confirm that intersecting lines in the geometry actually result
in connected nodes in the graph.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from src.sr_code.generate_line_segments_dynamic_thickness import generate_line_segments_dynamic_thickness
from src.adapter import sr_to_edge_list

# Ensure figures directory exists
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def generate_sr_system(size: int = 50, alpha: float = 0.5, box_size: float = 1.0):
    """
    Generate a Scale-Rich metamaterial system.
    
    Parameters
    ----------
    size : int
        Number of line segments to generate
    alpha : float
        Heterogeneity parameter (power-law exponent)
    box_size : float
        Size of the bounding box
        
    Returns
    -------
    segments_dict : dict
        Dictionary of LineSegment objects
    edge_list : np.ndarray
        Edge list for the primal graph [node1, node2, weight]
    num_nodes : int
        Number of nodes in the primal graph
    """
    # Setup thickness array using power-law distribution
    lambda_0 = 0.05
    t_steps = np.arange(1, size + 1)
    thickness_arr = lambda_0 * np.power(t_steps, -alpha)
    epsilon = 0.001
    
    print(f"Generating SR system: size={size}, alpha={alpha}")
    
    # Generate the SR geometry
    segments_dict, polygon_arr, segment_thickness_dict, _ = generate_line_segments_dynamic_thickness(
        size=size,
        thickness_arr=thickness_arr,
        angles='uniform',
        epsilon=epsilon,
        box_size=box_size
    )
    
    # Convert to edge list (primal graph)
    edge_list, num_nodes = sr_to_edge_list(segments_dict)
    
    print(f"Generated: {len(segments_dict)} segments -> {num_nodes} nodes, {len(edge_list)} edges")
    
    return segments_dict, edge_list, num_nodes, box_size


def plot_geometry(ax, segments_dict, box_size: float):
    """
    Plot the raw SR geometry (line segments / ligaments).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    segments_dict : dict
        Dictionary of LineSegment objects
    box_size : float
        Size of the bounding box
    """
    ax.set_xlim(-0.02, box_size + 0.02)
    ax.set_ylim(-0.02, box_size + 0.02)
    ax.set_aspect('equal')
    
    # Plot boundary
    boundary = plt.Rectangle((0, 0), box_size, box_size, 
                               fill=False, edgecolor='#2c3e50', linewidth=2, linestyle='--')
    ax.add_patch(boundary)
    
    # Color segments based on whether they are borders or interior
    for seg_id, segment in segments_dict.items():
        x1, y1 = segment.start
        x2, y2 = segment.end
        
        if isinstance(seg_id, str) and seg_id.startswith('b'):
            # Border segments
            ax.plot([x1, x2], [y1, y2], color='#34495e', linewidth=2.5, alpha=0.9)
        else:
            # Interior segments (ligaments)
            ax.plot([x1, x2], [y1, y2], color='#3498db', linewidth=1.5, alpha=0.8)
            
            # Mark endpoints with small dots
            ax.scatter([x1, x2], [y1, y2], c='#e74c3c', s=15, zorder=5, alpha=0.7)
    
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_title('Raw Geometry (Line Segments)', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, linestyle=':', alpha=0.5)


def plot_graph(ax, edge_list, num_nodes, box_size: float):
    """
    Plot the network graph with nodes at intersections.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    edge_list : np.ndarray
        Edge list [node1, node2, weight]
    num_nodes : int
        Number of nodes in the graph
    box_size : float
        Size of the bounding box
    """
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add edges with positions inferred from edge weights
    for u, v, weight in edge_list:
        G.add_edge(int(u), int(v), weight=weight)
    
    # We need to reconstruct node positions from the original geometry
    # For this, we'll use spring layout but constrained to approximate positions
    # For better accuracy, we'd need to pass coordinates from the adapter
    
    # Use spring layout for visualization
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=100)
    
    # Scale positions to box_size
    for node in pos:
        pos[node] = pos[node] * box_size * 0.45 + box_size * 0.5
    
    ax.set_xlim(-0.1, box_size + 0.1)
    ax.set_ylim(-0.1, box_size + 0.1)
    ax.set_aspect('equal')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, 
                           edge_color='#3498db', 
                           width=1.2, 
                           alpha=0.7)
    
    # Draw nodes
    node_sizes = [80 + 20 * G.degree(n) for n in G.nodes()]  # Size based on degree
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color='#e74c3c',
                           node_size=node_sizes,
                           alpha=0.9,
                           edgecolors='white',
                           linewidths=1.5)
    
    # Add node labels for smaller graphs
    if num_nodes <= 30:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color='white', font_weight='bold')
    
    ax.set_xlabel('X (layout)', fontsize=11)
    ax.set_ylabel('Y (layout)', fontsize=11)
    ax.set_title(f'Network Graph ({num_nodes} nodes, {len(edge_list)} edges)', 
                 fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, linestyle=':', alpha=0.5)


def print_graph_stats(edge_list, num_nodes):
    """Print statistics about the graph for sanity checking."""
    G = nx.Graph()
    for u, v, weight in edge_list:
        G.add_edge(int(u), int(v), weight=weight)
    
    print("\n" + "=" * 50)
    print("Graph Statistics (Sanity Check)")
    print("=" * 50)
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {len(edge_list)}")
    print(f"  Average degree: {2 * len(edge_list) / num_nodes:.2f}")
    
    # Connectivity check
    is_connected = nx.is_connected(G)
    num_components = nx.number_connected_components(G)
    print(f"  Connected: {is_connected}")
    print(f"  Number of components: {num_components}")
    
    # Degree distribution
    degrees = [d for _, d in G.degree()]
    print(f"  Min degree: {min(degrees)}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Mean degree: {np.mean(degrees):.2f}")
    
    # Edge weight statistics
    weights = edge_list[:, 2]
    print(f"\n  Edge lengths (weights):")
    print(f"    Min: {weights.min():.6f}")
    print(f"    Max: {weights.max():.6f}")
    print(f"    Mean: {weights.mean():.6f}")
    print("=" * 50)


def main():
    """Main entry point for the sanity check visualization."""
    print("=" * 60)
    print("SR System Sanity Check: Geometry vs Graph Visualization")
    print("=" * 60)
    
    # Generate a small SR system
    segments_dict, edge_list, num_nodes, box_size = generate_sr_system(
        size=50, 
        alpha=0.5, 
        box_size=1.0
    )
    
    # Print graph statistics
    print_graph_stats(edge_list, num_nodes)
    
    # Create side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Style the figure
    fig.patch.set_facecolor('white')
    
    # Left: Raw geometry
    plot_geometry(axes[0], segments_dict, box_size)
    
    # Right: Network graph
    plot_graph(axes[1], edge_list, num_nodes, box_size)
    
    # Add overall title
    fig.suptitle('Scale-Rich System: Geometry to Graph Conversion Check', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    output_path = FIGURES_DIR / "sr_graph_sanity_check.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path}")
    
    # Show plot
    plt.show()
    
    print("\nSanity check complete!")
    print("\nInterpretation:")
    print("  - Left plot shows the raw line segments (ligaments) of the SR system")
    print("  - Red dots mark segment endpoints (potential graph nodes)")
    print("  - Right plot shows the resulting network graph")
    print("  - Each node represents an intersection point")
    print("  - Each edge represents a ligament connecting two intersections")
    print("  - If the conversion is correct, the graph should be connected")
    print("    and the number of edges should match the number of interior segments")


if __name__ == "__main__":
    main()

