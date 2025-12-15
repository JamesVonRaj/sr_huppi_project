import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from src.sr_code.generate_line_segments_dynamic_thickness import generate_line_segments_dynamic_thickness
from src.huppi_analysis.measures import Network_Measures
from src.adapter import sr_to_edge_list

# Ensure figures directory exists
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def plot_network(segments_dict, edge_list, num_nodes, alpha, box_size, ter, avg_orc):
    """
    Create a visualization of the SR network for a given alpha value.
    Shows both the raw geometry and the network graph side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    # --- Left: Raw Geometry ---
    ax1 = axes[0]
    ax1.set_xlim(-0.02, box_size + 0.02)
    ax1.set_ylim(-0.02, box_size + 0.02)
    ax1.set_aspect('equal')
    
    # Plot boundary
    boundary = plt.Rectangle((0, 0), box_size, box_size, 
                               fill=False, edgecolor='#2c3e50', linewidth=2, linestyle='--')
    ax1.add_patch(boundary)
    
    # Plot segments
    for seg_id, segment in segments_dict.items():
        x1, y1 = segment.start
        x2, y2 = segment.end
        
        if isinstance(seg_id, str) and seg_id.startswith('b'):
            ax1.plot([x1, x2], [y1, y2], color='#34495e', linewidth=2, alpha=0.9)
        else:
            ax1.plot([x1, x2], [y1, y2], color='#3498db', linewidth=1.2, alpha=0.7)
    
    ax1.set_xlabel('X', fontsize=11)
    ax1.set_ylabel('Y', fontsize=11)
    ax1.set_title(f'SR Geometry (α = {alpha:.2f})', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.4)
    
    # --- Right: Network Graph ---
    ax2 = axes[1]
    
    # Create NetworkX graph
    G = nx.Graph()
    for u, v, weight in edge_list:
        G.add_edge(int(u), int(v), weight=weight)
    
    # Use spring layout
    pos = nx.spring_layout(G, seed=42, k=0.3, iterations=50)
    
    # Scale positions
    for node in pos:
        pos[node] = pos[node] * box_size * 0.45 + box_size * 0.5
    
    ax2.set_xlim(-0.1, box_size + 0.1)
    ax2.set_ylim(-0.1, box_size + 0.1)
    ax2.set_aspect('equal')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='#3498db', width=0.8, alpha=0.5)
    
    # Draw nodes - color by degree
    degrees = [G.degree(n) for n in G.nodes()]
    node_colors = degrees
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax2,
                                   node_color=node_colors,
                                   cmap=plt.cm.YlOrRd,
                                   node_size=30,
                                   alpha=0.9)
    
    ax2.set_xlabel('X (layout)', fontsize=11)
    ax2.set_ylabel('Y (layout)', fontsize=11)
    ax2.set_title(f'Network Graph ({num_nodes} nodes, {len(edge_list)} edges)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.4)
    
    # Add colorbar for node degree
    cbar = plt.colorbar(nodes, ax=ax2, shrink=0.8)
    cbar.set_label('Node Degree', fontsize=10)
    
    # Add metrics as text
    metrics_text = f'TER: {ter:.1f}\nMean ORC: {avg_orc:.3f}'
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Overall title
    fig.suptitle(f'Scale-Rich Network Analysis (α = {alpha:.2f})', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    output_path = FIGURES_DIR / f"network_alpha_{alpha:.2f}.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Network plot saved: {output_path.name}")


def run_experiment():
    # Setup Parameters - Using 10 values for smooth curves in plots
    alpha_values = np.linspace(0.1, 1.0, 10)
    size = 200
    box_size = 1.0
    epsilon = 0.001
    
    results = []
    measures = Network_Measures()

    print(f"Starting Scale-Rich vs HuPPI Metrics Comparison (Size={size})...")

    for alpha in alpha_values:
        print(f"\n--- Processing Alpha = {alpha} ---")
        
        # 1. Generate SR Geometry
        lambda_0 = 0.05 
        t_steps = np.arange(1, size + 1)
        thickness_arr = lambda_0 * np.power(t_steps, -alpha)
        
        try:
            segments_dict, _, _, _ = generate_line_segments_dynamic_thickness(
                size=size,
                thickness_arr=thickness_arr,
                angles='uniform',
                epsilon=epsilon,
                box_size=box_size
            )
        except Exception as e:
            print(f"Generation failed for alpha={alpha}: {e}")
            continue

        # 2. Convert to Primal Graph
        edge_list, num_nodes = sr_to_edge_list(segments_dict)
        print(f"Generated Graph: {num_nodes} nodes, {len(edge_list)} edges")

        # 3. Compute Metrics
        try:
            ter = measures.compute_effective_resistance(edge_list)
            print(f"TER: {ter:.4f}")
        except Exception as e:
            print(f"TER Calc failed: {e}")
            ter = np.nan

        try:
            orc_data = measures.ollivier_ricci_curvature(edge_list, alpha=0)
            avg_orc = np.mean(orc_data[:, 2])
            print(f"Avg ORC: {avg_orc:.4f}")
        except Exception as e:
            print(f"ORC Calc failed: {e}")
            avg_orc = np.nan

        results.append({
            'alpha': alpha,
            'nodes': num_nodes,
            'edges': len(edge_list),
            'TER': ter,
            'Mean_ORC': avg_orc
        })
        
        # 4. Generate network visualization
        plot_network(segments_dict, edge_list, num_nodes, alpha, box_size, ter, avg_orc)

    df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(df)
    df.to_csv("sr_huppi_comparison_results.csv", index=False)

if __name__ == "__main__":
    run_experiment()

