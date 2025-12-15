import numpy as np
import pandas as pd
from src.sr_code.generate_line_segments_dynamic_thickness import generate_line_segments_dynamic_thickness
from src.huppi_analysis.measures import Network_Measures
from src.adapter import sr_to_edge_list

def run_experiment():
    # Setup Parameters
    alpha_values = [0.1, 0.5, 0.9]
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

    df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(df)
    df.to_csv("sr_huppi_comparison_results.csv", index=False)

if __name__ == "__main__":
    run_experiment()

