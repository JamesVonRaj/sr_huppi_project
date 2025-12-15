# Scale-Rich & HuPPI Network Analysis

This project combines **Scale-Rich (SR) Metamaterial** geometry generation with **HuPPI Network Analysis** transport metrics to analyze metamaterial networks using effective resistance and curvature measures.

## Overview

The project bridges two research domains:
1. **Scale-Rich Metamaterials** - Generates complex metamaterial geometries with scale-free properties
2. **HuPPI Network Analysis** - Provides network analysis tools including effective resistance and Ollivier-Ricci curvature

## Project Structure

```
project_root/
│
├── src/
│   ├── __init__.py
│   │
│   ├── sr_code/                  # Scale-Rich Metamaterials code
│   │   ├── __init__.py
│   │   ├── Classes.py            # Line segment and polygon classes
│   │   ├── sample_in_polygon.py  # Polygon sampling utilities
│   │   └── generate_line_segments_dynamic_thickness.py
│   │
│   ├── huppi_analysis/           # HuPPI Network Analysis code
│   │   ├── __init__.py
│   │   └── measures.py           # Network measures (TER, ORC, etc.)
│   │
│   └── adapter.py                # Bridge between SR and HuPPI
│
├── main.py                       # Main experiment runner
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main experiment:

```bash
python main.py
```

This will:
1. Generate Scale-Rich geometries for different alpha values (0.1, 0.5, 0.9)
2. Convert the geometries to graph edge lists
3. Compute transport metrics:
   - **TER** (Total Effective Resistance)
   - **ORC** (Ollivier-Ricci Curvature)
4. Save results to `sr_huppi_comparison_results.csv`

## Key Metrics

### Total Effective Resistance (TER)
Measures how well the network conducts "flow" - lower values indicate better connectivity.

### Ollivier-Ricci Curvature (ORC)
A geometric measure of network structure:
- **Positive curvature**: Edges in densely connected regions
- **Negative curvature**: Edges that act as bridges or bottlenecks

## Source Repositories

- [Scale-Rich Metamaterials](https://github.com/Barabasi-Lab/Scale-Rich-Metamaterials)
- [HuPPI Network Analysis](https://github.com/DMREF-networks/HuPPI-Network-Analysis)

## License

Please refer to the original repositories for licensing information.

