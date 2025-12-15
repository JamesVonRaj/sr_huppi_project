import numpy as np
import networkx as nx
from typing import Dict, Tuple
from src.sr_code.Classes import LineSegment

def sr_to_edge_list(segments_dict: Dict[int, LineSegment]) -> Tuple[np.ndarray, int]:
    """
    Converts Scale-Rich segments dictionary into a Numpy Edge List for HuPPI analysis.
    """
    coord_to_id = {}
    next_id = 0
    edges = []
    
    def get_id(coord):
        nonlocal next_id
        # Rounding prevents floating point errors creating duplicate nodes
        rounded = (round(coord[0], 6), round(coord[1], 6))
        if rounded not in coord_to_id:
            coord_to_id[rounded] = next_id
            next_id += 1
        return coord_to_id[rounded]

    for seg_id, segment in segments_dict.items():
        u_coord = segment.start
        v_coord = segment.end
        
        u = get_id(u_coord)
        v = get_id(v_coord)
        length = segment.length()
        
        # Avoid self-loops or zero-length edges
        if u != v and length > 1e-6:
            edges.append([u, v, length])

    edge_array = np.array(edges, dtype=np.float64)
    return edge_array, next_id

