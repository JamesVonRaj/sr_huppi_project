import numpy as np
import networkx as nx
from typing import Dict, Tuple, List
from src.sr_code.Classes import LineSegment


def sr_to_edge_list(segments_dict: Dict[int, LineSegment]) -> Tuple[np.ndarray, int]:
    """
    Converts Scale-Rich segments dictionary into a Numpy Edge List for HuPPI analysis.
    
    This function creates a "primal graph" where:
    - Nodes are located at ALL intersection points (where segments cross each other)
    - Edges connect consecutive intersection points along each segment
    
    This ensures the resulting graph is connected if the SR system is properly generated.
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
    
    def distance_from_start(start: Tuple[float, float], point: Tuple[float, float]) -> float:
        """Calculate distance from start point to another point."""
        return np.sqrt((point[0] - start[0])**2 + (point[1] - start[1])**2)

    for seg_id, segment in segments_dict.items():
        # Collect all intersection points along this segment
        # This includes the endpoints (start, end) and all neighbor intersections
        intersection_points: List[Tuple[float, float]] = []
        
        # Add segment endpoints
        intersection_points.append(tuple(segment.start))
        intersection_points.append(tuple(segment.end))
        
        # Add all intersection points from neighbors
        # The neighbors dict maps neighbor_id -> intersection_coordinate
        if hasattr(segment, 'neighbors') and segment.neighbors:
            for neighbor_id, intersection_coord in segment.neighbors.items():
                if intersection_coord is not None:
                    intersection_points.append(tuple(intersection_coord))
        
        # Remove duplicates (same point might be added multiple times)
        unique_points = list(set(intersection_points))
        
        # Sort points by distance from segment start
        start = tuple(segment.start)
        unique_points.sort(key=lambda p: distance_from_start(start, p))
        
        # Create edges between consecutive intersection points
        for i in range(len(unique_points) - 1):
            p1 = unique_points[i]
            p2 = unique_points[i + 1]
            
            u = get_id(p1)
            v = get_id(p2)
            length = distance_from_start(p1, p2)
            
            # Avoid self-loops or zero-length edges
            if u != v and length > 1e-6:
                edges.append([u, v, length])

    edge_array = np.array(edges, dtype=np.float64)
    return edge_array, next_id

