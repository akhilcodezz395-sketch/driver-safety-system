import numpy as np
import math
from typing import List, Tuple, Union

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth in meters.
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing between two points.
    Returns degrees [0, 360).
    """
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    dLon = lon2 - lon1

    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360) % 360

def curvature_from_polyline(coords: List[Tuple[float, float]]) -> List[float]:
    """
    Calculate curvature for a sequence of (lat, lon) points.
    Curvature k = 1 / R. 
    Using the Menger curvature formula (circumcircle of 3 points) or 
    change in bearing over distance.
    
    Here we use change in bearing / distance_segment.
    Returns a list of curvature values (1/m) of same length as input 
    (padded with 0 at ends).
    """
    n = len(coords)
    if n < 3:
        return [0.0] * n
    
    curvatures = [0.0]
    
    for i in range(1, n - 1):
        p_prev = coords[i-1]
        p_curr = coords[i]
        p_next = coords[i+1]
        
        dist1 = haversine_m(p_prev[0], p_prev[1], p_curr[0], p_curr[1])
        dist2 = haversine_m(p_curr[0], p_curr[1], p_next[0], p_next[1])
        
        # Avoid division by zero
        if dist1 < 1e-5 or dist2 < 1e-5:
            curvatures.append(0.0)
            continue
            
        b1 = bearing(p_prev[0], p_prev[1], p_curr[0], p_curr[1])
        b2 = bearing(p_curr[0], p_curr[1], p_next[0], p_next[1])
        
        # Difference in bearing in range [-180, 180]
        angle_diff = (b2 - b1 + 180) % 360 - 180
        angle_diff_rad = math.radians(angle_diff)
        
        # Approximate curvature k = d_theta / d_s
        # We can take avg distance
        avg_dist = (dist1 + dist2) / 2.0
        k = abs(angle_diff_rad) / avg_dist if avg_dist > 1e-5 else 0.0
        
        curvatures.append(k)
        
    curvatures.append(0.0)
    return curvatures
