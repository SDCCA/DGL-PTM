# environment/grid_generator.py

import numpy as np
import osmnx as ox
from shapely.geometry import Polygon, box
import random

# --- Configuration for Grid Generation ---
# Central point for fetching Points of Interest (POIs)
AKUSE_CENTER_POINT = (6.100635676379143, 0.12736045405150456)
# Bounding coordinates for the Akuse area
AKUSE_BOUNDARY_COORDS = [
    (0.11640835827890646, 6.107983818842101),
    (0.13688109260416578, 6.106918939321614),
    (0.13518748441180858, 6.095601368066649),
    (0.11628382826476255, 6.090648307705022)
]
GRID_SIZE = 75  # The number of cells along one axis (e.g., 75x75 grid)
POI_FETCH_RADIUS = 1500  # Meters from the center point to fetch POIs
NUM_WATER_POINTS = 20
OUTPUT_FILENAME = "base_grid.npz"

def create_and_save_realistic_grid():
    """
    Fetches real-world geographic data to create and save a multi-layered
    base grid for the simulation environment.
    """
    print("--- Starting Realistic Grid Generation ---")
    
    # 1. Define the geographical boundary
    print(f"1. Defining boundary for Akuse...")
    boundary = Polygon(AKUSE_BOUNDARY_COORDS)
    minx, miny, maxx, maxy = boundary.bounds
    x_step = (maxx - minx) / GRID_SIZE
    y_step = (maxy - miny) / GRID_SIZE

    # 2. Identify all valid cells that fall within the boundary
    valid_cells_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = box(minx + c * x_step, miny + r * y_step,
                       minx + (c + 1) * x_step, miny + (r + 1) * y_step)
            if cell.intersects(boundary):
                valid_cells_mask[r, c] = 1
    
    # This mask will be the first layer, for placing residences
    grid_layers = [valid_cells_mask]
    property_map = {0: "residences"}

    # 3. Fetch POIs (schools, places of worship) from OpenStreetMap
    print(f"2. Fetching POIs within {POI_FETCH_RADIUS}m of the center point...")
    tags = {"amenity": ["school", "place_of_worship"]}
    pois_gdf = ox.features_from_point(AKUSE_CENTER_POINT, tags, dist=POI_FETCH_RADIUS)

    # 4. Create separate grid layers for each POI type
    print("3. Placing POIs onto the grid...")
    property_index = 1
    for amenity_type in tags["amenity"]:
        property_map[property_index] = amenity_type
        poi_layer = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        locations = pois_gdf[pois_gdf['amenity'] == amenity_type]
        
        for _, poi in locations.iterrows():
            if isinstance(poi['geometry'], Polygon):
                centroid = poi['geometry'].centroid
            else: # is a Point
                centroid = poi['geometry']
            
            # Find which grid cell this POI falls into
            c = int((centroid.x - minx) / x_step)
            r = int((centroid.y - miny) / y_step)
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                poi_layer[r, c] = 1
        
        grid_layers.append(poi_layer)
        property_index += 1

    # 5. Create a layer for randomly placed water points
    print("4. Placing random water points...")
    all_poi_locations = np.sum(np.stack(grid_layers[1:]), axis=0) > 0
    valid_cells_for_water = np.argwhere((valid_cells_mask == 1) & ~all_poi_locations)
    
    water_layer = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    if len(valid_cells_for_water) >= NUM_WATER_POINTS:
        selected_indices = random.sample(range(len(valid_cells_for_water)), NUM_WATER_POINTS)
        water_cells = valid_cells_for_water[selected_indices]
        water_layer[water_cells[:, 0], water_cells[:, 1]] = 1
    else:
        print(f"Warning: Not enough available cells ({len(valid_cells_for_water)}) to place {NUM_WATER_POINTS} water points.")

    grid_layers.append(water_layer)
    property_map[property_index] = "water"

    # 6. Combine layers and save to file
    final_grid = np.stack(grid_layers, axis=-1)
    
    # Ensure residences are not placed directly on top of POIs or water points
    non_residential_mask = np.sum(final_grid[:, :, 1:], axis=2) > 0
    final_grid[:, :, 0][non_residential_mask] = 0

    print(f"5. Saving base grid and metadata to {OUTPUT_FILENAME}...")
    np.savez_compressed(
        OUTPUT_FILENAME,
        grid=final_grid,
        bounds=np.array([minx, miny, maxx, maxy]),
        property_map=property_map
    )
    print("--- Grid Generation Complete ---")

if __name__ == '__main__':
    create_and_save_realistic_grid()