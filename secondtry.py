import os
import json
import numpy as np
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

# osmnx 2.x expects bbox as (west, south, east, north)
BBOX = (76.765, 30.730, 76.795, 30.755)   # (west, south, east, north)

GRAPH_CACHE_PATH = "chandigarh_core.graphml"
POI_JSON_PATH = "poi_selections.json"

CONSOLIDATE_TOLERANCE_M = 15

HIGHWAY_RISK = {
    "primary": 0.7, "primary_link": 0.7,
    "secondary": 0.5, "secondary_link": 0.5,
    "tertiary": 0.35, "tertiary_link": 0.35,
    "residential": 0.2,
    "living_street": 0.1,
    "unclassified": 0.25,
}

PROXIMITY_DECAY_M = 300
SCHOOL_WEIGHT = 0.30
OFFICE_WEIGHT = 0.15
INDUSTRIAL_WEIGHT = 0.10

FAR_FALLBACK_M = 5000.0


# ============================================================
# STEP 1: GET / LOAD GRAPH
# ============================================================

def get_graph():
    if os.path.exists(GRAPH_CACHE_PATH):
        print(f"Loading cached graph from {GRAPH_CACHE_PATH}")
        return ox.load_graphml(GRAPH_CACHE_PATH)

    print("Downloading graph from Overpass...")
    ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"
    ox.settings.use_cache = True
    ox.settings.timeout = 300

    try:
        G = ox.graph_from_bbox(bbox=BBOX, network_type="drive")
    except Exception as e:
        print(f"Primary Overpass endpoint failed ({e}), trying mirror...")
        ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"
        G = ox.graph_from_bbox(bbox=BBOX, network_type="drive")

    ox.save_graphml(G, filepath=GRAPH_CACHE_PATH)
    print(f"Graph saved to {GRAPH_CACHE_PATH}")
    return G


# ============================================================
# STEP 2: CONSOLIDATE INTERSECTIONS (merges roundabouts etc.)
# ============================================================

def consolidate(G):
    print("Consolidating intersections / roundabouts...")
    G_proj = ox.project_graph(G)

    G_consolidated = ox.consolidate_intersections(
        G_proj,
        tolerance=CONSOLIDATE_TOLERANCE_M,
        rebuild_graph=True,
        dead_ends=False
    )

    G_consolidated = ox.project_graph(G_consolidated, to_latlong=True)

    print(f"Before: {len(G.nodes)} nodes, {len(G.edges)} edges")
    print(f"After:  {len(G_consolidated.nodes)} nodes, {len(G_consolidated.edges)} edges")

    return G_consolidated


# ============================================================
# STEP 3: BUILD SEGMENT TABLE
# ============================================================

def build_segments(G):
    nodes, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()

    edges["segment_id"] = (
        edges["u"].astype(str) + "_" +
        edges["v"].astype(str) + "_" +
        edges["key"].astype(str)
    )

    print(f"Total segments: {len(edges)}")
    return edges


# ============================================================
# STEP 4: LOAD USER-SELECTED POIs
# ============================================================

def load_pois():
    if not os.path.exists(POI_JSON_PATH):
        raise FileNotFoundError(
            f"\n{POI_JSON_PATH} not found.\n"
            "Run poi_picker.py first, click your POIs in the browser, "
            "download the JSON, and place it in this folder."
        )

    with open(POI_JSON_PATH, "r") as f:
        raw = json.load(f)

    if len(raw) == 0:
        raise ValueError(f"{POI_JSON_PATH} is empty - you didn't select any points.")

    points = gpd.GeoDataFrame(
        raw,
        geometry=gpd.points_from_xy([p["lng"] for p in raw], [p["lat"] for p in raw]),
        crs="EPSG:4326"
    )

    schools = points[points["type"] == "school"]
    offices = points[points["type"] == "office"]
    industrial = points[points["type"] == "industrial"]

    print(f"Loaded POIs -> schools: {len(schools)}, offices: {len(offices)}, industrial: {len(industrial)}")

    return schools, offices, industrial


# ============================================================
# STEP 5: NEAREST-DISTANCE (VECTORIZED)
# ============================================================

def nearest_dist(edges_m, poi_gdf, colname):
    if len(poi_gdf) == 0:
        edges_m[colname] = FAR_FALLBACK_M
        return edges_m

    poi_pts = poi_gdf.to_crs(edges_m.crs)

    mid_gdf = gpd.GeoDataFrame(
        edges_m[["segment_id"]],
        geometry=edges_m["midpoint"],
        crs=edges_m.crs
    )

    joined = gpd.sjoin_nearest(mid_gdf, poi_pts[["geometry"]], distance_col=colname)
    joined = joined.drop_duplicates(subset="segment_id")

    edges_m = edges_m.merge(joined[["segment_id", colname]], on="segment_id", how="left")
    edges_m[colname] = edges_m[colname].fillna(FAR_FALLBACK_M)

    return edges_m


def attach_poi_distances(edges, schools, offices, industrial):
    edges_m = edges.to_crs(edges.estimate_utm_crs())
    edges_m["midpoint"] = edges_m.geometry.centroid

    edges_m = nearest_dist(edges_m, schools, "dist_to_school")
    edges_m = nearest_dist(edges_m, offices, "dist_to_office")
    edges_m = nearest_dist(edges_m, industrial, "dist_to_industrial")

    edges = edges.merge(
        edges_m[["segment_id", "dist_to_school", "dist_to_office", "dist_to_industrial"]],
        on="segment_id"
    )

    return edges


# ============================================================
# STEP 6: RISK SCORING
# ============================================================

def base_risk(hw):
    if isinstance(hw, list):
        hw = hw[0]
    return HIGHWAY_RISK.get(hw, 0.3)


def proximity_factor(dist_m, decay=PROXIMITY_DECAY_M):
    return np.exp(-dist_m / decay)


def compute_risk(edges):
    edges["base_risk"] = edges["highway"].apply(base_risk)

    edges["school_factor"] = edges["dist_to_school"].apply(proximity_factor)
    edges["office_factor"] = edges["dist_to_office"].apply(proximity_factor)
    edges["industrial_factor"] = edges["dist_to_industrial"].apply(proximity_factor)

    edges["risk"] = (
        edges["base_risk"]
        + SCHOOL_WEIGHT * edges["school_factor"]
        + OFFICE_WEIGHT * edges["office_factor"]
        + INDUSTRIAL_WEIGHT * edges["industrial_factor"]
    )

    rmin, rmax = edges["risk"].min(), edges["risk"].max()
    edges["risk_norm"] = (edges["risk"] - rmin) / (rmax - rmin) if rmax > rmin else 0.5

    return edges


# ============================================================
# STEP 7: OUTPUT MAPS
# ============================================================

def save_interactive_map(edges, schools, offices, industrial):
    m = edges.explore(
        column="risk_norm",
        cmap="RdYlGn_r",
        tiles="CartoDB positron",
        legend=True,
        style_kwds={"weight": 4}
    )

    for _, row in schools.iterrows():
        m = row.geometry
    # add POI markers on top
    import folium
    for _, row in schools.iterrows():
        folium.CircleMarker([row.geometry.y, row.geometry.x], radius=6, color="red",
                             fill=True, fill_color="red", popup="school").add_to(m)
    for _, row in offices.iterrows():
        folium.CircleMarker([row.geometry.y, row.geometry.x], radius=6, color="green",
                             fill=True, fill_color="green", popup="office").add_to(m)
    for _, row in industrial.iterrows():
        folium.CircleMarker([row.geometry.y, row.geometry.x], radius=6, color="orange",
                             fill=True, fill_color="orange", popup="industrial").add_to(m)

    m.save("chandigarh_risk_map.html")
    print("Saved interactive map -> chandigarh_risk_map.html")


def save_static_map(edges):
    fig, ax = plt.subplots(figsize=(12, 12))
    edges.plot(ax=ax, column="risk_norm", cmap="RdYlGn_r", linewidth=2, legend=True)
    plt.title("Chandigarh — Baseline Segment Risk")
    plt.savefig("chandigarh_risk_static.png", dpi=150)
    print("Saved static map -> chandigarh_risk_static.png")


# ============================================================
# MAIN
# ============================================================

def main():
    G = get_graph()
    G = consolidate(G)
    edges = build_segments(G)

    schools, offices, industrial = load_pois()
    edges = attach_poi_distances(edges, schools, offices, industrial)
    edges = compute_risk(edges)

    edges.to_file("chandigarh_segments_risk.geojson", driver="GeoJSON")
    print("Saved segment table -> chandigarh_segments_risk.geojson")

    save_interactive_map(edges, schools, offices, industrial)
    save_static_map(edges)

    print("\nDone. Top 10 highest-risk segments:")
    print(edges.sort_values("risk_norm", ascending=False)[
        ["segment_id", "highway", "risk_norm"]
    ].head(10).to_string(index=False))


if __name__ == "__main__":
    main()