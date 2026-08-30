import os
import random
import json

import osmnx as ox
import networkx as nx
import geopandas as gpd
import folium

# ============================================================
# CONFIG
# ============================================================

GRAPH_CACHE_PATH = "chandigarh_core.graphml"
SEGMENTS_GEOJSON_PATH = "chandigarh_segments_risk.geojson"

CONSOLIDATE_TOLERANCE_M = 15

# how strongly risk penalizes a route vs raw distance
# higher = routes will detour more aggressively to avoid risky segments
RISK_WEIGHT_MULTIPLIER = 3

# how many synthetic trips to simulate for "before/after" traffic comparison
N_SAMPLE_TRIPS = 50

# how many top-impacted segments to show in the output map
TOP_N_IMPACTED = 10

MAP_CENTER = [30.742, 76.780]


# ============================================================
# STEP 1: LOAD GRAPH + RISK DATA
# ============================================================

def load_graph_and_risk():
    if not os.path.exists(GRAPH_CACHE_PATH):
        raise FileNotFoundError(
            f"{GRAPH_CACHE_PATH} not found. Run build_risk_map.py first."
        )
    if not os.path.exists(SEGMENTS_GEOJSON_PATH):
        raise FileNotFoundError(
            f"{SEGMENTS_GEOJSON_PATH} not found. Run build_risk_map.py first."
        )

    print("Loading cached graph...")
    G = ox.load_graphml(GRAPH_CACHE_PATH)

    print("Consolidating intersections...")
    G = ox.project_graph(G)
    G = ox.consolidate_intersections(
        G, tolerance=CONSOLIDATE_TOLERANCE_M, rebuild_graph=True, dead_ends=False
    )
    G = ox.project_graph(G, to_latlong=True)

    print("Loading risk-scored segments...")
    edges = gpd.read_file(SEGMENTS_GEOJSON_PATH)
    risk_lookup = edges.set_index("segment_id")["risk_norm"].to_dict()

    print("Attaching risk as edge weight...")
    for u, v, k, data in G.edges(keys=True, data=True):
        seg_id = f"{u}_{v}_{k}"
        risk = risk_lookup.get(seg_id, 0.5)
        length_km = data.get("length", 100) / 1000
        data["risk_cost"] = length_km * (1 + RISK_WEIGHT_MULTIPLIER * risk)
        data["risk_norm"] = risk

    return G, edges, risk_lookup


# ============================================================
# STEP 2: ROUTING HELPERS
# ============================================================

def get_route(G, orig_lat, orig_lng, dest_lat, dest_lng, weight="risk_cost"):
    orig_node = ox.distance.nearest_nodes(G, orig_lng, orig_lat)
    dest_node = ox.distance.nearest_nodes(G, dest_lng, dest_lat)
    return nx.shortest_path(G, orig_node, dest_node, weight=weight)


def generate_sample_trips(G, n=N_SAMPLE_TRIPS, seed=42):
    random.seed(seed)
    nodes = list(G.nodes)
    pairs = []
    attempts = 0
    while len(pairs) < n and attempts < n * 10:
        attempts += 1
        a, b = random.sample(nodes, 2)
        # skip pairs with no path at all, so sample generation doesn't crash later
        if nx.has_path(G, a, b):
            pairs.append((
                G.nodes[a]["y"], G.nodes[a]["x"],
                G.nodes[b]["y"], G.nodes[b]["x"]
            ))
    print(f"Generated {len(pairs)} sample trip pairs")
    return pairs


# ============================================================
# STEP 3: CLOSURE SIMULATION
# ============================================================

def simulate_closure(G, segment_id_to_close, sample_pairs, weight="risk_cost"):
    parts = segment_id_to_close.split("_")
    if len(parts) != 3:
        raise ValueError(f"segment_id '{segment_id_to_close}' is not in u_v_key format")

    u, v, k = int(parts[0]), int(parts[1]), int(parts[2])

    if not G.has_edge(u, v, k):
        raise ValueError(
            f"Edge {segment_id_to_close} not found in graph. "
            "Pick a valid segment_id from chandigarh_segments_risk.geojson."
        )

    # --- BEFORE: route all sample trips normally, tally edge usage ---
    edge_usage_before = {}
    for (olat, olng, dlat, dlng) in sample_pairs:
        try:
            route = get_route(G, olat, olng, dlat, dlng, weight=weight)
        except nx.NetworkXNoPath:
            continue
        for i in range(len(route) - 1):
            key = (route[i], route[i + 1])
            edge_usage_before[key] = edge_usage_before.get(key, 0) + 1

    # --- CLOSE THE ROAD ---
    G_closed = G.copy()
    G_closed.remove_edge(u, v, k)

    # --- AFTER: reroute all sample trips on the closed graph ---
    edge_usage_after = {}
    failed_routes = 0
    for (olat, olng, dlat, dlng) in sample_pairs:
        try:
            orig_node = ox.distance.nearest_nodes(G_closed, olng, olat)
            dest_node = ox.distance.nearest_nodes(G_closed, dlng, dlat)
            route = nx.shortest_path(G_closed, orig_node, dest_node, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            failed_routes += 1
            continue
        for i in range(len(route) - 1):
            key = (route[i], route[i + 1])
            edge_usage_after[key] = edge_usage_after.get(key, 0) + 1

    # --- COMPARE: which edges gained the most traffic ---
    all_edges = set(edge_usage_before) | set(edge_usage_after)
    impact = []
    for e in all_edges:
        before = edge_usage_before.get(e, 0)
        after = edge_usage_after.get(e, 0)
        delta = after - before
        if delta > 0:
            impact.append((e, before, after, delta))

    impact.sort(key=lambda x: x[3], reverse=True)

    print(f"\nClosing segment: {segment_id_to_close}")
    print(f"Failed routes (no path found after closure): {failed_routes}/{len(sample_pairs)}")
    print(f"\nTop impacted edges (gained the most rerouted traffic):")
    for e, before, after, delta in impact[:TOP_N_IMPACTED]:
        seg = f"{e[0]}_{e[1]}_0"
        risk = None
        for kk in range(3):  # key might not be 0, try a few
            candidate = f"{e[0]}_{e[1]}_{kk}"
            if candidate in risk_lookup:
                risk = risk_lookup[candidate]
                break
        print(f"  {e}  usage {before} -> {after}  (+{delta})  risk_norm={risk}")

    return impact, G_closed, failed_routes


# ============================================================
# STEP 4: VISUALIZATION
# ============================================================

def plot_closure_impact(G, segment_id_closed, impact, top_n=TOP_N_IMPACTED):
    m = folium.Map(location=MAP_CENTER, zoom_start=14, tiles="CartoDB positron")

    # base graph, faint grey
    for u, v, k, data in G.edges(keys=True, data=True):
        try:
            coords = [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]
            folium.PolyLine(coords, color="#cccccc", weight=1, opacity=0.5).add_to(m)
        except KeyError:
            continue

    # highlight the closed segment in black
    parts = segment_id_closed.split("_")
    cu, cv = int(parts[0]), int(parts[1])
    if cu in G.nodes and cv in G.nodes:
        coords = [(G.nodes[cu]["y"], G.nodes[cu]["x"]), (G.nodes[cv]["y"], G.nodes[cv]["x"])]
        folium.PolyLine(coords, color="black", weight=6, opacity=1.0,
                         tooltip="CLOSED ROAD").add_to(m)

    # top impacted edges in red, thicker = more rerouted traffic
    for (e, before, after, delta) in impact[:top_n]:
        u, v = e
        if u in G.nodes and v in G.nodes:
            coords = [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]
            folium.PolyLine(
                coords, color="red", weight=4 + min(delta, 10), opacity=0.9,
                tooltip=f"+{delta} rerouted trips (usage {before} -> {after})"
            ).add_to(m)

    m.save("closure_impact_map.html")
    print("Saved closure_impact_map.html")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    G, edges, risk_lookup = load_graph_and_risk()

    print("\nTop 10 highest-risk segments (candidates to test closing):")
    top_risk = edges.sort_values("risk_norm", ascending=False)[
        ["segment_id", "highway", "risk_norm"]
    ].head(10)
    print(top_risk.to_string(index=False))

    # ------------------------------------------------------------
    # pick which segment to close — default: the single highest-risk one
    # override this manually with any segment_id from the list above
    # ------------------------------------------------------------
    segment_to_close = top_risk.iloc[0]["segment_id"]
    print(f"\nUsing segment_to_close = {segment_to_close}")
    print("(edit this line in the script to test a different segment)")

    sample_trips = generate_sample_trips(G, n=N_SAMPLE_TRIPS)

    impact, G_closed, failed_routes = simulate_closure(G, segment_to_close, sample_trips)

    if len(impact) == 0:
        print("\nNo edges gained traffic — either the segment is a dead-end "
              "or your sample trips didn't route through it. Try a different segment_id.")
    else:
        plot_closure_impact(G, segment_to_close, impact)
        print("\nDone. Open closure_impact_map.html to see which roads are most impacted.")