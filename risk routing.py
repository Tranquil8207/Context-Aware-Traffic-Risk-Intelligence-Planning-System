import osmnx as ox
import networkx as nx
import geopandas as gpd
import folium

# --- load your existing artifacts ---
G = ox.load_graphml("chandigarh_core.graphml")
G = ox.project_graph(G)
G = ox.consolidate_intersections(G, tolerance=15, rebuild_graph=True, dead_ends=False)
G = ox.project_graph(G, to_latlong=True)

edges = gpd.read_file("chandigarh_segments_risk.geojson")

# attach risk_norm back onto the graph's edges as a weight
risk_lookup = edges.set_index("segment_id")["risk_norm"].to_dict()

for u, v, k, data in G.edges(keys=True, data=True):
    seg_id = f"{u}_{v}_{k}"
    risk = risk_lookup.get(seg_id, 0.5)  # fallback if missing
    length_km = data.get("length", 100) / 1000

    # combine risk and distance so route isn't absurdly long just to dodge risk
    # tune this weight ratio based on what looks reasonable
    data["risk_cost"] = length_km * (1 + 3 * risk)


def get_route(orig_lat, orig_lng, dest_lat, dest_lng, weight="risk_cost"):
    orig_node = ox.distance.nearest_nodes(G, orig_lng, orig_lat)
    dest_node = ox.distance.nearest_nodes(G, dest_lng, dest_lat)
    route = nx.shortest_path(G, orig_node, dest_node, weight=weight)
    return route


def route_to_coords(G, route):
    return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]


def plot_route_comparison(orig_lat, orig_lng, dest_lat, dest_lng):
    route_safe = get_route(orig_lat, orig_lng, dest_lat, dest_lng, weight="risk_cost")
    route_short = get_route(orig_lat, orig_lng, dest_lat, dest_lng, weight="length")

    coords_safe = route_to_coords(G, route_safe)
    coords_short = route_to_coords(G, route_short)

    m = folium.Map(location=[orig_lat, orig_lng], zoom_start=15, tiles="CartoDB positron")

    folium.PolyLine(coords_short, color="blue", weight=5, opacity=0.7,
                     tooltip="Shortest route").add_to(m)
    folium.PolyLine(coords_safe, color="green", weight=5, opacity=0.9,
                     tooltip="Least-risk route").add_to(m)

    folium.Marker([orig_lat, orig_lng], popup="Start", icon=folium.Icon(color="black")).add_to(m)
    folium.Marker([dest_lat, dest_lng], popup="End", icon=folium.Icon(color="black")).add_to(m)

    m.save("route_comparison.html")
    print("Saved route_comparison.html — green = least-risk, blue = shortest")

    return route_safe, route_short

# example — plug in real coordinates within your bbox
if __name__ == "__main__":
    plot_route_comparison(30.740, 76.770, 30.748, 76.788)