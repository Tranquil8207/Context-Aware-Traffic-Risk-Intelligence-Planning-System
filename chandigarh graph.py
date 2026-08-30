import osmnx as ox


G = ox.graph_from_place("Chandigarh, India", network_type="drive")


ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 60

BBOX = (76.765, 30.730, 76.795, 30.755)

print("Starting download...")
G = ox.graph_from_bbox(bbox=BBOX, network_type="drive")
print("Done:", len(G.nodes), "nodes")