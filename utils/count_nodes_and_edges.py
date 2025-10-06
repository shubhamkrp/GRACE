# The data file must contain "nodes" and "edges" json key object.
import json

def count_nodes_edges(json_file):
    with open(json_file, 'r') as f:
        graph_data = json.load(f)

    num_nodes = len(graph_data.get("nodes", []))
    num_edges = len(graph_data.get("edges", []))

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")

json_file = "/home/user/data.json"
count_nodes_edges(json_file)
