"""
import yaml
import networkx as nx
import metis
import numpy as np

from lib.utils import load_graph_data

graph_pkl_filename="data/sensor_graph_arterial/adj_mx_arterial.pkl"

sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

G = nx.from_numpy_matrix(adj_mx, parallel_edges=False, create_using=nx.DiGraph)

import matplotlib.pyplot as plt

nx.draw(G)  # networkx draw()

plt.draw()  # pyplot draw()
"""

import networkx as nx
import metis
import pydot

from lib.utils import load_graph_data

from networkx.drawing.nx_agraph import write_dot

graph_pkl_filename="data/sensor_graph_arterial/adj_mx_arterial.pkl"

sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

G = nx.from_numpy_matrix(adj_mx, parallel_edges=False, create_using=nx.DiGraph)

print("loaded graph...")

(edgecuts, parts) = metis.part_graph(G, 15)

print("partioned graph...")

colors = ['red','yellow', 'blue', 'black', 'green', 'brown', 'purple', 'gray', 'orange', 'pink', 'cyan', 'maroon', 'coral', 'teal', 'crimson']
for i, p in enumerate(parts):
	try:
		G.nodes[i]['color'] = colors[p]
	except:
		print("Using 'azure' color instead of "+colors[p])
		G.nodes[i]['color'] = 'azure'


print("color coded graph...")

write_dot(G, 'graph.dot')