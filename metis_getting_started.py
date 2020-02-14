import yaml
import networkx as nx
import metis

from DCRNN_Highway.lib.utils import load_graph_data

graph_pkl_filename = '/home/users/anandgok/dcrnn_highway_adj_mx.pkl'

sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

G = nx.from_numpy_matrix(adj_mx, parallel_edges=False, create_using=nx.MultiDiGraph)

(edgecuts, parts) = metis.part_graph(G, 3)