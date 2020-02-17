import yaml
import networkx as nx
import metis
#import pydot

from DCRNN_Highway.lib.utils import load_graph_data

graph_pkl_filename = '/home/users/anandgok/dcrnn_highway_adj_mx.pkl'

sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

G = nx.from_numpy_matrix(adj_mx, parallel_edges=False, create_using=nx.DiGraph)

(edgecuts, parts) = metis.part_graph(G, 3)


'''
Based on the partitions created by metis, creating 3 lists with node-ids of each parition
'''

indexes=[0,0,0]
section0_elements=[]
section1_elements=[]
section2_elements=[]

for i in range(0,len(parts)):
	indexes[parts[i]]+=1
	if(parts[i])==0:
		section0_elements.append(i)
	elif(parts[i])==1:
		section1_elements.append(i)
	else:
		section2_elements.append(i)

#indexes=[298, 316, 309]

'''Useful Commands'''

#G.number_of_nodes()
#G.number_of_edges()

#list(G.edges)
#list(G.nodes)

'''
Creating Subgraphs based on the metis partitions
'''

#CREATING SUB-GRAPHS MANUALLY

#Sub-graph0
#subgraph0 = G.subgraph(section0_elements)
SG0 = G.__class__()
SG0.add_nodes_from((n, G.nodes[n]) for n in section0_elements)

SG0.add_weighted_edges_from((n, nbr, G[n][nbr]['weight'])
    for n, nbrs in G.adj.items() if n in section0_elements
    for nbr, keydict in nbrs.items() if nbr in section0_elements
    for key, d in keydict.items())


#Sub-graph1
SG1 = G.__class__()
SG1.add_nodes_from((n, G.nodes[n]) for n in section1_elements)

SG1.add_weighted_edges_from((n, nbr, G[n][nbr]['weight'])
    for n, nbrs in G.adj.items() if n in section1_elements
    for nbr, keydict in nbrs.items() if nbr in section1_elements
    for key, d in keydict.items())

#Sub-graph2
SG2 = G.__class__()
SG2.add_nodes_from((n, G.nodes[n]) for n in section2_elements)

SG2.add_weighted_edges_from((n, nbr, G[n][nbr]['weight'])
    for n, nbrs in G.adj.items() if n in section2_elements
    for nbr, keydict in nbrs.items() if nbr in section2_elements
    for key, d in keydict.items())


'''
Plotting
'''
#nx.nx_pydot.write_dot(adj_mx, 'original.dot')