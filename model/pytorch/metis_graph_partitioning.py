import yaml
import networkx as nx
import metis
import numpy as np
#import pydot

from lib.utils import load_graph_data
   
#metis python package needs to have the actual C library installed as well
#sudo apt-get install libmetis-dev  

"""
def partition_into_3subgraphs(graph_pkl_filename, required_graph_id=None):

	#graph_pkl_filename = '/home/users/anandgok/dcrnn_highway_adj_mx.pkl'

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

	'''
	Getting the nx graph in numpy format (.ie. the same as adj_mx) 

	Using Pandas here inititailly due to 2 reasons:
	1. Pandas provides easier way to verify since the headers .ie. row and col id are maintained in nx. So, can be compared against values from original adj_mx
	2. Numpy function does not work correctly for creating adjacency matrix in nx

	'''

	adj_mx0 = nx.to_pandas_adjacency(SG0)  
	adj_mx0 = adj_mx0. to_numpy() 

	adj_mx1 = nx.to_pandas_adjacency(SG1)
	adj_mx1 = adj_mx1. to_numpy() 

	adj_mx2 = nx.to_pandas_adjacency(SG2)
	adj_mx2 = adj_mx2. to_numpy() 


	'''
	Returning the correct adjacency matrix based on the requested index
	'''

	if required_graph_id=='-1': #Used for getting the partions to split train-val-test internally
		return (section0_elements,section1_elements,section2_elements)
	elif required_graph_id=='0':
		return adj_mx0
	elif required_graph_id=='1':
		return adj_mx1
	elif required_graph_id=='2':
		return adj_mx2
	else:
		raise ValueError('The required_graph_id should be a value from 0 to 2 only')
"""

def partition_into_n_subgraphs(graph_pkl_filename, required_graph_id=None, number_of_partitions=None):

	#if number_of_paritions==3:
	#	list0,list1,list2 = partition_into_3subgraphs(graph_pkl_filename, '-1')
	#	return [list0,list1,list2]

	#graph_pkl_filename = '/home/users/anandgok/dcrnn_highway_adj_mx.pkl'

	sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

	G = nx.from_numpy_matrix(adj_mx, parallel_edges=False, create_using=nx.DiGraph)

	(edgecuts, parts) = metis.part_graph(G, number_of_partitions)

	#Note 'parts' variable essentially is a list which tells which partition each element belongs to .ie. parts=[0,1,0,0,0,1,2,2,23,0,3,....]


	'''
	Based on the partitions created by metis, creating 3 lists with node-ids of each parition
	'''

	indexes = [0]*number_of_partitions
	listofpartitions = [[] for _ in range(number_of_partitions)]

	for i in range(0,len(parts)):
		indexes[parts[i]]+=1

		listofpartitions[parts[i]].append(i)


	import pdb; pdb.set_trace()

	'''Useful Commands'''

	#G.number_of_nodes()
	#G.number_of_edges()

	#list(G.edges)
	#list(G.nodes)


	if required_graph_id=='-1': #Used for getting the partions to split train-val-test internally
		return listofpartitions

	#Checking if the input is valid
	if int(required_graph_id) not in range(0,number_of_partitions):
		raise ValueError('The required_graph_id should be a value from 0 to '+str(number_of_partitions-1)+' only')

	'''
	Creating Subgraphs based on the metis partitions
	'''

	#CREATING SUB-GRAPHS MANUALLY

	#Single sub-graph whichever is required
	SG = G.__class__()
	SG.add_nodes_from((n, G.nodes[n]) for n in listofpartitions[int(required_graph_id)])

	SG.add_weighted_edges_from((n, nbr, G[n][nbr]['weight'])
	    for n, nbrs in G.adj.items() if n in listofpartitions[int(required_graph_id)]
	    for nbr, keydict in nbrs.items() if nbr in listofpartitions[int(required_graph_id)]
	    for key, d in keydict.items())	

	'''
	Plotting
	'''
	#nx.nx_pydot.write_dot(adj_mx, 'original.dot')

	'''
	Getting the nx graph in numpy format (.ie. the same as adj_mx) 

	Using Pandas here inititailly due to 2 reasons:
	1. Pandas provides easier way to verify since the headers .ie. row and col id are maintained in nx. So, can be compared against values from original adj_mx
	2. Numpy function does not work correctly for creating adjacency matrix in nx

	'''

	current_adj_mx = nx.to_pandas_adjacency(SG)  
	current_adj_mx = current_adj_mx. to_numpy() 

	'''
	Returning the correct adjacency matrix based on the requested index
	'''

	return current_adj_mx
