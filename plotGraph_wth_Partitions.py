import networkx as nx
import metis
#import pydot

from lib.utils import load_graph_data

#from networkx.drawing.nx_agraph import write_dot

graph_pkl_filename="data/sensor_graph_arterial/adj_mx_arterial.pkl"

sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

G = nx.from_numpy_matrix(adj_mx, parallel_edges=False, create_using=nx.DiGraph)

print("loaded graph...")

(edgecuts, parts) = metis.part_graph(G, 15)

print("partioned graph...")

listofedges = list(G.edges)

#Colors for each of the 15 partitions
colors = ['red','yellow', 'blue', 'black', 'green', 'brown', 'purple', 'gray', 'orange', 'pink', 'cyan', 'navyblue', 'coral', 'darkolivegreen', 'crimson']

###Writing list of edges to a txt file
# f = open( 'listofedges.txt', 'w' )
#f.write( str(listofedges))
#f.close()

###Writing list of edges to a CSV file
#import csv
#with open('listofedges.csv', 'w') as f:
#	writer = csv.writer(f , lineterminator='\n')
#	for tup in listofedges:
#		writer.writerow(tup)

###Writing list of edges to a CSV file so that it can be formatted into DOT using editors
import csv
with open('listofedges.csv', 'w') as f:
	writer = csv.writer(f , lineterminator='\n')
	writer.writerow(['digraph model {'])

	#Adding color coding
	for i in range(len(parts)):
		var = str(i)+' [color='+colors[parts[i]]+'];'
		writer.writerow([var])

	#Adding actual data. Replace the commas with ' -> '
	for tup in listofedges:
		writer.writerow(tup+(';',))
	writer.writerow('}')

#The end

'''
colors = ['red','yellow', 'blue', 'black', 'green', 'brown', 'purple', 'gray', 'orange', 'pink', 'cyan', 'maroon', 'coral', 'teal', 'crimson']
for i, p in enumerate(parts):
	try:
		G.nodes[i]['color'] = colors[p]
	except:
		print("Using 'azure' color instead of "+colors[p])
		G.nodes[i]['color'] = 'azure'


print("color coded graph...")

write_dot(G, 'graph.dot')
'''