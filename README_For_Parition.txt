README
------

If using Split graphs, please change the below fields:

1. On 
	model/pytorch/dcrnn_supervisor.py 
	model/pytorch/dcrnn_model.py 
	model/pytorch/and dcrnn_cell.py
change the 'device' to desired Cuda device

2. On 
	data/model/dcrnn_highway.yaml

change the below parameters to respective values:

	data-->dataset_dir: the corresponding directory of the parition .ie. data/Highway<0/1/2>
	split_into_subgraphs: true
	subgraph_id: either 0,1 or 2
	model-->num_nodes: the correspoding value based on the subgraph_id

	epochs: total number of epochs
	epoch: to load a previous model from the respective epoch. If 0, start from beginning



The final predictions are stored for all the partitions in:
data/results/highway_predictions<0/1/2>.npz