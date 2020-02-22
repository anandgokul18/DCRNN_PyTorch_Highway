README
------

If using Split graphs, please follow the below instructions:

0. Create a directory 'data/results/' or any other location which should be used as 'predictions_dir'

	mkdir data/results

1. Before everything, split the dataset into 3 partitions (needed only once):
	python -m scripts.generate_partitioned_training_data --output_dir=data/Highway --predictions_dir=data/results --traffic_df_filename=data/dcrnn_highway_6m.h5 --pkl_filename=data/sensor_graph_highway/dcrnn_highway_adj_mx.pkl --number_of_partitions=3


	Note: The predictions_dir MUST be the same as the "predictions_dir" in the configuration yaml file. This file will contain the original sensor ids with actual numbers and the zero-indexed arrays of sensor IDs in each of the partitions

2. On 
	model/pytorch/dcrnn_supervisor.py 
	model/pytorch/dcrnn_model.py 
	model/pytorch/dcrnn_cell.py
change the 'device' to desired Cuda device

3. On 
	data/model/dcrnn_highway.yaml

change the below parameters to respective values:

	data-->dataset_dir: the corresponding directory of the parition .ie. data/Highway<0/1/2>
	data-->graph_pkl_filename: the corresponding pkl file (same for all 3 partitions)
	split_into_subgraphs: true
	subgraph_id: either 0,1 or 2
	model-->num_nodes: the correspoding value based on the subgraph_id

	epochs: total number of epochs
	epoch: to load a previous model from the respective epoch. If 0, start from beginning

To train the model:
	python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_highway<0/1/2>.yaml


The final predictions are stored for all the partitions in:
data/results/highway_predictions<0/1/2>.npz


4. TO-DO: Combine all three predictions file back into their original order