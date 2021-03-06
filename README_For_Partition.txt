README
------

If using Split graphs, please follow the below instructions to train the model:
===============================================================================

((PREREQ 0)). Create a directory 'data/results/' or any other location which should be used as 'predictions_dir'

	mkdir data/results

((PREREQ 1)). Before everything, split the dataset into 3 partitions (needed only once):
	python -m scripts.generate_partitioned_training_data --output_dir=data/Highway --predictions_dir=data/results --traffic_df_filename=data/dcrnn_highway_6m.h5 --pkl_filename=data/sensor_graph_highway/dcrnn_highway_adj_mx.pkl --number_of_partitions=3


	Note: The predictions_dir MUST be the same as the "predictions_dir" in the configuration yaml file. This file will contain the original sensor ids with actual numbers and the zero-indexed arrays of sensor IDs in each of the partitions

To train the model:
	python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_highway.yaml --current_cuda_id=<0/1> --subgraph_id=<0/1/2>


The final predictions are stored for all the partitions in:
data/results/highway_predictions<0/1/2>.npz


4. TO-DO: Combine all three predictions file back into their original order

Graph:
======

1. python plotGraph_wth_Partitions.py
2. Then, scp the file 'listofedges.csv' to Desktop and edit with text editor.
3. rm -rf listofedges.csv
4. Find and Replace: 
		1. ';,' with ';'
		2. ',' with ' -> '
5. scp the listofedges back to the server to data/results directory

To predict on new test datatset:
=================================

1. Split the test dataset into partitions, but ONLY for test part .ie. don't split it 3 ways to train,test and val

2. For each partition, load the last saved epoch model. Then, run the eval method and get the final predictions for that.
