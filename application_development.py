'''
THIS FILE IS A DUMMY.
IT IS USED TO TEST THE FLASK'S PREDICTION FUNCTIONALITY WITHOUT NEEDING TO USE REST CALLS
'''

#!flask/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, jsonify, request

from flask_cors import CORS, cross_origin

import argparse
import yaml
import numpy as np
import os

from lib.utils import load_graph_data
from model.pytorch.metis_graph_partitioning import partition_into_n_subgraphs

#For setting the current cuda device
import torch
import lib.currentCuda as currentCuda

def predict(config_filename='data/model/dcrnn_highway_flask.yaml', current_cuda_id=0, use_cpu_only=False, subgraph_id=0):
	# get sensor data and save it into the test dataset dir
    #data = request.get_json()
    #sensor_data = np.array([data["sensor_data"]])

    with open(config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        split_into_subgraphs = bool(supervisor_config['data'].get('split_into_subgraphs'))
        if split_into_subgraphs:
            assert(subgraph_id!=None, 'Enter a subgraph_id as python argument')
            subgraph_id = str(subgraph_id)
            print('Splitting into Sub-graphs: True')
            print('Current Sub-graph ID: '+ subgraph_id)
            adj_mx = partition_into_n_subgraphs(graph_pkl_filename, subgraph_id, int(supervisor_config['data'].get('number_of_subgraphs')))

            #Choosing the correct dataset directory for current subgraph
            supervisor_config['data']['dataset_dir'] = supervisor_config['data'].get('dataset_dir')+subgraph_id

            #Choosing the correct number of nodes for current subgraph
            listofnodesizes = (supervisor_config['model'].get('num_nodes')).split(',')
            supervisor_config['model']['num_nodes'] = int(listofnodesizes[int(subgraph_id)])
        else:
            subgraph_id = str(subgraph_id)

        currentCuda.init()
        currentCuda.dcrnn_cudadevice = torch.device("cuda:"+str(current_cuda_id) if torch.cuda.is_available() else "cpu")


        #Saving the JSON test information to npz in test dir. Bypassing the requirement for needing actual train and val dataset
        if not split_into_subgraphs:
            if not os.path.exists(supervisor_config['data'].get('dataset_dir')):
                os.makedirs(supervisor_config['data'].get('dataset_dir'))
            #np.savez_compressed(supervisor_config['data'].get('dataset_dir')+'/'+'test.npz', x=sensor_data['x'], y=sensor_data['y'])
            np.savez_compressed(supervisor_config['data'].get('dataset_dir')+'/'+'train.npz', x=np.array([]), y=np.array([]), x_offset=None, y_offset=None)
            np.savez_compressed(supervisor_config['data'].get('dataset_dir')+'/'+'val.npz', x=np.array([]), y=np.array([]), x_offset=None, y_offset=None)

        #Moving import here since the global variable for cuda device is declared above
        import model.pytorch.dcrnn_supervisor as dcrnn_supervisor

        supervisor = dcrnn_supervisor.DCRNNSupervisor(adj_mx=adj_mx, subgraph_id=subgraph_id, **supervisor_config)

        #supervisor.train(subgraph_identifier=subgraph_id)
        #Loading the previously trained model
        supervisor.load_model(subgraph_id=subgraph_id)


        #Evaluating the model finally and storing the results
        mean_score, outputs = supervisor.evaluate('test')
        output_filename = supervisor_config.get('predictions_dir')+'/'+'final_predictions'+subgraph_id+'.npz'
        np.savez_compressed(output_filename, **outputs)
        print("MAE : {}".format(mean_score))
        print('Predictions saved as {}.'.format(output_filename))
        import pdb; pdb.set_trace()
        return "completed"


if __name__ == '__main__':
    predict()