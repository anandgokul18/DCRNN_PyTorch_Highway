from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import numpy as np

from lib.utils import load_graph_data
from model.pytorch.metis_graph_partitioning import partition_into_n_subgraphs

#For setting the current cuda device
import torch
import lib.currentCuda as currentCuda

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        split_into_subgraphs = bool(supervisor_config['data'].get('split_into_subgraphs'))
        if split_into_subgraphs:
            assert(args.subgraph_id!=None, 'Enter a subgraph_id as python argument')
            subgraph_id = str(args.subgraph_id)
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
        currentCuda.dcrnn_cudadevice = torch.device("cuda:"+str(args.current_cuda_id) if torch.cuda.is_available() else "cpu")

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--current_cuda_id', default=0, type=int, help='Enter the CUDA GPU ID based on nvidia-smi in which you want to train this partition')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--subgraph_id', default=0, type=int, help='Choose the current subgraph')
    args = parser.parse_args()
    main(args)
