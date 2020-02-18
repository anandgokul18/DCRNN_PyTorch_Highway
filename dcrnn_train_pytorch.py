from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor
from model.pytorch.metis_graph_partitioning import partition_into_3subgraphs

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        split_into_subgraphs = bool(supervisor_config['data'].get('split_into_subgraphs'))
        if split_into_subgraphs:
            subgraph_id = str(supervisor_config['data'].get('subgraph_id'))
            print('Splitting into Sub-graphs: True')
            print('Current Sub-graph ID: '+ subgraph_id)
            adj_mx = partition_into_3subgraphs(graph_pkl_filename, subgraph_id)
        else:
            subgraph_id = ''

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, subgraph_id=subgraph_id, **supervisor_config)

        supervisor.train(subgraph_id)

        #Evaluating the model finally and storing the results
        mean_score, outputs = supervisor.evaluate('test')
        output_filename = supervisor_config.get('predictions_dir')+'/'+'highway_predictions'+subgraph_id+'.npz'
        np.savez_compressed(output_filename, **outputs)
        print("MAE : {}".format(mean_score))
        print('Predictions saved as {}.'.format(output_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
