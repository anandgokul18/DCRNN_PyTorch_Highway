from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import deepspeed

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor_deepspeed import DCRNNSupervisor


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, args, **supervisor_config)

        supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/dcrnn_highway.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')

    #DeepSpeed Parser
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    main(args)