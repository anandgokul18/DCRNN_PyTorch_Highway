'''
import numpy as np
import pandas as pd

directory = "data/Highway_rough"

#train_data=np.load(directory+"/train.npz")
val_data=np.load(directory+"/val.npz")
#test_data=np.load(directory+"/test.npz")

val_df = pd.DataFrame(val_data['x'])
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd

from model.pytorch.metis_graph_partitioning import partition_into_n_subgraphs

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

def generate_train_val_test(df, partition_id, args):

    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("Currently processing Partition #: "+str(partition_id))
    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train


    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        target_location = args.output_dir+str(partition_id)
        if not os.path.exists(target_location):
            os.makedirs(target_location)
        np.savez_compressed(
            os.path.join(target_location, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

def generate_partitioned_data(args):

    #For getting the 3 partition nodes
    #list0,list1,list2 = partition_into_3subgraphs(args.pkl_filename, '-1')
    indexes, listofpartitions = partition_into_n_subgraphs(args.pkl_filename, '-1', args.number_of_partitions)

    df = pd.read_hdf(args.traffic_df_filename)

    #Renaming the df's column names
    numberofsensors=[]
    for i in range(df.shape[1]):
        numberofsensors.append(i)

    originalcolumnheaders = list(df.columns.values)

    df.columns = numberofsensors

    print("Generating partitions")
    '''
    #Making 3 deep copies of the original dataframe. One for each partition
    df0 = df.copy()
    df1 = df.copy()
    df2 = df.copy()
    '''

    #Making n deep copies of the orignal dataframe. One for each partition
    listofdf=[]
    for i in range(0,args.number_of_partitions):
        listofdf.append(df.copy())

    print("Putting each node in respective partition")
    '''
    #Deleting the columns from df0, df1 and df2 based on the partition lists
    for n in list0:
        del df1[n]
        del df2[n]
    for n in list1:
        del df0[n]
        del df2[n]
    for n in list2:
        del df0[n]
        del df1[n]
    '''
    for i in range(0,args.number_of_partitions):
        currentdf=listofdf[i]
        for j in range(0,len(listofpartitions)):
            if i!=j:
                for n in listofpartitions[j]:
                    del currentdf[n]

    '''
    generate_train_val_test(df0,'0',args)
    generate_train_val_test(df1,'1',args)
    generate_train_val_test(df2,'2',args)
    '''
    for i in range(0,args.number_of_partitions):
        generate_train_val_test(listofdf[i],str(i),args)


    #Saving the 4 lists as np.save(filename.npy,myList) files in predictions_dir
    print('Saving the sensor_ids to '+args.predictions_dir)
    np.save(args.predictions_dir+"/originalSensorIDs.npy",originalcolumnheaders)
    '''
    np.save(args.predictions_dir+"/sensorsInPartition0.npy",list0)
    np.save(args.predictions_dir+"/sensorsInPartition1.npy",list1)
    np.save(args.predictions_dir+"/sensorsInPartition2.npy",list2)
    '''
    for i in range(0,args.number_of_partitions):
        np.save(args.predictions_dir+"/sensorsInPartition"+str(i)+".npy",listofpartitions[i])

    print('Success...Exiting...')

    print("Copy the list to the config .yaml file under model--> num_nodes: '"+ str(indexes)+"'")


def main(args):
    print("Generating training data")
    generate_partitioned_data(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default=None,
        help="Raw traffic readings.",
    )
    parser.add_argument(
        "--pkl_filename",
        type=str,
        default=None,
        help="This pickle file is used to find the partions",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default=None,
        help="A new file will be created in this directory with the original sensor_ids and sensor_ids in each parition",
    )
    parser.add_argument(
        "--number_of_partitions",
        type=int,
        default=None,
        help="Enter the number of partitions required. Must match every where else",
    )
    args = parser.parse_args()
    main(args)
