
import gdown
import pandas as pd
import numpy as np
from sklearn import preprocessing
from tqdm.autonotebook import tqdm
import tensorflow as tf
from loguru import logger
from ngafiddataset.utils import shell_exec
import os
import tarfile
import shutil
import typing
from compress_pickle import load

from ngafiddataset.dataset.utils import *



class NGAFID_Dataset_Downloader:

    ngafid_urls = {
        "all_flights": "https://drive.google.com/uc?id=1-0pVPhwRQoifT_VuQyGDLXuzYPYySX-Y",
        "2days": "https://drive.google.com/uc?id=1-2pxwiQNhFnhTg7whosQoF_yztD5jOM2",
    }

    @classmethod
    def download(cls, name : str, destination:str = '', extract = True):

        assert name in cls.ngafid_urls.keys()

        url = cls.ngafid_urls[name]
        output =  os.path.join(destination, "%s.csv.gz" % name)

        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)

        if extract:
            logger.info('Extracting File')
            _ = tarfile.open(output).extractall(destination)

        return name, destination


class NGAFID_Dataset_Manager(NGAFID_Dataset_Downloader):

    channels = 23

    def __init__(self, name: str, destination:str = '',  max_length = 4096, extract = True, **kwargs):
        assert name in self.ngafid_urls.keys()

        if name == 'all_flights':
            logger.info('Downloading and extracting Parquet Files to %s\one_parq.  Please open them using dask dataframes' % destination)

            self.download(name, destination, extract = True)

        else:

            self.name = name
            self.max_length = max_length
            self.destination = destination

            self.files = ['flight_data.pkl', 'flight_header.csv', 'stats.csv']
            self.files = {file : os.path.join(destination, name, file) for file in self.files}

            self.download(name, destination, extract)

            self.flight_header_df = pd.read_csv(self.files['flight_header.csv'],  index_col='Master Index')
            self.flight_data_array = load(self.files['flight_data.pkl'])
            self.flight_stats_df = pd.read_csv(self.files['stats.csv'])

            # self.data_dict = self.construct_data_dictionary()

            self.maxs = self.flight_stats_df.iloc[0, 1:24].to_numpy(dtype = np.float32)
            self.mins = self.flight_stats_df.iloc[1, 1:24].to_numpy(dtype = np.float32)

    def construct_data_dictionary(self):
        data_dict = []

        for index, row in tqdm(self.flight_header_df.iterrows(), total = len(self.flight_header_df)):

            # pad array
            arr = np.zeros((self.max_length, self.channels), dtype = np.float16)
            to_pad = self.flight_data_array[index][-self.max_length:, :]
            arr[:to_pad.shape[0], :] += to_pad
            arr = tf.convert_to_tensor(arr, dtype = tf.bfloat16)

            data_dict.append({'id': index,
                              'data': arr,
                              'cluster': row.cluster,
                              'class': row['class'],
                              'fold': row['fold'],
                              'target_class': row['target_class'],
                              'before_after': row['before_after'],
                              'hclass': row['hclass']})

        return data_dict

    def get_tf_datset(self, fold = 0, training = False, shuffle = False, batch_size = 64, repeat = False,
                        mode = 'before_after'):

        ds = tf.data.Dataset.from_tensor_slices(to_dict_of_list(get_slice(self.data_dict, fold = fold, reverse = training)))

        ds = ds.repeat() if repeat else ds
        ds = ds.shuffle(shuffle) if shuffle else ds

        ds = ds.map(get_dict_mod('data', get_scaler(self.maxs, self.mins)))
        ds = ds.map(get_dict_mod('data', replace_nan_w_zero))
        ds = ds.map(get_dict_mod('data', lambda x: tf.cast(x, tf.float32)))

        if mode == 'before_after':
            ds = ds.map(lambda x: (x['data'], x['before_after']))
        elif mode == 'classes':
            ds = ds.map(lambda x: (x['data'], x['target_class']))
        elif mode == 'both':
            ds = ds.map(lambda x: (
            {'data': x['data']}, {'before_after': x['before_after'], 'target_class': x['target_class']}))
        elif mode == 'hierarchy_basic':
            ds = ds.map(
                lambda x: ({'data': x['data']}, {'before_after': x['before_after'], 'target_class': x['hclass']}))
        else:
            raise KeyError

        ds = ds.batch(batch_size, drop_remainder = True) if batch_size else ds

        return ds