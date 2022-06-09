# NGAFIDDATASET

This github repository contains code related to the paper A Large-Scale Annotated Multivariate Time Series Aviation Maintenance Dataset from the NGAFID. 

There are two notebooks for reproducing experiments and one example notebook for viewing all flight data using dask dataframes. 

The repository notebooks automatically download files hosted on Google Drive to run the benchmark experiments. 

The full dataset can be downloaded from https://doi.org/10.5281/zenodo.6624956 or https://www.kaggle.com/datasets/hooong/aviation-maintenance-dataset-from-the-ngafid. 

# Accessing Full Dataset using Colab

Please run the Dask Example notebook in the repository https://github.com/hyang0129/NGAFIDDATASET/blob/main/NGAFID_DATASET_DASK_EXAMPLE.ipynb.

In terms of data structure, the flight header dataframe uses the master index column to link to the index of the flight data dask dataframe. 

If you wish to use this for machine learning, it is best to extract the relevant flights into a format that works with your framework. The benchmark experiments use an extracted version stored as numpy arrays. 

# Running Benchmark Experiments using Colab

There are two setups for the benchmark experiments. 

To train InceptionTime or ConvMHSA models, run the notebook https://github.com/hyang0129/NGAFIDDATASET/blob/main/NGAFID_DATASET_TF_EXAMPLE.ipynb. 

To train MiniRocket, run the notebook https://github.com/hyang0129/NGAFIDDATASET/blob/main/NGAFID_DATASET_MINIROCKET_EXAMPLE.ipynb

# Accessing Full Dataset From Sources 

Download from https://doi.org/10.5281/zenodo.6624956 or https://www.kaggle.com/datasets/hooong/aviation-maintenance-dataset-from-the-ngafid. 

Unzip the all_flights.tar.gz file. Then use python to access the dask dataframe. 

```
import dask.dataframe as dd
import pandas as pd 

flight_data_df = dd.read_parquet('all_flights/one_parq')

flight_header_df = pd.read_csv('all_flights/flight_header.csv', index_col = 'Master Index')

```

# Data Structure 

The flight header df has the following columns. 

![image](https://user-images.githubusercontent.com/34040987/172897265-af793b45-9aa7-43f7-b0e3-b9a6c8b8c649.png)

The flight data df has the following columns below and a timestep column, which describes the order of the rows within a flight, determined by the index. The order is from low to high (0 means first timestep). 

![image](https://user-images.githubusercontent.com/34040987/172897391-24cb8664-4c81-4987-8229-7fcb0b6e93d9.png)

Below is an sample of data from the flight data df. Due to dask partitioning, it only guarantees partition ordering along the Index column, but not the timesteps column. Please sort by the timesteps when you extract a flight. 

![image](https://user-images.githubusercontent.com/34040987/172897809-01670e16-9dfa-4216-9682-938ebc1a88b8.png)





