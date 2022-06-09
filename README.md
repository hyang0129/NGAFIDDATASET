# NGAFIDDATASET

This github repository contains code related to the paper A Large-Scale Annotated Multivariate Time Series Aviation Maintenance Dataset from the NGAFID. 

There are two notebooks for reproducing experiments and one example notebook for viewing all flight data using dask dataframes. 

The repository notebooks automatically download files hosted on Google Drive to run the benchmark experiments. 

# Accessing Full Dataset 

Please run the Dask Example notebook in the repository https://github.com/hyang0129/NGAFIDDATASET/blob/main/NGAFID_DATASET_DASK_EXAMPLE.ipynb.

In terms of data structure, the flight header dataframe uses the master index column to link to the index of the flight data dask dataframe. 

If you wish to use this for machine learning, it is best to extract the relevant flights into a format that works with your framework. The benchmark experiments use an extracted version stored as numpy arrays. 

# Running Benchmark Experiments 

There are two setups for the benchmark experiments. 

To train InceptionTime or ConvMHSA models, run the notebook https://github.com/hyang0129/NGAFIDDATASET/blob/main/NGAFID_DATASET_TF_EXAMPLE.ipynb. 

To train MiniRocket, run the notebook https://github.com/hyang0129/NGAFIDDATASET/blob/main/NGAFID_DATASET_MINIROCKET_EXAMPLE.ipynb




