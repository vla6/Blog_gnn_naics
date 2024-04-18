# Blog_gnn_naics
A project exploring categorical feature encodings/embeddings, especially with graph neural networks techniques.  

## Towards Data Science "Exploring Hierarchical Blending in Target Encoding"

Table data is in the top level in the "tables.xlsx" document.  

Code is at the top level; notebooks would run in order.  Metrics are collected and summarized in 80_perf_summary.ipynb.

## Running Code

First, download the [SBA Loans Dataset from Kaggle](https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied).

Then, change setup.py
  * Make input_path point to the SBA Loans dataset on your system
  * temp_path should point to a writeable directory on your system

## Hardware and GPUs

Everything runs on my home hardware (Mac Book Air).  GNN methods use [Stellargraph](https://stellargraph.readthedocs.io/en/stable/), which requires certain Python versions; I use 3.8.18.  

Stellargraph uses keras/tensorflow.  Here are some relevant package versions
* keras 2.15.0
* networkx 3.1
* tensorflow 2.15.0
* tensorflow-macos 2.15.0 (mac users only)
* tensorflow-metal 1.1.0  (mac users only)
* xgboost  1.7.5

The full pip list output is also in the top level of this repository.

Because I use a Mac, I had to install tensorflow metal for neural networks and graph neural networks.  Here is a helpful article: https://medium.com/@angelgaspar/how-to-install-tensorflow-on-a-m1-m2-macbook-with-gpu-acceleration-acfeb988d27e

