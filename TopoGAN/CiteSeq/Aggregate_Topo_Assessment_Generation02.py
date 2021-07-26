"""
    This script will:
    1. Load all evaluation files
    2. Aggregate Percentage Cell Type Matches Epoch-wise
    3. Write output in a file
"""

import pandas as pd
import sys
model_suffix = str(sys.argv[1])

print("Aggregating results for: ", model_suffix)

# Define parameters
pairs = ["10_10", "10_11"]
suffixes = [i for i in range(1, 11)]

data_dir = "/home/nfs/asingh5/Unsupervised_Manifold_Alignment/GAN_Ensemble_Topology_Experiments/CiteSeq/Evaluation Results"

print("=======================================================================")
print("Loading data . . .")
print("=======================================================================")

# Initialise file
Aggregate_Results = pd.read_csv("{}/Evaluation Results Dummy.csv".format(data_dir), header=0, delimiter=',', index_col = 0)

for pair in pairs:
    for suffix in suffixes:
        Current_File = pd.read_csv("{}/Topological Assessment {}_{}_{}.csv".format(data_dir, model_suffix, pair, suffix), header=0, delimiter=',', index_col = 0)
        Aggregate_Results["Source to Projected {}_{}".format(pair, suffix)] = Current_File["Source to Projected Topology Loss"]

Aggregate_Results.to_csv(
    path_or_buf="{}/Aggregate Topology Assessment {}.csv".format(data_dir, model_suffix), header=True, index=False)


