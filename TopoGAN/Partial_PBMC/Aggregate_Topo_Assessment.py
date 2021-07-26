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

suffixes = [i for i in range(1, 21)]

data_dir = "/home/nfs/asingh5/Unsupervised_Manifold_Alignment/GAN_Ensemble_Topology_Experiments/Partial_PBMC/Evaluation Results"

print("=======================================================================")
print("Loading data . . .")
print("=======================================================================")

# Initialise file
Aggregate_Results = pd.read_csv("{}/Evaluation Results Dummy.csv".format(data_dir), header=0, delimiter=',', index_col = 0)

for suffix in suffixes:
    Current_File = pd.read_csv("{}/Topological Assessment {}_MODEL_{}.csv".format(data_dir, model_suffix, suffix), header=0, delimiter=',', index_col = 0)
    Aggregate_Results["Source to Projected {}".format(suffix)] = Current_File["Source to Projected Topology Loss"]

col = Aggregate_Results.loc[: , "Source to Projected 1":"Source to Projected 20"]
Aggregate_Results['Mean'] = col.mean(axis=1)
Aggregate_Results['Std'] = col.std(axis=1)

Aggregate_Results.to_csv(
    path_or_buf="{}/Aggregate Topology Assessment {}.csv".format(data_dir, model_suffix), header=True, index=False)


