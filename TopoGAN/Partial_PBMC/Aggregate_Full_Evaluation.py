"""
    This script will:
    1. Load all evaluation files
    2. Aggregate Percentage Cell Type Matches Epoch-wise
    3. Write output in a file
"""

import pandas as pd
import sys
method = "TopoGAN"
dataset = "Partial CiteSeq"
model_suffix = "Generation02_5_5_"

print("Aggregating results for: ", model_suffix)

# Define parameters

suffixes = [i for i in range(1, 11)]

data_dir = "F:/Akash/TU Delft/Thesis/Manifold Alignment/Full Evaluation Results/{}/{}".format(dataset, method)

print("=======================================================================")
print("Loading data . . .")
print("=======================================================================")

# Initialise file
columns = ["k",
           "Percentage sub cell_type matches","Percentage cell type matches",
           "Label transfer accuracy","Sub Label transfer accuracy",
           "Silhouette score"]

all_evaluations = []
for suffix in suffixes:
    Current_File = pd.read_csv("{}/Full Evaluation Results {}{}.csv".format(data_dir, model_suffix, suffix),
                               header=0, delimiter=',')
    Current_File["Experiment"] = suffix
    print(Current_File)
    all_evaluations.append(Current_File)

Aggregate_Results = pd.concat(all_evaluations)
#Aggregate_Results = pd.DataFrame(all_evaluations, columns=columns)

Aggregate_Results.to_csv(
    path_or_buf="{}/Aggregate Full Evaluation Results {}.csv".format(data_dir, model_suffix), header=True, index=False)


