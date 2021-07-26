"""
    This script will:
        1. Load data
        2. Implement UnionCom
        3. Save and evaluate integrated data
"""

# Import modules
import pandas as pd
import torchvision
from unioncom import UnionCom
from sklearn.neighbors import NearestNeighbors
import numpy as np
import sys

random_seed = 42
print("Random seed : {}".format(random_seed))

# Define parameters
source_tech_name = "ADT"
epochs = 1000
path_suffix = "Reduced_from_50_{}_{}_Partial_Small".format(epochs, random_seed)
path_prefix = "Unsupervised_Manifold_Alignment/UnionCom"
data_dir = "Unsupervised_Manifold_Alignment/Data/CiteSeq Data"

print("=======================================================================")
print("Loading data . . .")
print("=======================================================================")

# Load data from all domains
if source_tech_name == "RNA":
  target_tech_name = "ADT"

else:
  target_tech_name = "RNA"

source_tech = pd.read_csv("{}/{}_PCA_Embeddings Partial Small.csv".format(data_dir, source_tech_name), header=0, delimiter=',', index_col = 0)

target_tech = pd.read_csv("{}/{}_PCA_Embeddings Partial Small.csv".format(data_dir, target_tech_name), header=0, delimiter=',', index_col = 0)

source_indices = list(source_tech.index)
target_indices = list(target_tech.index)

Annotation_Data_Source = pd.read_csv("{}/{}_annotations_Partial.csv".format(data_dir, source_tech_name)
, header=0, delimiter=',', index_col = 0)
Annotation_Data_Target = pd.read_csv("{}/{}_annotations_Partial.csv".format(data_dir, target_tech_name)
, header=0, delimiter=',', index_col = 0)

# Add source labels
Annotation_Data_Source["source"] = Annotation_Data_Source.apply(lambda row: source_tech_name, axis=1)
Annotation_Data_Target["source"] = Annotation_Data_Target.apply(lambda row: target_tech_name, axis=1)

# Append Annotation Data
Annotation_Integrated = Annotation_Data_Target.append(Annotation_Data_Source)

source_tech = source_tech.to_numpy()
source_tech = source_tech.astype(float)
print("Source Technology: ", np.shape(source_tech))
target_tech = target_tech.to_numpy()
target_tech = target_tech.astype(float)
print("Target Technology: ", np.shape(target_tech))

uc = UnionCom.UnionCom(epoch_pd=epochs, epoch_DNN=200, manual_seed=random_seed, output_dim=8)
integrated_data = uc.fit_transform(dataset=[source_tech,target_tech])

tech_01_projected = pd.DataFrame(integrated_data[0], index=source_indices).astype("float")
tech_02_projected = pd.DataFrame(integrated_data[1], index=target_indices).astype("float")

# Annotate data
tech_01_projected_annotated = tech_01_projected.merge(Annotation_Data_Source, how='inner', left_index=True, right_index=True)
tech_02_projected_annotated = tech_02_projected.merge(Annotation_Data_Target, how='inner', left_index=True, right_index=True)

# Write to file
tech_01_projected.to_csv(path_or_buf="{}/CiteSeq Integrated Data/UnionCom_Integrated_{}_{}.csv".format(path_prefix, path_suffix, source_tech_name), header=True, index=True)
tech_02_projected.to_csv(path_or_buf="{}/CiteSeq Integrated Data/UnionCom_Integrated_{}_{}.csv".format(path_prefix, path_suffix, target_tech_name), header=True, index=True)

def get_majority_local_category(tech_02_projected_annotated, indices, category):
    categories = [tech_02_projected_annotated.iloc[[index]][category].item() for index in indices]
    return max(categories, key=categories.count)

def evaluate_first_k_neighbors(tech_01_projected, tech_01_projected_annotated,
                               tech_02_projected, tech_02_projected_annotated, k):
    # Perfom knn estimation
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(tech_02_projected)  # Dataset to find the neighbours in
    all_distances, all_indices = neighbors.kneighbors(tech_01_projected)  # Query dataset

    exact_matches = []
    sub_cell_type_matches = []
    cell_type_matches = []
    for i in range(len(all_indices)):
        if (i % 500) == 0:
            print('\r', "Evaluating sample #{}".format(i), end='')
        sub_cell_type_source = tech_01_projected_annotated.iloc[[i]]["L2"].item()
        cell_type_source = tech_01_projected_annotated.iloc[[i]]["L1"].item()

        sub_cell_type_neighbors = get_majority_local_category(tech_02_projected_annotated, all_indices[i], "L2")
        cell_type_neighbors = get_majority_local_category(tech_02_projected_annotated, all_indices[i], "L1")

        exact_matches.append(1 * (i in all_indices[i]))
        sub_cell_type_matches.append(1 * (sub_cell_type_source == sub_cell_type_neighbors))
        cell_type_matches.append(1 * (cell_type_source == cell_type_neighbors))

    evaluation = {"Percentage exact matches": 100 * (sum(exact_matches) / len(exact_matches)),
                  "Percentage sub cell_type matches": 100 * (sum(sub_cell_type_matches) / len(sub_cell_type_matches)),
                  "Percentage cell type matches": 100 * (sum(cell_type_matches) / len(cell_type_matches))}

    return evaluation


Evaluation_Columns = ["Model", "Tech 01", "Tech 02", "k",
                      "Percentage Exact Matches",
                      "Percentage sub cell_type matches",
                      "Percentage cell type matches"]

evaluation_results = []

#k_array = [1, 5, 10, 15, 20, 50, 100, 200]
k_array = [5]

print("Total cells: ", len(tech_01_projected))
for k in k_array:
    print("############################################################")
    print("")
    print("Evaluating for k = ", k)
    print("")
    results = evaluate_first_k_neighbors(tech_01_projected, tech_01_projected_annotated,
                                         tech_02_projected, tech_02_projected_annotated,
                                         k)
    print('')
    print(results)

    evaluation_results.append(["UnionCom", source_tech_name, target_tech_name, k,
                               results["Percentage exact matches"],
                               results["Percentage sub cell_type matches"],
                               results["Percentage cell type matches"]])

# Save evaluation results
evaluation_results = pd.DataFrame(evaluation_results, columns=Evaluation_Columns)
evaluation_results.to_csv(
    path_or_buf="{}/CiteSeq Evaluation Results/Evaluation Results_{}.csv".format(path_prefix, path_suffix),
    header=True, index=False)
