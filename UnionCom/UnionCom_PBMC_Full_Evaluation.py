"""
    This script will:
    1. Import required modules
    2. Define required parameters
    3. Load data
    4. Perform evaluation on ALL metrics discussed in the report
    5. Save evaluation results
"""
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np
import sys
random_seed = int(sys.argv[1])
# Define parameters

dataset = "PBMC"
tech_01 = "ATAC"
tech_02 = "RNA"
label = "cell_type"
sub_label = "cell_type_00"

####################################################
# MAKE SURE path suffix and path prefix are correct
####################################################

annotations_data_dir = "Unsupervised_Manifold_Alignment/Data/PBMC Data"
results_data_dir = "Unsupervised_Manifold_Alignment/UnionCom/{} Evaluation Results".format(dataset)
integrated_data_dir = "Unsupervised_Manifold_Alignment/UnionCom/{} Integrated Data".format(dataset)

print("=======================================================================")
print("Loading data . . .")
print("=======================================================================")

# Load data

tech_01_projected = pd.read_csv("{}/UnionCom_Integrated_Reduced_from_50_1000_{}_{}.csv".format(integrated_data_dir, random_seed, tech_01)
, header=0, delimiter=',', index_col = 0)

tech_02_projected = pd.read_csv("{}/UnionCom_Integrated_Reduced_from_50_1000_{}_{}.csv".format(integrated_data_dir, random_seed, tech_02)
, header=0, delimiter=',', index_col = 0)

Annotation_Data_01 = pd.read_csv("{}/Annotations_{}.csv".format(annotations_data_dir, tech_01)
, header=0, delimiter=',', index_col = 0)
Annotation_Data_02 = pd.read_csv("{}/Annotations_{}.csv".format(annotations_data_dir, tech_02)
, header=0, delimiter=',', index_col = 0)

tech_01_projected_annotated = tech_01_projected.merge(Annotation_Data_01, how='inner', left_index=True, right_index=True)
tech_02_projected_annotated = tech_02_projected.merge(Annotation_Data_02, how='inner', left_index=True, right_index=True)


# Define necessary functions for evaluation
def calc_sil(x1_mat, x2_mat, x1_lab, x2_lab):  # function to calculate Silhouette scores

    x = np.concatenate((x1_mat, x2_mat))
    lab = np.concatenate((x1_lab, x2_lab))

    sil_score = silhouette_samples(x, lab)
    avg = np.mean(sil_score)

    return avg

def transfer_accuracy(train_tech, test_tech, train_labels, test_labels, k):
    knn = KNeighborsClassifier(k)
    knn.fit(train_tech, train_labels)
    test_label_predict = knn.predict(test_tech)
    count = 0
    for label1, label2 in zip(test_label_predict, test_labels):
      if label1 == label2:
        count += 1
    return 100*(count / len(test_labels))

def get_majority_local_category(target_tech_annotated, indices, category):
    categories = [target_tech_annotated.iloc[[index]][category].item() for index in indices]
    return max(categories, key=categories.count)

def evaluate_first_k_neighbors(source_to_target, source_to_target_annotated,
                               target_tech, target_tech_annotated, k,
                               label, sub_label):
    # Perfom knn estimation
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(target_tech)  # Dataset to find the neighbours in
    all_distances, all_indices = neighbors.kneighbors(source_to_target)  # Query dataset

    sub_cell_type_matches = []
    cell_type_matches = []
    for i in range(len(all_indices)):
        if (i % 500) == 0:
            print('\r', "Evaluating sample #{}".format(i), end='')
        sub_cell_type_source = source_to_target_annotated.iloc[[i]][sub_label].item()
        cell_type_source = source_to_target_annotated.iloc[[i]][label].item()

        sub_cell_type_neighbors = get_majority_local_category(target_tech_annotated, all_indices[i], sub_label)
        cell_type_neighbors = get_majority_local_category(target_tech_annotated, all_indices[i], label)

        sub_cell_type_matches.append(1 * (sub_cell_type_source == sub_cell_type_neighbors))
        cell_type_matches.append(1 * (cell_type_source == cell_type_neighbors))

    train_labels = list(target_tech_annotated[label])
    train_sub_labels = list(target_tech_annotated[sub_label])
    test_labels = list(source_to_target_annotated[label])
    test_sub_labels = list(source_to_target_annotated[sub_label])
    label_transfer_accuracy = transfer_accuracy(target_tech, source_to_target,
                                                train_labels, test_labels, k)
    sub_label_transfer_accuracy = transfer_accuracy(target_tech, source_to_target,
                                                    train_sub_labels, test_sub_labels, k)

    sil1 = calc_sil(target_tech, source_to_target,
                    train_labels, test_labels)
    sil2 = calc_sil(source_to_target, target_tech,
                    test_labels, train_labels)
    silhouette_score = (sil1+sil2)/2

    evaluation = {"Percentage sub cell_type matches": 100 * (sum(sub_cell_type_matches) / len(sub_cell_type_matches)),
                  "Percentage cell type matches": 100 * (sum(cell_type_matches) / len(cell_type_matches)),
                  "Label transfer accuracy" : label_transfer_accuracy,
                  "Sub Label transfer accuracy" : sub_label_transfer_accuracy,
                  "Silhouette score" : silhouette_score}

    return evaluation


Evaluation_Columns = ["Model", "k",
                      "Percentage sub cell_type matches",
                      "Percentage cell type matches",
                      "Label transfer accuracy",
                      "Sub Label transfer accuracy",
                      "Silhouette score"]

evaluation_results = []

print("#########################################################")
print("#########################################################")
print("")

k_array = [5, 50, 100]
#k_array = [5]
print("Total cells: ", len(tech_01_projected))
for k in k_array:
    print("############################################################")
    print("")
    print("Evaluating for k = ", k)
    print("")
    results = evaluate_first_k_neighbors(tech_01_projected, tech_01_projected_annotated,
                                         tech_02_projected, tech_02_projected_annotated, k,
                                         label, sub_label)
    print('')
    print(results)

    evaluation_results.append(["TopoGAN", k,
                            results["Percentage sub cell_type matches"],
                            results["Percentage cell type matches"],
                           results["Label transfer accuracy"],
                           results["Sub Label transfer accuracy"],
                           results["Silhouette score"]])

# Save evaluation results
evaluation_results = pd.DataFrame(evaluation_results, columns=Evaluation_Columns)
evaluation_results.to_csv(
    path_or_buf="{}/Full Evaluation Results PBMC {}.csv".format(results_data_dir, random_seed),
    header=True, index=False)