"""
    This script will:
    1. Import required modules
    2. Define required parameters
    3. Load data
    4. Load Generator Networks as specified
	5. Train and store ensemble model
	6. Evaluate stored ensemble model
"""

from Manifold_Alignment_GAN_Ensemble import *
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

# Define parameters

source_tech_name = "ADT"
label = "L1"
sub_label = "L2"
#epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
epochs = [800, 900, 1000]

####################################################
# MAKE SURE path suffix and path prefix are correct
####################################################

path_suffix = "Generation02_{}_{}_{}".format(model_01, model_02, str(random_seed))

data_dir = "/home/nfs/asingh5/Unsupervised_Manifold_Alignment/Data/CiteSeq Data"
ensemble_path_prefix = "/home/nfs/asingh5/Unsupervised_Manifold_Alignment/GAN_Ensemble_Topology_Experiments/CiteSeq"

print("=======================================================================")
print("Loading data . . .")
print("=======================================================================")

# Load data from all domains

if source_tech_name == "RNA":
  target_tech_name = "ADT"

else:
  target_tech_name = "RNA"

source_tech = Manifold_Data = pd.read_csv(
    "{}/{} CiteSeq TopoAE 8 Dimensions Partial Small.csv".format(data_dir, source_tech_name)
    , header=0, delimiter=',', index_col = 0)

target_tech = Manifold_Data = pd.read_csv(
    "{}/{} CiteSeq TopoAE 8 Dimensions Partial Small.csv".format(data_dir, target_tech_name)
    , header=0, delimiter=',', index_col = 0)

features_threshold_source = len(source_tech.columns)
features_threshold_target = len(target_tech.columns)
Annotation_Data_Source = pd.read_csv("{}/{}_annotations_Partial.csv".format(data_dir, source_tech_name)
, header=0, delimiter=',', index_col = 0)
Annotation_Data_Target = pd.read_csv("{}/{}_annotations_Partial.csv".format(data_dir, target_tech_name)
, header=0, delimiter=',', index_col = 0)

Source_Data = source_tech.merge(Annotation_Data_Source, how='inner', left_index=True, right_index=True)
Target_Data = target_tech.merge(Annotation_Data_Target, how='inner', left_index=True, right_index=True)

source_tech = source_tech.to_numpy()
source_tech = source_tech.astype(float)
print("Source Technology: ", np.shape(source_tech))
target_tech = target_tech.to_numpy()
target_tech = target_tech.astype(float)
print("Target Technology: ", np.shape(target_tech))

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

for epoch in epochs:

    print("#########################################################")
    print("#########################################################")
    print("")
    print("Evaluating for epoch: ", epoch)
    path = "{}/Models/{}_to_{}_Generator_{}_{}.pt".format(ensemble_path_prefix,
                                                             source_tech_name,
                                                             target_tech_name,
                                                             epoch,
                                                             path_suffix)
    print("Model path: ", path)


    # Project source into target space
    model = torch.load(path)  # Load trained model
    source_to_target = torch.empty(0)

    source_tech = torch.tensor(source_tech).float()
    for tensor in source_tech:
        projected_tensor = model(tensor).reshape(1, 8)
        source_to_target = torch.cat((source_to_target, projected_tensor))

    source_to_target = source_to_target.detach()
    print("source data shape: ", source_to_target.shape)

    # Convert to dataframes
    source_indices = list(Source_Data.index.values)
    source_to_target = pd.DataFrame(source_to_target, index=source_indices).astype("float")

    target_indices = list(Target_Data.index.values)
    target_tech = pd.DataFrame(target_tech, index=target_indices).astype("float")

    # Annotate data

    source_to_target_annotated = source_to_target.merge(Annotation_Data_Source, how='inner', left_index=True, right_index=True)
    target_tech_annotated = target_tech.merge(Annotation_Data_Target, how='inner', left_index=True, right_index=True)

    k_array = [5, 50, 100]
    #k_array = [5]
    print("Total cells: ", len(target_tech))
    for k in k_array:
        print("############################################################")
        print("")
        print("Evaluating for k = ", k)
        print("")
        results = evaluate_first_k_neighbors(source_to_target, source_to_target_annotated,
                                             target_tech, target_tech_annotated, k,
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
    path_or_buf="{}/Evaluation Results/Full Evaluation Results {}.csv".format(ensemble_path_prefix, path_suffix),
    header=True, index=False)