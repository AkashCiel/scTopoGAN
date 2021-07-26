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
from src.models.Assess_Topology_V02 import *

# Define parameters

batch_size = 50
num_classes = 8
source_tech_name = "ADT"
latent_dimensions = 8
topology_weight = 2.0 # There are multiple latent projections of CiteSeq data, defined by topology weight
topology_batch_size = 1000
epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#epochs = [100]

learning_rate = 0.0001
checkpoint_epoch = 100

total_epochs = 1001
#total_epochs = 101

####################################################
# MAKE SURE path suffix and path prefix are correct
####################################################

core_suffix = "Generation01"
path_suffix = "Generation02_{}_{}_{}".format(model_01, model_02, str(random_seed))

model_01_name = "ADT_to_RNA_Generator_{}_{}_MODEL_{}".format(epoch_01, core_suffix, model_01)
model_02_name = "ADT_to_RNA_Generator_{}_{}_MODEL_{}".format(epoch_02, core_suffix, model_02)

data_dir = "/home/nfs/asingh5/Unsupervised_Manifold_Alignment/Data/CiteSeq Data"
load_path_prefix = "/home/nfs/asingh5/Unsupervised_Manifold_Alignment/GAN_Ensemble_Topology_Experiments/CiteSeq"
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
    "{}/{} TopoAE 8 Dimensions {}.csv".format(data_dir, source_tech_name, topology_weight)
    , header=0, delimiter=',', index_col = 0)

target_tech = Manifold_Data = pd.read_csv(
    "{}/{} TopoAE 8 Dimensions {}.csv".format(data_dir, target_tech_name, topology_weight)
    , header=0, delimiter=',', index_col = 0)

features_threshold_source = len(source_tech.columns)
features_threshold_target = len(target_tech.columns)
Annotation_Data_Source = pd.read_csv("{}/{}_annotations.csv".format(data_dir, source_tech_name)
, header=0, delimiter=',', index_col = 0)
Annotation_Data_Target = pd.read_csv("{}/{}_annotations.csv".format(data_dir, target_tech_name)
, header=0, delimiter=',', index_col = 0)

Source_Data = source_tech.merge(Annotation_Data_Source, how='inner', left_index=True, right_index=True)
Target_Data = target_tech.merge(Annotation_Data_Target, how='inner', left_index=True, right_index=True)

source_tech = source_tech.to_numpy()
source_tech = source_tech.astype(float)
print("Source Technology: ", np.shape(source_tech))
target_tech = target_tech.to_numpy()
target_tech = target_tech.astype(float)
print("Target Technology: ", np.shape(target_tech))

source_target_numpy = np.concatenate((source_tech, target_tech), axis=1)
source_target_tensor = torch.tensor(source_target_numpy).float() # Convert to tensor

# Define data loader
data_loader = DataLoader(source_target_tensor, batch_size=batch_size, shuffle=False, num_workers=0)

# Train ensemble model

print("=======================================================================")
print("Training ")
print("=======================================================================")

path_01 = "{}/Models/{}.pt".format(load_path_prefix, model_01_name)
path_02 = "{}/Models/{}.pt".format(load_path_prefix, model_02_name)

# Load models and create ensemble Generator Net
generator_1 = torch.load(path_01)
generator_2 = torch.load(path_02)
generator_ensemble = GeneratorNet(input_dim=latent_dimensions, output_dim=latent_dimensions)

beta = 0.5 #The mixing factor
params1 = generator_1.named_parameters()
params2 = generator_2.named_parameters()
params_ensemble = generator_ensemble.named_parameters()

dict_params1 = dict(params1)
dict_params2 = dict(params2)
dict_params_ensemble = dict(params_ensemble)

for name1 in dict_params_ensemble:
    dict_params_ensemble[name1].data.copy_(beta * dict_params1[name1].data + (1 - beta) * dict_params2[name1].data)

generator_ensemble.load_state_dict(dict_params_ensemble)

discriminator = DiscriminatorNet(input_dim=latent_dimensions)
techs = [source_tech_name, target_tech_name]

trained_generator = train(generator_ensemble, discriminator, data_loader,
                          total_epochs, checkpoint_epoch,
                          learning_rate, techs, ensemble_path_prefix, path_suffix)

# Define necessary functions for evaluation
def get_majority_local_category(target_tech_annotated, indices, category):
    categories = [target_tech_annotated.iloc[[index]][category].item() for index in indices]
    return max(categories, key=categories.count)


def evaluate_first_k_neighbors(source_to_target, source_to_target_annotated,
                               target_tech, target_tech_annotated, k):
    # Perfom knn estimation
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(target_tech)  # Dataset to find the neighbours in
    all_distances, all_indices = neighbors.kneighbors(source_to_target)  # Query dataset

    sub_cell_type_matches = []
    cell_type_matches = []
    for i in range(len(all_indices)):
        if (i % 500) == 0:
            print('\r', "Evaluating sample #{}".format(i), end='')
        sub_cell_type_source = source_to_target_annotated.iloc[[i]]["L2"].item()
        cell_type_source = source_to_target_annotated.iloc[[i]]["L1"].item()

        sub_cell_type_neighbors = get_majority_local_category(target_tech_annotated, all_indices[i], "L2")
        cell_type_neighbors = get_majority_local_category(target_tech_annotated, all_indices[i], "L1")

        sub_cell_type_matches.append(1 * (sub_cell_type_source == sub_cell_type_neighbors))
        cell_type_matches.append(1 * (cell_type_source == cell_type_neighbors))

    evaluation = {"Percentage sub cell_type matches": 100 * (sum(sub_cell_type_matches) / len(sub_cell_type_matches)),
                  "Percentage cell type matches": 100 * (sum(cell_type_matches) / len(cell_type_matches))}

    return evaluation

Topo_Evaluation_Columns = ["Model", "Source", "Target", "Iteration", "Epoch",
                      "Source to Projected Topology Loss"]

Evaluation_Columns = ["Model", "Source", "Target", "Epoch", "k",
                      "Percentage sub cell_type matches",
                      "Percentage cell type matches"]

Topo_evaluation_results = []

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

    source_to_target = source_to_target.detach().numpy()
    print("source data shape: ", source_to_target.shape)
    source_projected_numpy = np.concatenate((source_tech, source_to_target), axis=1)
    source_projected_tensor = torch.tensor(source_projected_numpy).float()  # Convert to tensor
    # Define data loader
    data_loader_source_projected = DataLoader(source_projected_tensor, batch_size=topology_batch_size, shuffle=False, num_workers=0)

    # Evaluate topology between source and projected
    total_topo_loss_source_projected = 0
    for n_batch, data in enumerate(data_loader_source_projected):
        first_dim = data.shape[0]
        source_batch = tf.slice(data, [0, 0], [first_dim, 8])
        source_to_target_batch = tf.slice(data, [0, 8], [first_dim, 8])

        # Convert source and target from eager tensor to native tensor
        source_to_target_batch = source_to_target_batch.numpy()
        source_to_target_batch = torch.tensor(source_to_target_batch)
        source_batch = source_batch.numpy()
        source_batch = torch.tensor(source_batch)

        topo_error = compute_topological_error(source_batch, source_to_target_batch)
        total_topo_loss_source_projected += topo_error.item()

    Topo_evaluation_results.append(["GAN", source_tech_name, target_tech_name, path_suffix, epoch,
                               total_topo_loss_source_projected])


# Save evaluation results
Topo_evaluation_results = pd.DataFrame(Topo_evaluation_results, columns=Topo_Evaluation_Columns)
Topo_evaluation_results.to_csv(
    path_or_buf="{}/Evaluation Results/Topological Assessment {}.csv".format(ensemble_path_prefix, path_suffix),
    header=True, index=False)

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

    #k_array = [1, 5, 10, 15, 20, 50, 100]
    k_array = [5]
    print("Total cells: ", len(target_tech))
    for k in k_array:
        print("############################################################")
        print("")
        print("Evaluating for k = ", k)
        print("")
        results = evaluate_first_k_neighbors(source_to_target, source_to_target_annotated, target_tech, target_tech_annotated, k)
        print('')
        print(results)

        evaluation_results.append(["GAN", source_tech_name, target_tech_name, epoch, k,
                                   results["Percentage sub cell_type matches"],
                                   results["Percentage cell type matches"]])

# Save evaluation results
evaluation_results = pd.DataFrame(evaluation_results, columns=Evaluation_Columns)
evaluation_results.to_csv(
    path_or_buf="{}/Evaluation Results/Evaluation Results_{}.csv".format(ensemble_path_prefix, path_suffix),
    header=True, index=False)