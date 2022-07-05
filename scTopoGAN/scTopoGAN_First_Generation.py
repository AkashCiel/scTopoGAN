"""
    This script will:
    1. Train multiple (default 20) models with different initialisations
    2. Evaluate average topological error per model after 500 epochs
    3. Write all average topological errors in a csv file
    4. Provide recommended models to select for the next generation, based on topological error calculation
"""

# <editor-fold desc="Import modules">
import pandas as pd
import os
from Manifold_Alignment_GAN import *
from src.models.Assess_Topology_V02 import *
# </editor-fold>

# <editor-fold desc="Define parameters">
core_suffix = "scTopoGAN_Generation01"
data_dir = "PBMC"
path_prefix = "Results" # This is where all the trained models and evaluation results will be written
# Check whether the specified path exists or not
isExist = os.path.exists(path_prefix)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path_prefix)
    os.makedirs("{}/Models".format(path_prefix))
    os.makedirs("{}/Evaluation Results".format(path_prefix))
batch_size = 50
topology_batch_size = 1000
num_classes = 8

# Specify source and target technologies
source_tech_name = "ATAC"
target_tech_name = "RNA"
latent_dimensions = 8 # The dimensions for the latent embeddings

num_iterations = 5 # Number of different scTopoGANs to be initialised and trained (recommended at least 10)
epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#epochs = [100]

learning_rate = 0.0001
checkpoint_epoch = 100

total_epochs = 1001
#total_epochs = 101
# </editor-fold>

# <editor-fold desc="Load data from all modalities">
print("=======================================================================")
print("Loading data . . .")
print("=======================================================================")

source_tech = Manifold_Data = pd.read_csv(
    "{}/PBMC {} TopoAE 8 Dimensions.csv".format(data_dir, source_tech_name)
    , header=0, delimiter=',', index_col = 0)

target_tech = Manifold_Data = pd.read_csv(
    "{}/PBMC {} TopoAE 8 Dimensions.csv".format(data_dir, target_tech_name)
    , header=0, delimiter=',', index_col = 0)

features_threshold_source = len(source_tech.columns)
features_threshold_target = len(target_tech.columns)
Annotation_Data = pd.read_csv("{}/Annotations.csv".format(data_dir), header=0, delimiter=',', index_col = 0)
Annotation_Data_Source = pd.read_csv("{}/Annotations_{}.csv".format(data_dir, source_tech_name)
, header=0, delimiter=',', index_col = 0)
Annotation_Data_Target = pd.read_csv("{}/Annotations_{}.csv".format(data_dir, target_tech_name)
, header=0, delimiter=',', index_col = 0)
Annotation_Data = Annotation_Data["cell_type"] # Pick only "Cell_Type" column

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
# </editor-fold>

# <editor-fold desc="Begin training">
print("=======================================================================")
print("Training ")
print("=======================================================================")
generator = GeneratorNet(input_dim=latent_dimensions, output_dim=latent_dimensions)
discriminator = DiscriminatorNet(input_dim=latent_dimensions)
techs = [source_tech_name, target_tech_name]

aggregate_topo_error = []

for random_seed in range(num_iterations):
    # Set random seed for model uniqueness and reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    tf.random.set_seed(random_seed)
    path_suffix = "{}_MODEL_{}".format(core_suffix, str(random_seed))
    trained_generator = train(generator, discriminator, data_loader,
                              total_epochs, checkpoint_epoch,
                              learning_rate, techs, path_prefix, path_suffix)

    Evaluation_Columns = ["Model", "Source", "Target", "Iteration", "Epoch",
                          "Source to Projected Topology Loss"]

    evaluation_results = []
    topo_errors = []

    for epoch in epochs:

        print("#########################################################")
        print("#########################################################")
        print("")
        print("Evaluating for epoch: ", epoch)
        path = "{}/Models/{}_to_{}_Generator_{}_{}.pt".format(path_prefix,
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

        evaluation_results.append(["GAN", source_tech_name, target_tech_name, path_suffix, epoch,
                                   total_topo_loss_source_projected])
        if epoch > 500:
            topo_errors.append(total_topo_loss_source_projected)
    mean_topo_error = sum(topo_errors)/len(topo_errors)
    aggregate_topo_error.append([random_seed, mean_topo_error])

    # Save evaluation results
    evaluation_results = pd.DataFrame(evaluation_results, columns=Evaluation_Columns)
    evaluation_results.to_csv(
        path_or_buf="{}/Evaluation Results/Topological Assessment {}.csv".format(path_prefix, path_suffix),
        header=True, index=False)
# </editor-fold>

aggregate_topo_error = pd.DataFrame(data=aggregate_topo_error, columns=["Seed", "Mean Error"]).sort_values(by="Mean Error", ascending=True)
print("Recommended model numbers for second generation: ")
print(aggregate_topo_error.head(2))
aggregate_topo_error.to_csv(
    path_or_buf="{}/Evaluation Results/Aggregate Topological Assessment First Generation.csv".format(path_prefix),
    header=True, index=False)
