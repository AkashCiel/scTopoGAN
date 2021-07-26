"""
    This script will:
    1. Import required modules
    2. Define required parameters
    3. Load data
	4. Train and store model
	5. Evaluate topology loss of stored model
"""

from Manifold_Alignment_GAN import *
from src.models.Assess_Topology_V02 import *

# Define parameters
batch_size = 50
topology_batch_size = 1000
num_classes = 8
source_tech_name = "ADT"
latent_dimensions = 8
topology_weight = 2.0 # There are multiple latent projections of CiteSeq data, defined by topology weight

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
path_suffix = "{}_MODEL_{}".format(core_suffix, str(random_seed))
data_dir = "/home/nfs/asingh5/Unsupervised_Manifold_Alignment/Data/CiteSeq Data"
path_prefix = "/home/nfs/asingh5/Unsupervised_Manifold_Alignment/GAN_Ensemble_Topology_Experiments/Partial_CiteSeq"

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

source_target_numpy = np.concatenate((source_tech, target_tech), axis=1)
source_target_tensor = torch.tensor(source_target_numpy).float() # Convert to tensor

# Define data loader
data_loader = DataLoader(source_target_tensor, batch_size=batch_size, shuffle=False, num_workers=0)

# Train model

print("=======================================================================")
print("Training ")
print("=======================================================================")
generator = GeneratorNet(input_dim=latent_dimensions, output_dim=latent_dimensions)
discriminator = DiscriminatorNet(input_dim=latent_dimensions)
techs = [source_tech_name, target_tech_name]
trained_generator = train(generator, discriminator, data_loader,
                          total_epochs, checkpoint_epoch,
                          learning_rate, techs, path_prefix, path_suffix)

Evaluation_Columns = ["Model", "Source", "Target", "Iteration", "Epoch",
                      "Source to Projected Topology Loss"]

evaluation_results = []

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

# Save evaluation results
evaluation_results = pd.DataFrame(evaluation_results, columns=Evaluation_Columns)
evaluation_results.to_csv(
    path_or_buf="{}/Evaluation Results/Topological Assessment {}.csv".format(path_prefix, path_suffix),
    header=True, index=False)
