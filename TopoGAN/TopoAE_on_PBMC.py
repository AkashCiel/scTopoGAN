"""
    This script projects processed PBMC data into an 8 dimensional space from 50
    1. Loads the 50 dimensional PBMC data, according to the modality specified by the user (ATAC or RNA)
    2. Projects the 50 dimensional data in 2-d using tSNE
    3. Trains TopoAE for 100 epochs or until performance on validation set stops improving (whichever is earlier)
    4. Obtains latent embeddings
    5. Computes evaluation metrics (silhouette scores of original and latent dimensionalities, KL-Global Density score)
    6. Projects the latent embeddings into 2-d using tSNE
    7. Writes the trained latent embeddings and evaluation results into csv files at a user specified location
"""

# Import modules
from copy import deepcopy
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from src.models.approx_based import *
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import seaborn as sns
from Qualitative_Metrics import *
from src.models.submodules import VAE_PBMC
from TopoGAN_Functions import *
# Directory where the pre-processed data is located (below)
in_directory = "PBMC"

# Directory where the latent embeddings and evaluation results for the embeddings will be saved (below)
out_directory = "PBMC"
raw_data = "ATAC" # Specify which modality you wish to obtain embeddings for

# The following variables can be changed but it is recommended to keep them as is for PBMC Data
batch_size = 50
topology_regulariser_coefficient = 0.5
initial_LR = 0.001
num_latent_dims = 8
print("=======================================================================")
print("Loading data . . .")
print("=======================================================================")

VAE_Data_Features_tensor, VAE_Data_Labels, VAE_Data, VAE_Data_Features, all_samples, dataloaders = prepare_manifold_data(in_directory, raw_data)

# Plot data BEFORE projecting into latent space
print("==================================================================")
print("Plotting data using tSNE . . .")

import time
tsne_data = VAE_Data_Features_tensor
tsne_labels_01 = "cell_type"
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(tsne_data)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

original_score = silhouette_score(tsne_results, VAE_Data_Labels)
print("Silhouette score for tSNE on original dimensionality is: ", original_score)

VAE_Data["tsne_x"] = tsne_results[:,0]
VAE_Data["tsne_y"] = tsne_results[:,1]

num_classes = len(set(list(VAE_Data[tsne_labels_01])))

plt.figure(figsize=(10,10))
latent_tsne_plot = sns.scatterplot(
    x="tsne_x", y="tsne_y",
    hue=tsne_labels_01,
    palette=sns.color_palette("bright", num_classes),
    data=VAE_Data,
    legend="full",
    alpha=0.3
)
latent_tsne_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
plt.show()


Evaluation_Columns = ["Number of Dimensions",
                      "Topology Coefficient",
                      "Silhouette Score Original",
                      "Silhouette Score Latent",
                      "KL Global Density 0.01"]
evaluation_results = []

print("=======================================================================")
print("Computing pairwise distances for original data . . .")
print("=======================================================================")

# Compute pairwise distances for evaluation later
pairwise_X = squareform(pdist(VAE_Data_Features))

print("Training for topology weight of ", topology_regulariser_coefficient)
my_model = TopologicallyRegularizedAutoencoder(lam=topology_regulariser_coefficient,autoencoder_model="MLPAutoencoder_PBMC",toposig_kwargs={"match_edges": "symmetric"})

# Define training loop

print("=======================================================================")
print("Training Topological Autoencoder model now . . .")
print("=======================================================================")

trained_model = train_TopoAE(n_epochs=101,
                      my_model=my_model,
                      learning_rate=initial_LR,
                      weight_decay=1e-05,
                      early_stop_threshold=10,
                      _rnd=42,
                      grad_threshold=50)

print("=======================================================================")
print("model trained, estimating latent embeddings now . . .")
print("=======================================================================")

data_loader = DataLoader(VAE_Data_Features, batch_size=1)
latent_data = []
latent_data = torch.empty(0)
for index, original_sample in enumerate(data_loader):
    latent_sample = trained_model.encode(original_sample.float())
    latent_data = torch.cat((latent_data, latent_sample))
latent_data = latent_data.detach()

print("Evaluating model now . . .")
print("=======================================================================")

pairwise_Z = squareform(pdist(latent_data))
KL_Global_Density_01 = evaluate_model(pairwise_X, pairwise_Z, 0.01)

print("Shape of latent embeddings: ", latent_data.shape)
print("=======================================================================")
print("Plotting latent embeddings for train data now . . .")
plot_labels = "cell_type"

x_axis_label = "latent_x_topo_weight_{}".format(topology_regulariser_coefficient)
y_axis_label = "latent_y_topo_weight_{}".format(topology_regulariser_coefficient)

if num_latent_dims == 2:
    VAE_Data[x_axis_label] = latent_data[:,0]
    VAE_Data[y_axis_label] = latent_data[:,1]

elif num_latent_dims > 2:
    import time

    tsne_data = latent_data

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(tsne_data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    latent_score = silhouette_score(tsne_results, VAE_Data_Labels)
    print("Silhouette score for tSNE on latent dimensionality is: ", latent_score)

    VAE_Data[x_axis_label] = tsne_results[:, 0]
    VAE_Data[y_axis_label] = tsne_results[:, 1]

plt.figure(figsize=(10, 10))
num_classes = len(set(list(VAE_Data[plot_labels])))
latent_plot_test = sns.scatterplot(
    x=x_axis_label, y=y_axis_label,
    hue=plot_labels,
    palette=sns.color_palette("bright", num_classes),
    data=VAE_Data,
    legend="full",
    alpha=0.3
)

latent_plot_test.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
latent_plot_test.xlabel = "topo weight {}".format(topology_regulariser_coefficient)
plt.show()

latent_data = latent_data.numpy()
latent_data = pd.DataFrame(latent_data, index=all_samples)
latent_data.to_csv(path_or_buf="{}/{} TopoAE {} Dimensions.csv".format(out_directory, raw_data, num_latent_dims), header=True, index=True)


evaluation_results.append([num_latent_dims, topology_regulariser_coefficient, original_score, latent_score, KL_Global_Density_01])
evaluation_results = pd.DataFrame(evaluation_results, columns=Evaluation_Columns)
evaluation_results.to_csv(path_or_buf="{}/{} Quantitative {} Dimensions.csv".format(out_directory, raw_data, num_latent_dims),
                          header=True, index=False)