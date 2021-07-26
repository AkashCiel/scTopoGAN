"""
    This script loads data in its post-processing dimensionality and extracts a lower dimensional embedding for it
    This is referred to as the 'Manifold Projection' step in the report
    The methodology used in this script is Topological Autoencoder
"""

# Import modules
from copy import deepcopy
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from src.models.approx_based import *
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

############################
# User Input Starts
############################

batch_size = 50
topology_regulariser_coefficient = 1.0

modality = "" # Enter the modality name here ('RNA', 'ATAC', or 'ADT')
raw_data = "" # Enter name of raw data csv file (without the .csv extension) here
annotations = "" # Enter name of the annotations csv file (without the .csv extension) here
initial_LR = 0.001
num_latent_dims = 8

data_load_dir = "Unsupervised_Manifold_Alignment/Data/PBMC Data"
data_write_dir = "Unsupervised_Manifold_Alignment/Topological_Autoencoder/Output"

############################
# User Input Ends
############################

print("=======================================================================")
print("Loading data . . .")
print("=======================================================================")

Manifold_Data = pd.read_csv("{}/{}.csv".format(data_load_dir, raw_data),
                            header=0, delimiter=',', index_col = 0)
all_samples = list(Manifold_Data.index.values)
features_threshold = len(Manifold_Data.columns)
Annotation_Data = pd.read_csv("{}/{}.csv".format(data_load_dir, annotations), header=0, delimiter=',', index_col = 0)
VAE_Data = Manifold_Data.merge(Annotation_Data, how='inner', left_index=True, right_index=True)
VAE_Data_numpy = VAE_Data.to_numpy()
VAE_Data_Features, VAE_Data_Labels = np.array_split(VAE_Data_numpy, [features_threshold], axis=1)
VAE_Data_Features = VAE_Data_Features.astype(float)
VAE_Data_Features_tensor = torch.tensor(VAE_Data_Features).float() # Convert to tensor

vae_train, vae_val, labels_train, labels_val = train_test_split(VAE_Data_Features_tensor, VAE_Data_Labels,
                                                                test_size=0.20, random_state = 42)

dataloaders = {'train': DataLoader(vae_train, batch_size=batch_size, shuffle=True, num_workers=0),
                'val': DataLoader(vae_val, batch_size=batch_size, shuffle=True, num_workers=0)
                }

print("Train data: ", vae_train.shape)
print("Val data: ", vae_val.shape)

print("Training for topology weight of ", topology_regulariser_coefficient)
my_model = TopologicallyRegularizedAutoencoder(lam=topology_regulariser_coefficient,
                                               autoencoder_model="MLPAutoencoder_{}".format(modality),
                                               toposig_kwargs={"match_edges": "symmetric"})

# Define training loop
def train(n_epochs, batch_size, learning_rate, weight_decay, val_size,
          early_stop_threshold, _rnd, grad_threshold):

    model = my_model

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate,
        weight_decay=weight_decay)

    time_start = datetime.now().time()
    train_losses = []
    val_losses = []
    no_imp_count = 0
    best_val_loss = 1000000  # Set high initial value of best validation loss
    model = model.float()
    for epoch in range(n_epochs):
        train_loss = 0
        model.train()
        for train_batch_id, data in enumerate(dataloaders['train']):
            # Set model into training mode and compute loss
            #print(data.dtype)
            """
            for parameter in model.parameters():
                print(parameter.dtype)
            """
            loss, loss_components = model(data.float())

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_threshold)  # Normalise gradients to prevent gradient explosion
            optimizer.step()

        train_loss = train_loss / train_batch_id
        train_losses.append(train_loss)
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss))

        val_loss = 0
        model.eval()
        for val_batch_id, data in enumerate(dataloaders['val']):
            loss, loss_components = model(data.float())
            val_loss += loss.item()
        val_loss = val_loss / val_batch_id
        val_losses.append(val_loss)

        if (val_loss < best_val_loss):
            print("Improved model with validation loss: {}".format(val_loss))
            best_model_wts = deepcopy(model.state_dict())
            best_val_loss = val_loss
            no_imp_count = 0

        if (val_loss >= best_val_loss):
            no_imp_count += 1
            if epoch > early_stop_threshold:
                if (no_imp_count > 10):
                    print("No improvements for 10 epochs stopping early")
                    model.load_state_dict(best_model_wts)
                    x = range(epoch + 1)
                    return model
    return model

print("=======================================================================")
print("Training Topological Autoencoder model now . . .")
print("=======================================================================")

trained_model = train(n_epochs=101,
                      batch_size=50,
                      learning_rate=initial_LR,
                      weight_decay=1e-05,
                      val_size=0.15,
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

print("Shape of latent embeddings: ", latent_data.shape)

latent_data = latent_data.numpy()
latent_data = pd.DataFrame(latent_data, index=all_samples)
latent_data.to_csv(
    path_or_buf="{}/{} TopoAE {} Dimensions Partial.csv".format(data_write_dir, modality, num_latent_dims),
    header=True, index=True)
