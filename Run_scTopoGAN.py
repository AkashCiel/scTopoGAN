import numpy as np
import pandas as pd

from scTopoGAN_Functions import get_TopoAE_Embeddings, run_scTopoGAN

# load data
PBMC_RNA = pd.read_csv('Data/PBMC Multiome/RNA_PCA.csv',header=0,index_col=0)
PBMC_ATAC = pd.read_csv('Data/PBMC Multiome/ATAC_LSI.csv',header=0,index_col=0)

# Step 1: Get TopoAE embeddings
# set topology_regulariser_coefficient between 0.5 to 3.0

target_latent = get_TopoAE_Embeddings(Manifold_Data = PBMC_RNA, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=3, initial_LR=0.001)

source_latent = get_TopoAE_Embeddings(Manifold_Data = PBMC_ATAC, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=0.5, initial_LR=0.001)

## Step 2: Manifold alignment using scTopoGAN
source_aligned = run_scTopoGAN(source_latent, target_latent, source_tech_name="ATAC", target_tech_name="RNA", 
                               batch_size=512, topology_batch_size=1000, total_epochs=1001, num_iterations=20, 
                               checkpoint_epoch=100, learning_rate=1e-4, path_prefix="Results")

source_aligned.to_csv('ATAC_aligned.csv')