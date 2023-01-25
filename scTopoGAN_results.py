import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
import time

from scTopoGAN_Functions import get_TopoAE_Embeddings, run_scTopoGAN
from Qualitative_Metrics import evaluate_neighbors

#%% Full PBMC

PBMC_RNA = pd.read_csv('Data/PBMC Multiome/RNA_PCA.csv',header=0,index_col=0)
PBMC_ATAC = pd.read_csv('Data/PBMC Multiome/ATAC_LSI.csv',header=0,index_col=0)
meta_data = pd.read_csv('Data/PBMC Multiome/annotations.csv',header=0,index_col=0)

target_latent = get_TopoAE_Embeddings(Manifold_Data = PBMC_RNA, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=3, initial_LR=0.001)

source_latent = get_TopoAE_Embeddings(Manifold_Data = PBMC_ATAC, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=0.5, initial_LR=0.001)

## scTopoGAN
start = time.time()
source_aligned = run_scTopoGAN(source_latent, target_latent, source_tech_name="ATAC", target_tech_name="RNA", 
                               batch_size=512, topology_batch_size=1000, total_epochs=1001, num_iterations=20, 
                               checkpoint_epoch=100, learning_rate=1e-4, path_prefix="Results_Full_PBMC")
end = time.time()
print("TopoGAN run time = ", end - start)

source_aligned.to_csv('ATAC_aligned_scTopoGAN.csv')

CTM = evaluate_neighbors(source_aligned, target_latent, meta_data['Celltype'], meta_data['Celltype'], k=5)
SCTM = evaluate_neighbors(source_aligned, target_latent, meta_data['Subcelltype'], meta_data['Subcelltype'], k=5)
print('Celltype matching =', CTM)
print('Subcelltype matching =', SCTM)

batch = np.concatenate((np.repeat('ATAC',source_aligned.shape[0]),np.repeat('RNA',target_latent.shape[0])))
celltype = np.concatenate((meta_data['Celltype'],meta_data['Celltype']))
subcelltype = np.concatenate((meta_data['Subcelltype'],meta_data['Subcelltype']))

Aligned_metadata = pd.DataFrame(np.array([batch,celltype,subcelltype]).T,columns=['Batch','Celltype','Subcelltype'],
                                index=np.concatenate((source_aligned.index,target_latent.index)))
Aligned_data = pd.DataFrame(np.concatenate((source_aligned,target_latent),axis=0),index=np.concatenate((source_aligned.index,target_latent.index)))

adata_RNA_ATAC_aligned = sc.AnnData(X = Aligned_data, obs = Aligned_metadata)
sc.pp.neighbors(adata_RNA_ATAC_aligned, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_ATAC_aligned)
sc.pl.umap(adata_RNA_ATAC_aligned, color='Batch', title = 'Dataset')
plt.savefig("PBMC_scTopoGAN_batch.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_ATAC_aligned, color='Celltype', title = 'Celltype')
plt.savefig("PBMC_scTopoGAN_celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_ATAC_aligned, color='Subcelltype', title = 'Subcelltype')
plt.savefig("PBMC_scTopoGAN_subcelltype.tiff", bbox_inches='tight', dpi=150)

#%% Partial PBMC

PBMC_RNA = pd.read_csv('Data/PBMC Multiome/Partial_RNA.csv',header=0,index_col=0)
PBMC_ATAC = pd.read_csv('Data/PBMC Multiome/Partial_ATAC.csv',header=0,index_col=0)
Annotation_RNA = pd.read_csv('Data/PBMC Multiome/annotations_Partial_RNA.csv',header=0,index_col=0)
Annotation_ATAC = pd.read_csv('Data/PBMC Multiome/annotations_Partial_ATAC.csv',header=0,index_col=0)

target_latent = get_TopoAE_Embeddings(Manifold_Data = PBMC_RNA, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=3, initial_LR=0.001)

source_latent = get_TopoAE_Embeddings(Manifold_Data = PBMC_ATAC, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=0.5, initial_LR=0.001)

## scTopoGAN
start = time.time()
source_aligned = run_scTopoGAN(source_latent, target_latent, source_tech_name="ATAC", target_tech_name="RNA", 
                               batch_size=50, topology_batch_size=1000, total_epochs=1001, num_iterations=20, 
                               checkpoint_epoch=100, learning_rate=1e-4, path_prefix="Results_Partial_PBMC")
end = time.time()
print("TopoGAN run time = ", end - start)

source_aligned.to_csv('Partial_ATAC_aligned_scTopoGAN.csv')

CTM = evaluate_neighbors(source_aligned, target_latent, Annotation_ATAC['Celltype'], Annotation_RNA['Celltype'], k=5)
SCTM = evaluate_neighbors(source_aligned, target_latent, Annotation_ATAC['Subcelltype'], Annotation_RNA['Subcelltype'], k=5)
print('Celltype matching =', CTM)
print('Subcelltype matching =', SCTM)

batch = np.concatenate((np.repeat('ATAC',source_aligned.shape[0]),np.repeat('RNA',target_latent.shape[0])))
celltype = np.concatenate((Annotation_ATAC['Celltype'],Annotation_RNA['Celltype']))
subcelltype = np.concatenate((Annotation_ATAC['Subcelltype'],Annotation_RNA['Subcelltype']))

Aligned_metadata = pd.DataFrame(np.array([batch,celltype,subcelltype]).T,columns=['Batch','Celltype','Subcelltype'],
                                index=np.concatenate((source_aligned.index,target_latent.index)))
Aligned_data = pd.DataFrame(np.concatenate((source_aligned,target_latent),axis=0),index=np.concatenate((source_aligned.index,target_latent.index)))

adata_RNA_ATAC_aligned = sc.AnnData(X = Aligned_data, obs = Aligned_metadata)
sc.pp.neighbors(adata_RNA_ATAC_aligned, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_ATAC_aligned)
sc.pl.umap(adata_RNA_ATAC_aligned, color='Batch', title = 'Dataset')
plt.savefig("Partial_PBMC_scTopoGAN_batch.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_ATAC_aligned, color='Celltype', title = 'Celltype')
plt.savefig("Partial_PBMC_scTopoGAN_celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_ATAC_aligned, color='Subcelltype', title = 'Subcelltype')
plt.savefig("Partial_PBMC_scTopoGAN_subcelltype.tiff", bbox_inches='tight', dpi=150)

#%% BM

BM_RNA = pd.read_csv('Data/BM Citeseq/Partial_RNA.csv',header=0,index_col=0)
BM_ADT = pd.read_csv('Data/BM Citeseq/Partial_ADT.csv',header=0,index_col=0)
Annotation_RNA = pd.read_csv('Data/BM Citeseq/annotations_Partial_RNA.csv',header=0,index_col=0)
Annotation_ADT = pd.read_csv('Data/BM Citeseq/annotations_Partial_ADT.csv',header=0,index_col=0)

target_latent = get_TopoAE_Embeddings(Manifold_Data = BM_RNA, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=2, initial_LR=0.001)

source_latent = get_TopoAE_Embeddings(Manifold_Data = BM_ADT, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [25, 16, 16, 8], topology_regulariser_coefficient=0.5, initial_LR=0.001)

## scTopoGAN
start = time.time()
source_aligned = run_scTopoGAN(source_latent, target_latent, source_tech_name="ADT", target_tech_name="RNA", 
                               batch_size=512, topology_batch_size=1000, total_epochs=1001, num_iterations=20, 
                               checkpoint_epoch=100, learning_rate=1e-4, path_prefix="Results_BM")
end = time.time()
print("TopoGAN run time = ", end - start)

source_aligned.to_csv('ADT_aligned_scTopoGAN.csv')

CTM = evaluate_neighbors(source_aligned, target_latent, Annotation_ADT['Celltype'], Annotation_RNA['Celltype'], k=5)
SCTM = evaluate_neighbors(source_aligned, target_latent, Annotation_ADT['Subcelltype'], Annotation_RNA['Subcelltype'], k=5)
print('Celltype matching =', CTM)
print('Subcelltype matching =', SCTM)

batch = np.concatenate((np.repeat('ADT',source_aligned.shape[0]),np.repeat('RNA',target_latent.shape[0])))
celltype = np.concatenate((Annotation_ADT['Celltype'],Annotation_RNA['Celltype']))
subcelltype = np.concatenate((Annotation_ADT['Subcelltype'],Annotation_RNA['Subcelltype']))

Aligned_metadata = pd.DataFrame(np.array([batch,celltype,subcelltype]).T,columns=['Batch','Celltype','Subcelltype'],
                                index=np.concatenate((source_aligned.index,target_latent.index)))
Aligned_data = pd.DataFrame(np.concatenate((source_aligned,target_latent),axis=0),index=np.concatenate((source_aligned.index,target_latent.index)))

adata_RNA_ADT_aligned = sc.AnnData(X = Aligned_data, obs = Aligned_metadata)
sc.pp.neighbors(adata_RNA_ADT_aligned, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_ADT_aligned)
sc.pl.umap(adata_RNA_ADT_aligned, color='Batch', title = 'Dataset')
plt.savefig("BM_scTopoGAN_batch.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_ADT_aligned, color='Celltype', title = 'Celltype')
plt.savefig("BM_scTopoGAN_celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_ADT_aligned, color='Subcelltype', title = 'Subcelltype')
plt.savefig("BM_scTopoGAN_subcelltype.tiff", bbox_inches='tight', dpi=150)