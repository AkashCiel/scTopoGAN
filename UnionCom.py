import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from unioncom import UnionCom
import scanpy as sc
from Qualitative_Metrics import evaluate_neighbors

random_seed = 1     # define random seed
print("Random seed : {}".format(random_seed))

#%% Full PBMC

# Define parameters
source_tech_name = "ATAC"
epochs = 1000

source_tech = pd.read_csv('Data/PBMC Multiome/ATAC_LSI.csv',header=0,index_col=0)
target_tech = pd.read_csv('Data/PBMC Multiome/RNA_PCA.csv',header=0,index_col=0)
meta_source = pd.read_csv('Data/PBMC Multiome/annotations.csv',header=0,index_col=0)
meta_target = pd.read_csv('Data/PBMC Multiome/annotations.csv',header=0,index_col=0)

source_indices = list(source_tech.index)
target_indices = list(target_tech.index)

source_tech = source_tech.to_numpy()
source_tech = source_tech.astype(float)
print("Source Technology: ", np.shape(source_tech))
target_tech = target_tech.to_numpy()
target_tech = target_tech.astype(float)
print("Target Technology: ", np.shape(target_tech))

shuffle_idx = np.random.choice(source_tech.shape[0],size=source_tech.shape[0],replace=False)
meta_source_shuffled = meta_source.iloc[shuffle_idx,:]

uc = UnionCom.UnionCom(epoch_pd=epochs, epoch_DNN=200, manual_seed=random_seed, output_dim=8)
start = time.time()
integrated_data = uc.fit_transform(dataset=[source_tech[shuffle_idx,:],target_tech])
end = time.time()
print("UnionCOM run time = ", end - start)

source_indices = np.array(source_indices)[shuffle_idx]
source_projected = pd.DataFrame(integrated_data[0], index=source_indices).astype("float")
target_projected = pd.DataFrame(integrated_data[1], index=target_indices).astype("float")

source_projected.to_csv('ATAC_aligned_UnionCom.csv')
target_projected.to_csv('RNA_aligned_UnionCom.csv')

CTM = evaluate_neighbors(source_projected, target_projected, meta_source_shuffled['Celltype'], meta_target['Celltype'], k=5)
SCTM = evaluate_neighbors(source_projected, target_projected, meta_source_shuffled['Subcelltype'], meta_target['Subcelltype'], k=5)
print('Celltype matching =', CTM)
print('Subcelltype matching =', SCTM)

batch = np.concatenate((np.repeat('ATAC',source_projected.shape[0]),np.repeat('RNA',target_projected.shape[0])))
celltype = np.concatenate((meta_source_shuffled['Celltype'],meta_target['Celltype']))
subcelltype = np.concatenate((meta_source_shuffled['Subcelltype'],meta_target['Subcelltype']))

Aligned_metadata = pd.DataFrame(np.array([batch,celltype,subcelltype]).T,columns=['Batch','Celltype','Subcelltype'],
                                index=np.concatenate((source_projected.index,target_projected.index)))
Aligned_data = pd.DataFrame(np.concatenate((source_projected,target_projected),axis=0),index=np.concatenate((source_projected.index,target_projected.index)))

adata_RNA_ATAC_aligned = sc.AnnData(X = Aligned_data, obs = Aligned_metadata)
sc.pp.neighbors(adata_RNA_ATAC_aligned, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_ATAC_aligned)
sc.pl.umap(adata_RNA_ATAC_aligned, color='Batch', title = 'Dataset')
plt.savefig("PBMC_UnionCom_batch.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_ATAC_aligned, color='Celltype', title = 'Celltype')
plt.savefig("PBMC_UnionCom_celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_ATAC_aligned, color='Subcelltype', title = 'Subcelltype')
plt.savefig("PBMC_UnionCom_subcelltype.tiff", bbox_inches='tight', dpi=150)

#%% Partial PBMC

# Define parameters
source_tech_name = "ATAC"
epochs = 1000

source_tech = pd.read_csv('Data/PBMC Multiome/Partial_ATAC.csv',header=0,index_col=0)
target_tech = pd.read_csv('Data/PBMC Multiome/Partial_RNA.csv',header=0,index_col=0)
meta_source = pd.read_csv('Data/PBMC Multiome/annotations_Partial_ATAC.csv',header=0,index_col=0)
meta_target = pd.read_csv('Data/PBMC Multiome/annotations_Partial_RNA.csv',header=0,index_col=0)

source_indices = list(source_tech.index)
target_indices = list(target_tech.index)

source_tech = source_tech.to_numpy()
source_tech = source_tech.astype(float)
print("Source Technology: ", np.shape(source_tech))
target_tech = target_tech.to_numpy()
target_tech = target_tech.astype(float)
print("Target Technology: ", np.shape(target_tech))

shuffle_idx = np.random.choice(source_tech.shape[0],size=source_tech.shape[0],replace=False)
meta_source_shuffled = meta_source.iloc[shuffle_idx,:]

uc = UnionCom.UnionCom(epoch_pd=epochs, epoch_DNN=200, manual_seed=random_seed, output_dim=8)
start = time.time()
integrated_data = uc.fit_transform(dataset=[source_tech[shuffle_idx,:],target_tech])
end = time.time()
print("UnionCOM run time = ", end - start)

source_indices = np.array(source_indices)[shuffle_idx]
source_projected = pd.DataFrame(integrated_data[0], index=source_indices).astype("float")
target_projected = pd.DataFrame(integrated_data[1], index=target_indices).astype("float")

source_projected.to_csv('Partial_ATAC_aligned_UnionCom.csv')
target_projected.to_csv('Partial_RNA_aligned_UnionCom.csv')

CTM = evaluate_neighbors(source_projected, target_projected, meta_source_shuffled['Celltype'], meta_target['Celltype'], k=5)
SCTM = evaluate_neighbors(source_projected, target_projected, meta_source_shuffled['Subcelltype'], meta_target['Subcelltype'], k=5)
print('Celltype matching =', CTM)
print('Subcelltype matching =', SCTM)

batch = np.concatenate((np.repeat('ATAC',source_projected.shape[0]),np.repeat('RNA',target_projected.shape[0])))
celltype = np.concatenate((meta_source_shuffled['Celltype'],meta_target['Celltype']))
subcelltype = np.concatenate((meta_source_shuffled['Subcelltype'],meta_target['Subcelltype']))

Aligned_metadata = pd.DataFrame(np.array([batch,celltype,subcelltype]).T,columns=['Batch','Celltype','Subcelltype'],
                                index=np.concatenate((source_projected.index,target_projected.index)))
Aligned_data = pd.DataFrame(np.concatenate((source_projected,target_projected),axis=0),index=np.concatenate((source_projected.index,target_projected.index)))

adata_RNA_ATAC_aligned = sc.AnnData(X = Aligned_data, obs = Aligned_metadata)
sc.pp.neighbors(adata_RNA_ATAC_aligned, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_ATAC_aligned)
sc.pl.umap(adata_RNA_ATAC_aligned, color='Batch', title = 'Dataset')
plt.savefig("Partial_PBMC_UnionCom_batch.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_ATAC_aligned, color='Celltype', title = 'Celltype')
plt.savefig("Partial_PBMC_UnionCom_celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_ATAC_aligned, color='Subcelltype', title = 'Subcelltype')
plt.savefig("Partial_PBMC_UnionCom_subcelltype.tiff", bbox_inches='tight', dpi=150)

#%% BM

# Define parameters
source_tech_name = "ADT"
epochs = 1000

source_tech = pd.read_csv('Data/BM Citeseq/Partial_ADT.csv',header=0,index_col=0)
target_tech = pd.read_csv('Data/BM Citeseq/Partial_RNA.csv',header=0,index_col=0)
meta_source = pd.read_csv('Data/BM Citeseq/annotations_Partial_ADT.csv',header=0,index_col=0)
meta_target = pd.read_csv('Data/BM Citeseq/annotations_Partial_RNA.csv',header=0,index_col=0)

source_indices = list(source_tech.index)
target_indices = list(target_tech.index)

source_tech = source_tech.to_numpy()
source_tech = source_tech.astype(float)
print("Source Technology: ", np.shape(source_tech))
target_tech = target_tech.to_numpy()
target_tech = target_tech.astype(float)
print("Target Technology: ", np.shape(target_tech))

shuffle_idx = np.random.choice(source_tech.shape[0],size=source_tech.shape[0],replace=False)
meta_source_shuffled = meta_source.iloc[shuffle_idx,:]

uc = UnionCom.UnionCom(epoch_pd=epochs, epoch_DNN=200, manual_seed=random_seed, output_dim=8)
start = time.time()
integrated_data = uc.fit_transform(dataset=[source_tech[shuffle_idx,:],target_tech])
end = time.time()
print("UnionCOM run time = ", end - start)

source_indices = np.array(source_indices)[shuffle_idx]
source_projected = pd.DataFrame(integrated_data[0], index=source_indices).astype("float")
target_projected = pd.DataFrame(integrated_data[1], index=target_indices).astype("float")

source_projected.to_csv('ADT_aligned_UnionCom.csv')
target_projected.to_csv('RNA_aligned_UnionCom.csv')

CTM = evaluate_neighbors(source_projected, target_projected, meta_source_shuffled['Celltype'], meta_target['Celltype'], k=5)
SCTM = evaluate_neighbors(source_projected, target_projected, meta_source_shuffled['Subcelltype'], meta_target['Subcelltype'], k=5)
print('Celltype matching =', CTM)
print('Subcelltype matching =', SCTM)

batch = np.concatenate((np.repeat('ADT',source_projected.shape[0]),np.repeat('RNA',target_projected.shape[0])))
celltype = np.concatenate((meta_source_shuffled['Celltype'],meta_target['Celltype']))
subcelltype = np.concatenate((meta_source_shuffled['Subcelltype'],meta_target['Subcelltype']))

Aligned_metadata = pd.DataFrame(np.array([batch,celltype,subcelltype]).T,columns=['Batch','Celltype','Subcelltype'],
                                index=np.concatenate((source_projected.index,target_projected.index)))
Aligned_data = pd.DataFrame(np.concatenate((source_projected,target_projected),axis=0),index=np.concatenate((source_projected.index,target_projected.index)))

adata_RNA_ATAC_aligned = sc.AnnData(X = Aligned_data, obs = Aligned_metadata)
sc.pp.neighbors(adata_RNA_ATAC_aligned, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_ATAC_aligned)
sc.pl.umap(adata_RNA_ATAC_aligned, color='Batch', title = 'Dataset')
plt.savefig("BM_UnionCom_batch.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_ATAC_aligned, color='Celltype', title = 'Celltype')
plt.savefig("BM_UnionCom_celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_ATAC_aligned, color='Subcelltype', title = 'Subcelltype')
plt.savefig("BM_UnionCom_subcelltype.tiff", bbox_inches='tight', dpi=150)