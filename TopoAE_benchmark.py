import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score

from scTopoGAN_Functions import get_TopoAE_Embeddings, get_AE_Embeddings
from Qualitative_Metrics import density_kl_div

#%% PBMC Multiome RNA
PBMC_RNA = pd.read_csv('Data/PBMC Multiome/RNA_PCA.csv',header=0,index_col=0)
meta_data = pd.read_csv('Data/PBMC Multiome/annotations.csv',header=0,index_col=0)

Results = pd.DataFrame(columns=['Silhouette_Celltype','Silhouette_Subcelltype','KL_0.01'])

# UMAP 8
adata_RNA = sc.AnnData(X = PBMC_RNA, obs = meta_data)
sc.pp.neighbors(adata_RNA, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA,n_components=8)

Results.loc['UMAP_8','Silhouette_Celltype'] = silhouette_score(adata_RNA.obsm['X_umap'], meta_data['Celltype'])
Results.loc['UMAP_8','Silhouette_Subcelltype'] = silhouette_score(adata_RNA.obsm['X_umap'], meta_data['Subcelltype'])

Results.loc['UMAP_8','KL_0.01'] = density_kl_div(np.array(PBMC_RNA),np.array(adata_RNA.obsm['X_umap']),sigma=0.01)

# UMAP 2
adata_RNA = sc.AnnData(X = PBMC_RNA, obs = meta_data)
sc.pp.neighbors(adata_RNA, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA)
sc.pl.umap(adata_RNA, color='Celltype')
plt.savefig("UMAP_PBMC_RNA_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA, color='Subcelltype',legend_loc='on data')
plt.savefig("UMAP_PBMC_RNA_Subcelltype.tiff", bbox_inches='tight', dpi=150)

Results.loc['UMAP_2','Silhouette_Celltype'] = silhouette_score(adata_RNA.obsm['X_umap'], meta_data['Celltype'])
Results.loc['UMAP_2','Silhouette_Subcelltype'] = silhouette_score(adata_RNA.obsm['X_umap'], meta_data['Subcelltype'])

Results.loc['UMAP_2','KL_0.01'] = density_kl_div(np.array(PBMC_RNA),np.array(adata_RNA.obsm['X_umap']),sigma=0.01)

# TopoAE 0.5 (run 10 times)
target_latent = get_TopoAE_Embeddings(Manifold_Data = PBMC_RNA, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=0.5, initial_LR=0.001)

Results.loc['TopoAE_8_lam_0.5','Silhouette_Celltype'] = silhouette_score(target_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_0.5','Silhouette_Subcelltype'] = silhouette_score(target_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_0.5','KL_0.01'] = density_kl_div(np.array(PBMC_RNA),np.array(target_latent),sigma=0.01)

adata_RNA_Topo = sc.AnnData(X = pd.DataFrame(target_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_RNA_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_Topo)
sc.pl.umap(adata_RNA_Topo, color='Celltype')
plt.savefig("TopoAE_lam0.5_PBMC_RNA_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam0.5_PBMC_RNA_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# TopoAE 1.0 (run 10 times)
target_latent = get_TopoAE_Embeddings(Manifold_Data = PBMC_RNA, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=1, initial_LR=0.001)

Results.loc['TopoAE_8_lam_1.0','Silhouette_Celltype'] = silhouette_score(target_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_1.0','Silhouette_Subcelltype'] = silhouette_score(target_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_1.0','KL_0.01'] = density_kl_div(np.array(PBMC_RNA),np.array(target_latent),sigma=0.01)

adata_RNA_Topo = sc.AnnData(X = pd.DataFrame(target_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_RNA_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_Topo)
sc.pl.umap(adata_RNA_Topo, color='Celltype')
plt.savefig("TopoAE_lam1.0_PBMC_RNA_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam1.0_PBMC_RNA_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# TopoAE 2.0 (run 10 times)
target_latent = get_TopoAE_Embeddings(Manifold_Data = PBMC_RNA, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=2, initial_LR=0.001)

Results.loc['TopoAE_8_lam_2.0','Silhouette_Celltype'] = silhouette_score(target_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_2.0','Silhouette_Subcelltype'] = silhouette_score(target_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_2.0','KL_0.01'] = density_kl_div(np.array(PBMC_RNA),np.array(target_latent),sigma=0.01)

adata_RNA_Topo = sc.AnnData(X = pd.DataFrame(target_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_RNA_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_Topo)
sc.pl.umap(adata_RNA_Topo, color='Celltype')
plt.savefig("TopoAE_lam2.0_PBMC_RNA_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam2.0_PBMC_RNA_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# TopoAE 3.0 (run 10 times)
target_latent = get_TopoAE_Embeddings(Manifold_Data = PBMC_RNA, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=3, initial_LR=0.001)

Results.loc['TopoAE_8_lam_3.0','Silhouette_Celltype'] = silhouette_score(target_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_3.0','Silhouette_Subcelltype'] = silhouette_score(target_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_3.0','KL_0.01'] = density_kl_div(np.array(PBMC_RNA),np.array(target_latent),sigma=0.01)

adata_RNA_Topo = sc.AnnData(X = pd.DataFrame(target_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_RNA_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_Topo)
sc.pl.umap(adata_RNA_Topo, color='Celltype')
plt.savefig("TopoAE_lam3.0_PBMC_RNA_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam3.0_PBMC_RNA_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# VAE 8 (run 10 times)
target_latent = get_AE_Embeddings(Manifold_Data = PBMC_RNA, batch_size=50, model_type= "Variational", AE_arch = [50, 32, 32, 8], initial_LR=0.001)

Results.loc['VAE_8','Silhouette_Celltype'] = silhouette_score(target_latent, meta_data['Celltype'])
Results.loc['VAE_8','Silhouette_Subcelltype'] = silhouette_score(target_latent, meta_data['Subcelltype'])

Results.loc['VAE_8','KL_0.01'] = density_kl_div(np.array(PBMC_RNA),np.array(target_latent),sigma=0.01)

adata_RNA_VAE = sc.AnnData(X = pd.DataFrame(target_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_RNA_VAE, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_VAE)
sc.pl.umap(adata_RNA_VAE, color='Celltype')
plt.savefig("VAE_PBMC_RNA_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_VAE, color='Subcelltype',legend_loc='on data')
plt.savefig("VAE_PBMC_RNA_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# AE 8 (run 10 times)
target_latent = get_AE_Embeddings(Manifold_Data = PBMC_RNA, batch_size=50, model_type= "regular", AE_arch = [50, 32, 32, 8], initial_LR=0.001)

Results.loc['AE_8','Silhouette_Celltype'] = silhouette_score(target_latent, meta_data['Celltype'])
Results.loc['AE_8','Silhouette_Subcelltype'] = silhouette_score(target_latent, meta_data['Subcelltype'])

Results.loc['AE_8','KL_0.01'] = density_kl_div(np.array(PBMC_RNA),np.array(target_latent),sigma=0.01)

adata_RNA_AE = sc.AnnData(X = pd.DataFrame(target_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_RNA_AE, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_AE)
sc.pl.umap(adata_RNA_AE, color='Celltype')
plt.savefig("AE_PBMC_RNA_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_AE, color='Subcelltype',legend_loc='on data')
plt.savefig("AE_PBMC_RNA_Subcelltype.tiff", bbox_inches='tight', dpi=150)

Results_PBMC_RNA = Results.copy()
#%% PBMC Multiome ATAC
PBMC_ATAC = pd.read_csv('Data/PBMC Multiome/ATAC_LSI.csv',header=0,index_col=0)
meta_data = pd.read_csv('Data/PBMC Multiome/annotations.csv',header=0,index_col=0)

Results = pd.DataFrame(columns=['Silhouette_Celltype','Silhouette_Subcelltype','KL_0.01'])

# UMAP 8
adata_ATAC = sc.AnnData(X = PBMC_ATAC, obs = meta_data)
sc.pp.neighbors(adata_ATAC, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ATAC,n_components=8)

Results.loc['UMAP_8','Silhouette_Celltype'] = silhouette_score(adata_ATAC.obsm['X_umap'], meta_data['Celltype'])
Results.loc['UMAP_8','Silhouette_Subcelltype'] = silhouette_score(adata_ATAC.obsm['X_umap'], meta_data['Subcelltype'])

Results.loc['UMAP_8','KL_0.01'] = density_kl_div(np.array(PBMC_ATAC),np.array(adata_ATAC.obsm['X_umap']),sigma=0.01)

# UMAP 2
adata_ATAC = sc.AnnData(X = PBMC_ATAC, obs = meta_data)
sc.pp.neighbors(adata_ATAC, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ATAC)
sc.pl.umap(adata_ATAC, color='Celltype')
plt.savefig("UMAP_PBMC_ATAC_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_ATAC, color='Subcelltype',legend_loc='on data')
plt.savefig("UMAP_PBMC_ATAC_Subcelltype.tiff", bbox_inches='tight', dpi=150)

Results.loc['UMAP_2','Silhouette_Celltype'] = silhouette_score(adata_ATAC.obsm['X_umap'], meta_data['Celltype'])
Results.loc['UMAP_2','Silhouette_Subcelltype'] = silhouette_score(adata_ATAC.obsm['X_umap'], meta_data['Subcelltype'])

Results.loc['UMAP_2','KL_0.01'] = density_kl_div(np.array(PBMC_ATAC),np.array(adata_ATAC.obsm['X_umap']),sigma=0.01)

# TopoAE 0.5 (run 10 times)
source_latent = get_TopoAE_Embeddings(Manifold_Data = PBMC_ATAC, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=0.5, initial_LR=0.001)

Results.loc['TopoAE_8_lam_0.5','Silhouette_Celltype'] = silhouette_score(source_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_0.5','Silhouette_Subcelltype'] = silhouette_score(source_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_0.5','KL_0.01'] = density_kl_div(np.array(PBMC_ATAC),np.array(source_latent),sigma=0.01)

adata_ATAC_Topo = sc.AnnData(X = pd.DataFrame(source_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_ATAC_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ATAC_Topo)
sc.pl.umap(adata_ATAC_Topo, color='Celltype')
plt.savefig("TopoAE_lam0.5_PBMC_ATAC_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_ATAC_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam0.5_PBMC_ATAC_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# TopoAE 1.0 (run 10 times)
source_latent = get_TopoAE_Embeddings(Manifold_Data = PBMC_ATAC, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=1, initial_LR=0.001)

Results.loc['TopoAE_8_lam_1.0','Silhouette_Celltype'] = silhouette_score(source_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_1.0','Silhouette_Subcelltype'] = silhouette_score(source_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_1.0','KL_0.01'] = density_kl_div(np.array(PBMC_ATAC),np.array(source_latent),sigma=0.01)

adata_ATAC_Topo = sc.AnnData(X = pd.DataFrame(source_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_ATAC_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ATAC_Topo)
sc.pl.umap(adata_ATAC_Topo, color='Celltype')
plt.savefig("TopoAE_lam1.0_PBMC_ATAC_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_ATAC_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam1.0_PBMC_ATAC_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# TopoAE 2.0 (run 10 times)
source_latent = get_TopoAE_Embeddings(Manifold_Data = PBMC_ATAC, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=2, initial_LR=0.001)

Results.loc['TopoAE_8_lam_2.0','Silhouette_Celltype'] = silhouette_score(source_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_2.0','Silhouette_Subcelltype'] = silhouette_score(source_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_2.0','KL_0.01'] = density_kl_div(np.array(PBMC_ATAC),np.array(source_latent),sigma=0.01)

adata_ATAC_Topo = sc.AnnData(X = pd.DataFrame(source_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_ATAC_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ATAC_Topo)
sc.pl.umap(adata_ATAC_Topo, color='Celltype')
plt.savefig("TopoAE_lam2.0_PBMC_ATAC_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_ATAC_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam2.0_PBMC_ATAC_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# TopoAE 3.0 (run 10 times)
source_latent = get_TopoAE_Embeddings(Manifold_Data = PBMC_ATAC, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=3, initial_LR=0.001)

Results.loc['TopoAE_8_lam_3.0','Silhouette_Celltype'] = silhouette_score(source_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_3.0','Silhouette_Subcelltype'] = silhouette_score(source_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_3.0','KL_0.01'] = density_kl_div(np.array(PBMC_ATAC),np.array(source_latent),sigma=0.01)

adata_ATAC_Topo = sc.AnnData(X = pd.DataFrame(source_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_ATAC_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ATAC_Topo)
sc.pl.umap(adata_ATAC_Topo, color='Celltype')
plt.savefig("TopoAE_lam3.0_PBMC_ATAC_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_ATAC_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam3.0_PBMC_ATAC_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# VAE 8 (run 10 times)
source_latent = get_AE_Embeddings(Manifold_Data = PBMC_ATAC, batch_size=50, model_type= "Variational", AE_arch = [50, 32, 32, 8], initial_LR=0.001)

Results.loc['VAE_8','Silhouette_Celltype'] = silhouette_score(source_latent, meta_data['Celltype'])
Results.loc['VAE_8','Silhouette_Subcelltype'] = silhouette_score(source_latent, meta_data['Subcelltype'])

Results.loc['VAE_8','KL_0.01'] = density_kl_div(np.array(PBMC_ATAC),np.array(source_latent),sigma=0.01)

adata_ATAC_VAE = sc.AnnData(X = pd.DataFrame(source_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_ATAC_VAE, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ATAC_VAE)
sc.pl.umap(adata_ATAC_VAE, color='Celltype')
plt.savefig("VAE_PBMC_ATAC_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_ATAC_VAE, color='Subcelltype',legend_loc='on data')
plt.savefig("VAE_PBMC_ATAC_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# AE 8 (run 10 times)
source_latent = get_AE_Embeddings(Manifold_Data = PBMC_ATAC, batch_size=50, model_type= "regular", AE_arch = [50, 32, 32, 8], initial_LR=0.001)

Results.loc['AE_8','Silhouette_Celltype'] = silhouette_score(source_latent, meta_data['Celltype'])
Results.loc['AE_8','Silhouette_Subcelltype'] = silhouette_score(source_latent, meta_data['Subcelltype'])

Results.loc['AE_8','KL_0.01'] = density_kl_div(np.array(PBMC_ATAC),np.array(source_latent),sigma=0.01)

adata_ATAC_AE = sc.AnnData(X = pd.DataFrame(source_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_ATAC_AE, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ATAC_AE)
sc.pl.umap(adata_ATAC_AE, color='Celltype')
plt.savefig("AE_PBMC_ATAC_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_ATAC_AE, color='Subcelltype',legend_loc='on data')
plt.savefig("AE_PBMC_ATAC_Subcelltype.tiff", bbox_inches='tight', dpi=150)

Results_PBMC_ATAC = Results.copy()
#%% BM CITE-seq RNA
BM_RNA = pd.read_csv('Data/BM Citeseq/RNA_PCA.csv',header=0,index_col=0)
meta_data = pd.read_csv('Data/BM Citeseq/annotations.csv',header=0,index_col=0)

Results = pd.DataFrame(columns=['Silhouette_Celltype','Silhouette_Subcelltype','KL_0.01'])

# UMAP 8
adata_RNA = sc.AnnData(X = BM_RNA, obs = meta_data)
sc.pp.neighbors(adata_RNA, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA,n_components=8)

Results.loc['UMAP_8','Silhouette_Celltype'] = silhouette_score(adata_RNA.obsm['X_umap'], meta_data['Celltype'])
Results.loc['UMAP_8','Silhouette_Subcelltype'] = silhouette_score(adata_RNA.obsm['X_umap'], meta_data['Subcelltype'])

Results.loc['UMAP_8','KL_0.01'] = density_kl_div(np.array(BM_RNA),np.array(adata_RNA.obsm['X_umap']),sigma=0.01)

# UMAP 2
adata_RNA = sc.AnnData(X = BM_RNA, obs = meta_data)
sc.pp.neighbors(adata_RNA, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA)
sc.pl.umap(adata_RNA, color='Celltype')
plt.savefig("UMAP_BM_RNA_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA, color='Subcelltype',legend_loc='on data')
plt.savefig("UMAP_BM_RNA_Subcelltype.tiff", bbox_inches='tight', dpi=150)

Results.loc['UMAP_2','Silhouette_Celltype'] = silhouette_score(adata_RNA.obsm['X_umap'], meta_data['Celltype'])
Results.loc['UMAP_2','Silhouette_Subcelltype'] = silhouette_score(adata_RNA.obsm['X_umap'], meta_data['Subcelltype'])

Results.loc['UMAP_2','KL_0.01'] = density_kl_div(np.array(BM_RNA),np.array(adata_RNA.obsm['X_umap']),sigma=0.01)

# TopoAE 0.5 (run 10 times)
target_latent = get_TopoAE_Embeddings(Manifold_Data = BM_RNA, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=0.5, initial_LR=0.001)

Results.loc['TopoAE_8_lam_0.5','Silhouette_Celltype'] = silhouette_score(target_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_0.5','Silhouette_Subcelltype'] = silhouette_score(target_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_0.5','KL_0.01'] = density_kl_div(np.array(BM_RNA),np.array(target_latent),sigma=0.01)

adata_RNA_Topo = sc.AnnData(X = pd.DataFrame(target_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_RNA_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_Topo)
sc.pl.umap(adata_RNA_Topo, color='Celltype')
plt.savefig("TopoAE_lam0.5_BM_RNA_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam0.5_BM_RNA_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# TopoAE 1.0 (run 10 times)
target_latent = get_TopoAE_Embeddings(Manifold_Data = BM_RNA, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=1, initial_LR=0.001)

Results.loc['TopoAE_8_lam_1.0','Silhouette_Celltype'] = silhouette_score(target_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_1.0','Silhouette_Subcelltype'] = silhouette_score(target_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_1.0','KL_0.01'] = density_kl_div(np.array(BM_RNA),np.array(target_latent),sigma=0.01)

adata_RNA_Topo = sc.AnnData(X = pd.DataFrame(target_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_RNA_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_Topo)
sc.pl.umap(adata_RNA_Topo, color='Celltype')
plt.savefig("TopoAE_lam1.0_BM_RNA_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam1.0_BM_RNA_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# TopoAE 2.0 (run 10 times)
target_latent = get_TopoAE_Embeddings(Manifold_Data = BM_RNA, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=2, initial_LR=0.001)

Results.loc['TopoAE_8_lam_2.0','Silhouette_Celltype'] = silhouette_score(target_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_2.0','Silhouette_Subcelltype'] = silhouette_score(target_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_2.0','KL_0.01'] = density_kl_div(np.array(BM_RNA),np.array(target_latent),sigma=0.01)

adata_RNA_Topo = sc.AnnData(X = pd.DataFrame(target_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_RNA_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_Topo)
sc.pl.umap(adata_RNA_Topo, color='Celltype')
plt.savefig("TopoAE_lam2.0_BM_RNA_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam2.0_BM_RNA_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# TopoAE 3.0 (run 10 times)
target_latent = get_TopoAE_Embeddings(Manifold_Data = BM_RNA, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [50, 32, 32, 8], topology_regulariser_coefficient=3, initial_LR=0.001)

Results.loc['TopoAE_8_lam_3.0','Silhouette_Celltype'] = silhouette_score(target_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_3.0','Silhouette_Subcelltype'] = silhouette_score(target_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_3.0','KL_0.01'] = density_kl_div(np.array(BM_RNA),np.array(target_latent),sigma=0.01)

adata_RNA_Topo = sc.AnnData(X = pd.DataFrame(target_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_RNA_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_Topo)
sc.pl.umap(adata_RNA_Topo, color='Celltype')
plt.savefig("TopoAE_lam3.0_BM_RNA_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam3.0_BM_RNA_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# VAE 8 (run 10 times)
target_latent = get_AE_Embeddings(Manifold_Data = BM_RNA, batch_size=50, model_type= "Variational", AE_arch = [50, 32, 32, 8], initial_LR=0.001)

Results.loc['VAE_8','Silhouette_Celltype'] = silhouette_score(target_latent, meta_data['Celltype'])
Results.loc['VAE_8','Silhouette_Subcelltype'] = silhouette_score(target_latent, meta_data['Subcelltype'])

Results.loc['VAE_8','KL_0.01'] = density_kl_div(np.array(BM_RNA),np.array(target_latent),sigma=0.01)

adata_RNA_VAE = sc.AnnData(X = pd.DataFrame(target_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_RNA_VAE, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_VAE)
sc.pl.umap(adata_RNA_VAE, color='Celltype')
plt.savefig("VAE_BM_RNA_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_VAE, color='Subcelltype',legend_loc='on data')
plt.savefig("VAE_BM_RNA_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# AE 8 (run 10 times)
target_latent = get_AE_Embeddings(Manifold_Data = BM_RNA, batch_size=50, model_type= "regular", AE_arch = [50, 32, 32, 8], initial_LR=0.001)

Results.loc['AE_8','Silhouette_Celltype'] = silhouette_score(target_latent, meta_data['Celltype'])
Results.loc['AE_8','Silhouette_Subcelltype'] = silhouette_score(target_latent, meta_data['Subcelltype'])

Results.loc['AE_8','KL_0.01'] = density_kl_div(np.array(BM_RNA),np.array(target_latent),sigma=0.01)

adata_RNA_AE = sc.AnnData(X = pd.DataFrame(target_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_RNA_AE, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_RNA_AE)
sc.pl.umap(adata_RNA_AE, color='Celltype')
plt.savefig("AE_BM_RNA_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_RNA_AE, color='Subcelltype',legend_loc='on data')
plt.savefig("AE_BM_RNA_Subcelltype.tiff", bbox_inches='tight', dpi=150)

Results_BM_RNA = Results.copy()
#%% BM CITE-seq ADT
BM_ADT = pd.read_csv('Data/BM Citeseq/ADT_Scaled.csv',header=0,index_col=0)
meta_data = pd.read_csv('Data/BM Citeseq/annotations.csv',header=0,index_col=0)

Results = pd.DataFrame(columns=['Silhouette_Celltype','Silhouette_Subcelltype','KL_0.01'])

# UMAP 8
adata_ADT = sc.AnnData(X = BM_ADT, obs = meta_data)
sc.pp.neighbors(adata_ADT, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ADT,n_components=8)

Results.loc['UMAP_8','Silhouette_Celltype'] = silhouette_score(adata_ADT.obsm['X_umap'], meta_data['Celltype'])
Results.loc['UMAP_8','Silhouette_Subcelltype'] = silhouette_score(adata_ADT.obsm['X_umap'], meta_data['Subcelltype'])

Results.loc['UMAP_8','KL_0.01'] = density_kl_div(np.array(BM_ADT),np.array(adata_ADT.obsm['X_umap']),sigma=0.01)

# UMAP 2
adata_ADT = sc.AnnData(X = BM_ADT, obs = meta_data)
sc.pp.neighbors(adata_ADT, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ADT)
sc.pl.umap(adata_ADT, color='Celltype')
plt.savefig("UMAP_BM_ADT_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_ADT, color='Subcelltype',legend_loc='on data')
plt.savefig("UMAP_BM_ADT_Subcelltype.tiff", bbox_inches='tight', dpi=150)

Results.loc['UMAP_2','Silhouette_Celltype'] = silhouette_score(adata_ADT.obsm['X_umap'], meta_data['Celltype'])
Results.loc['UMAP_2','Silhouette_Subcelltype'] = silhouette_score(adata_ADT.obsm['X_umap'], meta_data['Subcelltype'])

Results.loc['UMAP_2','KL_0.01'] = density_kl_div(np.array(BM_ADT),np.array(adata_ADT.obsm['X_umap']),sigma=0.01)

# TopoAE 0.5 (run 10 times)
source_latent = get_TopoAE_Embeddings(Manifold_Data = BM_ADT, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [25, 16, 16, 8], topology_regulariser_coefficient=0.5, initial_LR=0.001)

Results.loc['TopoAE_8_lam_0.5','Silhouette_Celltype'] = silhouette_score(source_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_0.5','Silhouette_Subcelltype'] = silhouette_score(source_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_0.5','KL_0.01'] = density_kl_div(np.array(BM_ADT),np.array(source_latent),sigma=0.01)

adata_ADT_Topo = sc.AnnData(X = pd.DataFrame(source_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_ADT_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ADT_Topo)
sc.pl.umap(adata_ADT_Topo, color='Celltype')
plt.savefig("TopoAE_lam0.5_BM_ADT_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_ADT_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam0.5_BM_ADT_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# TopoAE 1.0 (run 10 times)
source_latent = get_TopoAE_Embeddings(Manifold_Data = BM_ADT, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [25, 16, 16, 8], topology_regulariser_coefficient=1, initial_LR=0.001)

Results.loc['TopoAE_8_lam_1.0','Silhouette_Celltype'] = silhouette_score(source_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_1.0','Silhouette_Subcelltype'] = silhouette_score(source_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_1.0','KL_0.01'] = density_kl_div(np.array(BM_ADT),np.array(source_latent),sigma=0.01)

adata_ADT_Topo = sc.AnnData(X = pd.DataFrame(source_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_ADT_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ADT_Topo)
sc.pl.umap(adata_ADT_Topo, color='Celltype')
plt.savefig("TopoAE_lam1.0_BM_ADT_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_ADT_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam1.0_BM_ADT_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# TopoAE 2.0 (run 10 times)
source_latent = get_TopoAE_Embeddings(Manifold_Data = BM_ADT, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [25, 16, 16, 8], topology_regulariser_coefficient=2, initial_LR=0.001)

Results.loc['TopoAE_8_lam_2.0','Silhouette_Celltype'] = silhouette_score(source_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_2.0','Silhouette_Subcelltype'] = silhouette_score(source_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_2.0','KL_0.01'] = density_kl_div(np.array(BM_ADT),np.array(source_latent),sigma=0.01)

adata_ADT_Topo = sc.AnnData(X = pd.DataFrame(source_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_ADT_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ADT_Topo)
sc.pl.umap(adata_ADT_Topo, color='Celltype')
plt.savefig("TopoAE_lam2.0_BM_ADT_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_ADT_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam2.0_BM_ADT_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# TopoAE 3.0 (run 10 times)
source_latent = get_TopoAE_Embeddings(Manifold_Data = BM_ADT, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC", 
                                      AE_arch = [25, 16, 16, 8], topology_regulariser_coefficient=3, initial_LR=0.001)

Results.loc['TopoAE_8_lam_3.0','Silhouette_Celltype'] = silhouette_score(source_latent, meta_data['Celltype'])
Results.loc['TopoAE_8_lam_3.0','Silhouette_Subcelltype'] = silhouette_score(source_latent, meta_data['Subcelltype'])

Results.loc['TopoAE_8_lam_3.0','KL_0.01'] = density_kl_div(np.array(BM_ADT),np.array(source_latent),sigma=0.01)

adata_ADT_Topo = sc.AnnData(X = pd.DataFrame(source_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_ADT_Topo, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ADT_Topo)
sc.pl.umap(adata_ADT_Topo, color='Celltype')
plt.savefig("TopoAE_lam3.0_BM_ADT_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_ADT_Topo, color='Subcelltype',legend_loc='on data')
plt.savefig("TopoAE_lam3.0_BM_ADT_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# VAE 8 (run 10 times)
source_latent = get_AE_Embeddings(Manifold_Data = BM_ADT, batch_size=50, model_type= "Variational", AE_arch = [25, 16, 16, 8], initial_LR=0.001)

Results.loc['VAE_8','Silhouette_Celltype'] = silhouette_score(source_latent, meta_data['Celltype'])
Results.loc['VAE_8','Silhouette_Subcelltype'] = silhouette_score(source_latent, meta_data['Subcelltype'])

Results.loc['VAE_8','KL_0.01'] = density_kl_div(np.array(BM_ADT),np.array(source_latent),sigma=0.01)

adata_ADT_VAE = sc.AnnData(X = pd.DataFrame(source_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_ADT_VAE, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ADT_VAE)
sc.pl.umap(adata_ADT_VAE, color='Celltype')
plt.savefig("VAE_BM_ADT_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_ADT_VAE, color='Subcelltype',legend_loc='on data')
plt.savefig("VAE_BM_ADT_Subcelltype.tiff", bbox_inches='tight', dpi=150)

# AE 8 (run 10 times)
source_latent = get_AE_Embeddings(Manifold_Data = BM_ADT, batch_size=50, model_type= "regular", AE_arch = [25, 16, 16, 8], initial_LR=0.001)

Results.loc['AE_8','Silhouette_Celltype'] = silhouette_score(source_latent, meta_data['Celltype'])
Results.loc['AE_8','Silhouette_Subcelltype'] = silhouette_score(source_latent, meta_data['Subcelltype'])

Results.loc['AE_8','KL_0.01'] = density_kl_div(np.array(BM_ADT),np.array(source_latent),sigma=0.01)

adata_ADT_AE = sc.AnnData(X = pd.DataFrame(source_latent,index=meta_data.index), obs = meta_data)
sc.pp.neighbors(adata_ADT_AE, n_neighbors=30, n_pcs=0)
sc.tl.umap(adata_ADT_AE)
sc.pl.umap(adata_ADT_AE, color='Celltype')
plt.savefig("AE_BM_ADT_Celltype.tiff", bbox_inches='tight', dpi=150)
sc.pl.umap(adata_ADT_AE, color='Subcelltype',legend_loc='on data')
plt.savefig("AE_BM_ADT_Subcelltype.tiff", bbox_inches='tight', dpi=150)

Results_BM_ADT = Results.copy()