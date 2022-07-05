"""
    Just to try things suddenly
"""
import learn as learn
from sklearn.metrics import silhouette_score
from Qualitative_Metrics import *
import pandas as pd
import time
import matplotlib.pyplot as plt
import umap.umap_ as umap
from datetime import datetime
import seaborn as sns

rows_to_consider = 10500
ATAC_Reduced = pd.read_csv("PBMC/ATAC Processed Reduced.csv", header=0, delimiter=',', index_col = 0, nrows=rows_to_consider)
ATAC_Labels = pd.read_csv("PBMC/Annotations_ATAC.csv", header=0, delimiter=',', index_col = 0, nrows=rows_to_consider)
RNA_Reduced = pd.read_csv("PBMC/RNA Processed Reduced.csv", header=0, delimiter=',', index_col = 0, nrows=rows_to_consider)
RNA_Labels = pd.read_csv("PBMC/Annotations_RNA.csv", header=0, delimiter=',', index_col = 0, nrows=rows_to_consider)

def perform_dim_reduction(tsne_data, tsne_data_annotated, xlabel, ylabel):
    time_start = time.time()
    umap_instance = umap.UMAP(n_components=2, random_state=42)
    umap_results = umap_instance.fit_transform(tsne_data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    tsne_with_indices = pd.DataFrame(index=list(tsne_data.index.values))
    tsne_with_indices[xlabel] = umap_results[:,0]
    tsne_with_indices[ylabel] = umap_results[:,1]

    tsne_data_annotated = tsne_data_annotated.merge(tsne_with_indices, how='inner', left_index=True, right_index=True)

    return tsne_data_annotated

def plot_embeddings(tsne_data_annotated, xlabel, ylabel, tsne_labels):

    num_classes = len(set(list(tsne_data_annotated[tsne_labels])))
    plt.figure(figsize=(15,15))
    latent_tsne_plot = sns.scatterplot(
        x=xlabel, y=ylabel,
        hue=tsne_labels,
        palette=sns.color_palette("bright", num_classes),
        data=tsne_data_annotated,
        legend="full",
        alpha=0.3
    )
    latent_tsne_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
    plt.show()
    return 0


print("Doing ATAC Data")

ATAC_Labels = perform_dim_reduction(ATAC_Reduced, ATAC_Labels, "Embedding_X", "Embedding_Y")

ATAC_Embeddings = ATAC_Labels[["Embedding_X", "Embedding_Y"]]
ATAC_Label = ATAC_Labels[["cell_type"]]

# Compute pairwise distances for evaluation later
print("Computing pairwise distances . . .")
pairwise_ATAC = squareform(pdist(ATAC_Reduced))
pairwise_ATAC_embeddings = squareform(pdist(ATAC_Embeddings))

#plot_embeddings(ATAC_Labels, "Embedding_X", "Embedding_Y", "cell_type")
#plot_embeddings(RNA_Labels, "Embedding_X", "Embedding_Y", "cell_type")

latent_score_ATAC = silhouette_score(ATAC_Embeddings, ATAC_Label)
KL_Global_Density_ATAC = evaluate_model(pairwise_ATAC, pairwise_ATAC_embeddings, 0.01)

print("Silhouette score: ", latent_score_ATAC)
print("KL Metric: ", KL_Global_Density_ATAC)

print("Doing RNA Data")

RNA_Labels = perform_dim_reduction(RNA_Reduced, RNA_Labels, "Embedding_X", "Embedding_Y")

RNA_Embeddings = RNA_Labels[["Embedding_X", "Embedding_Y"]]
RNA_Label = RNA_Labels[["cell_type"]]

# Compute pairwise distances for evaluation later
print("Computing pairwise distances . . .")
pairwise_RNA = squareform(pdist(RNA_Reduced))
pairwise_RNA_embeddings = squareform(pdist(RNA_Embeddings))

#plot_embeddings(ATAC_Labels, "Embedding_X", "Embedding_Y", "cell_type")
#plot_embeddings(RNA_Labels, "Embedding_X", "Embedding_Y", "cell_type")

latent_score_RNA = silhouette_score(RNA_Embeddings, RNA_Label)
KL_Global_Density_RNA = evaluate_model(pairwise_RNA, pairwise_RNA_embeddings, 0.01)

print("Silhouette score: ", latent_score_RNA)
print("KL Metric: ", KL_Global_Density_RNA)