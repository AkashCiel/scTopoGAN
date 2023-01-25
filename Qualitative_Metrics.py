import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KNeighborsClassifier

def density_kl_div(X, Z, sigma):
    pairwise_X = squareform(pdist(X))
    pairwise_X = pairwise_X / pairwise_X.max()
    pairwise_Z = squareform(pdist(Z))
    pairwise_Z = pairwise_Z / pairwise_Z.max()

    density_x = np.sum(np.exp(-(pairwise_X ** 2) / sigma), axis=-1)
    density_x /= density_x.sum(axis=-1)

    density_z = np.sum(np.exp(-(pairwise_Z ** 2) / sigma), axis=-1)
    density_z /= density_z.sum(axis=-1)

    return (density_x * (np.log(density_x) - np.log(density_z))).sum()

def evaluate_neighbors(source_aligned, target_tech, source_annotation, target_annotation, k):
    # Perfom knn estimation
    neighbors = KNeighborsClassifier(n_neighbors=k, algorithm='auto').fit(target_tech, target_annotation)  # Dataset to find the neighbours in
    predicted_annotation = neighbors.predict(source_aligned)  # Query dataset
    
    return 100 * np.sum(source_annotation == predicted_annotation)/len(source_annotation)

