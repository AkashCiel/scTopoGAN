import numpy as np
from scipy.spatial.distance import pdist, squareform


def get_neighbours_and_ranks(distances, k):
    """
    Inputs:
    - distances,        distance matrix [n times n],
    - k,                number of nearest neighbours to consider
    Returns:
    - neighbourhood,    contains the sample indices (from 0 to n-1) of kth nearest neighbor of current sample [n times k]
    - ranks,            contains the rank of each sample to each sample [n times n], whereas entry (i,j) gives the rank that sample j has to i (the how many 'closest' neighbour j is to i)
    """
    # Warning: this is only the ordering of neighbours that we need to
    # extract neighbourhoods below. The ranking comes later!
    indices = np.argsort(distances, axis=-1, kind='stable')

    # Extract neighbourhoods.
    neighbourhood = indices[:, 1:k + 1]

    # Convert this into ranks (finally)
    ranks = indices.argsort(axis=-1, kind='stable')

    return neighbourhood, ranks

def get_trustworthiness(X_neighbourhood, X_ranks, Z_neighbourhood, Z_ranks, n, k):
    '''
    Checks to what extent the k nearest neighbours of a point are preserved when going from the
    original space to the latent space

    Calculates the trustworthiness measure between the data space `X`
    and the latent space `Z`, given a neighbourhood parameter `k` for
    defining the extent of neighbourhoods.

    HIGHER is better
    '''

    result = 0.0

    # Calculate number of neighbours that are in the $k$-neighbourhood
    # of the latent space but not in the $k$-neighbourhood of the data
    # space.
    for row in range(X_ranks.shape[0]):
        missing_neighbours = np.setdiff1d(
            Z_neighbourhood[row],
            X_neighbourhood[row]
        )

        for neighbour in missing_neighbours:
            result += (X_ranks[row, neighbour] - k)

    return 1 - 2 / (n * k * (2 * n - 3 * k - 1) ) * result

def get_mrre(X_neighbourhood, X_ranks, Z_neighbourhood, Z_ranks, n, k):
    '''
    Calculates the mean relative rank error quality metric of the data
    space `X` with respect to the latent space `Z`, subject to its $k$
    nearest neighbours.

    LOWER is better
    '''

    # First component goes from the latent space to the data space, i.e.
    # the relative quality of neighbours in `Z`.

    mrre_ZX = 0.0
    for row in range(n):
        for neighbour in Z_neighbourhood[row]:
            rx = X_ranks[row, neighbour]
            rz = Z_ranks[row, neighbour]

            mrre_ZX += abs(rx - rz) / rz

    # Second component goes from the data space to the latent space,
    # i.e. the relative quality of neighbours in `X`.

    mrre_XZ = 0.0
    for row in range(n):
        # Note that this uses a different neighbourhood definition!
        for neighbour in X_neighbourhood[row]:
            rx = X_ranks[row, neighbour]
            rz = Z_ranks[row, neighbour]

            # Note that this uses a different normalisation factor
            mrre_XZ += abs(rx - rz) / rx

    # Normalisation constant
    C = n * sum([abs(2*j - n - 1) / j for j in range(1, k+1)])
    return mrre_XZ / C, mrre_ZX / C

def density_kl_global(pairwise_X, pairwise_Z, sigma):
    X = pairwise_X
    X = X / X.max()
    Z = pairwise_Z
    Z = Z / Z.max()

    density_x = np.sum(np.exp(-(X ** 2) / sigma), axis=-1)
    density_x /= density_x.sum(axis=-1)

    density_z = np.sum(np.exp(-(Z ** 2) / sigma), axis=-1)
    density_z /= density_z.sum(axis=-1)

    return (density_x * (np.log(density_x) - np.log(density_z))).sum()

def evaluate_model(pairwise_X, pairwise_Z, sigma):

    qualitative_metrics = density_kl_global(pairwise_X, pairwise_Z, sigma=sigma)
    """
    X_neighbourhood, X_ranks = get_neighbours_and_ranks(pairwise_X, k)
    Z_neighbourhood, Z_ranks = get_neighbours_and_ranks(pairwise_Z, k)

    n = pairwise_X.shape[0]

    MRRE_original_to_latent, MRRE_latent_to_original = get_mrre(X_neighbourhood, X_ranks, Z_neighbourhood, Z_ranks, n, k)
    qualitative_metrics = {"Density_KL_Global_1" : density_kl_global(pairwise_X, pairwise_Z, sigma=1.0),
                           "Density_KL_Global_01" : density_kl_global(pairwise_X, pairwise_Z, sigma=0.1),
                           "Density_KL_Global_001" : density_kl_global(pairwise_X, pairwise_Z, sigma=0.01),
                           "Trustworthiness" : get_trustworthiness(X_neighbourhood, X_ranks, Z_neighbourhood,
                                     Z_ranks, n, k),
                           "Continuity" : get_trustworthiness(Z_neighbourhood, Z_ranks, X_neighbourhood,
                                     X_ranks, n, k),
                           "MRRE_original_to_latent" : MRRE_original_to_latent,
                           "MRRE_latent_to_original" : MRRE_latent_to_original}
    """
    return qualitative_metrics

