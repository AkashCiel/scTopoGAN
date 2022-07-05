"""Competitor dimensionality reduction algorithms."""
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import umap as UMAP

try:
    from MulticoreTSNE import MulticoreTSNE as TSNE

except ImportError:
    from sklearn.manifold import TSNE
