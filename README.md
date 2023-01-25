# scTopoGAN
## Unsupervised manifold alignment of single-cell data

### Implementation description
This repository contains the Python implementation of the scTopoGAN method. The ```scTopoGAN_Functions.py``` file contains the implementation of all the necessary functions. The main two functions are 1) ```get_TopoAE_Embeddings``` which obtains the low-dimensional manifold of each dataset separately using Topological Autoencoder, and 2) ```run_scTopoGAN``` which aligns the source manifold to the target manifold using the topology-guided GAN model.

The ```Run_scTopoGAN.py``` script provides an example for running scTopoGAN on a new data.

### Results reproducibility
The following scripts can be used to reproduce the results obtained in the preprint:

1) The ```TopoAE_benchmark.py``` script compares the performance of the Topological Autoencoder with regular Autoencoder, Aariational Autoencoder and UMAP. This script provides the low-dimensinal representation for each modality using different methods. Please note, this script takes as input the pre-processed data provided in the ```Data``` folder. 
2) The ```scTopoGAN_results.py``` script reproduce the alignment results of scTopoGAN using different datasets. 
3) The ```UnionCom.py``` and ```manifold_align_mmd_pytorch.py``` scripts are used to apply UnionCom and MMD-MA, respectively. Both methods were used to benchmark the performance of scTopoGAN.

For citation and further information please refer to: "scTopoGAN: Unsupervised manifold alignment of single-cell data", [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.04.27.489829v2)
