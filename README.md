# TopoGAN
## Unsupervised manifold alignment of single-cell data

### Implementation description
Python implementation can be found in the 'TopoGAN' folder. The ```TopoGAN_Functions.py``` file contains the implementation of all the necessary functions. The main two functions are 1) ```get_TopoAE_Embeddings``` which obtains the low-dimensional manifold of each dataset separately using Topological Autoencoder, and 2) ```run_TopoGAN``` which aligns the source manifold to the target manifold using the topology-guided GAN model.

The ```Run_TopoGAN.py``` script provides an example for running TopoGAN on a new data. User needs to specify the input directory, as well as the source data and target data names matching the input file names.

### Results reproducibility
To reproduce the results obtained for the PBMC data, the following scripts can be run in order:

1) The ```TopoAE_on_PBMC.py``` script should be performed twice, once for each modality (RNA and ATAC). This will provide the Topological Autoencoder low-dimensinal representation for each modality used as input to the next step. Please note, this script takes as input the pre-processed data. 
2) The ```TopoGAN_First_Generation.py``` script aligns the source and target manifolds using 20 different GAN models, and calculates the average topological loss of each model to select the final model with the lowest value. 
3) The ```TopoGAN_Second_Generation.py``` script trains the final model for an additional 1000 epochs to provide the final alignment, and evaluates the alignment using the celltype-matching and Subcelltype-matching scores.

For citation and further information please refer to: "TopoGAN: Unsupervised manifold alignment of single-cell data", [bioRxiv2022](https://www.biorxiv.org/)
