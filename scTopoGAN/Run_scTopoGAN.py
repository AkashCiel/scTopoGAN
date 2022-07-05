from scTopoGAN_Functions import *

in_directory = "PBMC"
source_tech_name = "ATAC_Raw"
target_tech_name = "RNA_Raw"

source_latent, source_annotations = get_TopoAE_Embeddings(in_directory, raw_data=source_tech_name, batch_size=50, topology_regulariser_coefficient=0.5, initial_LR=0.001)
target_latent, target_annotations = get_TopoAE_Embeddings(in_directory, raw_data=target_tech_name, batch_size=50, topology_regulariser_coefficient=0.5, initial_LR=0.001)

source_aligned = run_scTopoGAN(source_tech=source_latent, target_tech=target_latent,
                           source_tech_name=source_tech_name, target_tech_name=target_tech_name,
                           batch_size=50, num_iterations=2, total_epochs=1001, checkpoint_epoch=100, learning_rate=0.0001,
                           path_prefix="Results", topology_batch_size=1000,
                           Annotation_Data_Source=source_annotations, Annotation_Data_Target=target_annotations)

print(source_aligned)
source_aligned.to_csv(path_or_buf="{}/{}_Aligned.csv".format(in_directory, source_tech_name), header=True, index=True)
