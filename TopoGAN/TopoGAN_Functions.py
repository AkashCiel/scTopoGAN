# Import modules
from copy import deepcopy
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from src.models.approx_based import *
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import seaborn as sns
from Qualitative_Metrics import *
from Manifold_Alignment_GAN import *
from Manifold_Alignment_GAN_Ensemble import *
from src.models.Assess_Topology_V02 import *
from src.models.submodules import VAE_PBMC
import os

def prepare_manifold_data(in_directory, raw_data, batch_size):
    Manifold_Data = pd.read_csv("{}/{}.csv".format(in_directory, raw_data),
                                header=0, delimiter=',', index_col = 0)
    all_samples = list(Manifold_Data.index.values)
    features_threshold = len(Manifold_Data.columns)
    Annotation_Data = pd.read_csv("{}/Annotations_{}.csv".format(in_directory, raw_data), header=0, delimiter=',', index_col = 0)
    VAE_Data = Manifold_Data.merge(Annotation_Data, how='inner', left_index=True, right_index=True)
    VAE_Data_numpy = VAE_Data.to_numpy()
    VAE_Data_Features, VAE_Data_Labels = np.array_split(VAE_Data_numpy, [features_threshold], axis=1)
    VAE_Data_Features = VAE_Data_Features.astype(float)
    VAE_Data_Features_tensor = torch.tensor(VAE_Data_Features).float() # Convert to tensor

    vae_train, vae_val, labels_train, labels_val = train_test_split(VAE_Data_Features_tensor, VAE_Data_Labels,
                                                                    test_size=0.20, random_state = 42)

    dataloaders = {'train': DataLoader(vae_train, batch_size=batch_size, shuffle=True, num_workers=0),
                    'val': DataLoader(vae_val, batch_size=batch_size, shuffle=True, num_workers=0)
                    }

    print("Train data: ", vae_train.shape)
    print("Val data: ", vae_val.shape)

    return VAE_Data_Features, all_samples, Annotation_Data, dataloaders

def plot_training(x, train_losses, val_losses):
    plt.plot(x, train_losses, label="Training loss")
    plt.plot(x, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def train_TopoAE(n_epochs, my_model, learning_rate, weight_decay,
          early_stop_threshold, _rnd, grad_threshold, dataloaders):

    model = my_model

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate,
        weight_decay=weight_decay)

    time_start = datetime.now().time()
    train_losses = []
    val_losses = []
    no_imp_count = 0
    best_val_loss = 1000000  # Set high initial value of best validation loss
    model = model.float()
    for epoch in range(n_epochs):
        train_loss = 0
        model.train()
        for train_batch_id, data in enumerate(dataloaders['train']):
            # Set model into training mode and compute loss
            #print(data.dtype)
            """
            for parameter in model.parameters():
                print(parameter.dtype)
            """
            loss, loss_components = model(data.float())

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_threshold)  # Normalise gradients to prevent gradient explosion
            optimizer.step()

        if train_batch_id == 0:
            train_batch_id+=1
        train_loss = train_loss / train_batch_id
        train_losses.append(train_loss)
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss))

        val_loss = 0
        model.eval()
        for val_batch_id, data in enumerate(dataloaders['val']):
            loss, loss_components = model(data.float())
            val_loss += loss.item()
        if val_batch_id == 0:
            val_batch_id+=1
        val_loss = val_loss / val_batch_id
        val_losses.append(val_loss)

        if (val_loss < best_val_loss):
            print("Improved model with validation loss: {}".format(val_loss))
            best_model_wts = deepcopy(model.state_dict())
            best_val_loss = val_loss
            no_imp_count = 0

        if (val_loss >= best_val_loss):
            no_imp_count += 1
            if epoch > early_stop_threshold:
                if (no_imp_count > 10):
                    print("No improvements for 10 epochs stopping early")
                    model.load_state_dict(best_model_wts)
                    x = range(epoch + 1)
                    plot_training(x, train_losses, val_losses)
                    return model

    x = range(epoch + 1)
    plot_training(x, train_losses, val_losses)
    return model

def get_TopoAE_Embeddings(in_directory, raw_data, batch_size, topology_regulariser_coefficient, initial_LR):
    VAE_Data_Features, \
    all_samples, Annotation_Data, dataloaders = prepare_manifold_data(in_directory, raw_data, batch_size)

    my_model = TopologicallyRegularizedAutoencoder(lam=topology_regulariser_coefficient,
                                                   autoencoder_model="MLPAutoencoder_PBMC",
                                                   toposig_kwargs={"match_edges": "symmetric"})
    trained_model = train_TopoAE(n_epochs=101, my_model=my_model, learning_rate=initial_LR,
                                 weight_decay=1e-05, early_stop_threshold=10, _rnd=42,
                                 grad_threshold=50, dataloaders=dataloaders)

    data_loader = DataLoader(VAE_Data_Features, batch_size=1)
    latent_data = []
    latent_data = torch.empty(0)
    for index, original_sample in enumerate(data_loader):
        latent_sample = trained_model.encode(original_sample.float())
        latent_data = torch.cat((latent_data, latent_sample))
    latent_data = latent_data.detach()

    latent_data = latent_data.numpy()
    latent_data = pd.DataFrame(latent_data, index=all_samples)

    return latent_data, Annotation_Data

def get_majority_local_category(target_tech_annotated, indices, category):
    categories = [target_tech_annotated.iloc[[index]][category].item() for index in indices]
    return max(categories, key=categories.count)

def evaluate_first_k_neighbors(source_to_target, source_to_target_annotated,
                               target_tech, target_tech_annotated, k):
    # Perfom knn estimation
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(target_tech)  # Dataset to find the neighbours in
    all_distances, all_indices = neighbors.kneighbors(source_to_target)  # Query dataset

    sub_cell_type_matches = []
    cell_type_matches = []
    for i in range(len(all_indices)):
        if (i % 500) == 0:
            print('\r', "Evaluating sample #{}".format(i), end='')
        sub_cell_type_source = source_to_target_annotated.iloc[[i]]["cell_type_00"].item()
        cell_type_source = source_to_target_annotated.iloc[[i]]["cell_type"].item()

        sub_cell_type_neighbors = get_majority_local_category(target_tech_annotated, all_indices[i], "cell_type_00")
        cell_type_neighbors = get_majority_local_category(target_tech_annotated, all_indices[i], "cell_type")

        sub_cell_type_matches.append(1 * (sub_cell_type_source == sub_cell_type_neighbors))
        cell_type_matches.append(1 * (cell_type_source == cell_type_neighbors))

    evaluation = {"Percentage sub cell_type matches": 100 * (sum(sub_cell_type_matches) / len(sub_cell_type_matches)),
                  "Percentage cell type matches": 100 * (sum(cell_type_matches) / len(cell_type_matches))}

    return evaluation

def run_TopoGAN(source_tech, target_tech, source_tech_name, target_tech_name, batch_size, num_iterations,
                total_epochs, checkpoint_epoch, learning_rate, path_prefix, topology_batch_size,
                Annotation_Data_Source, Annotation_Data_Target):
    epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    core_suffix = "TopoGAN_Generation01"
    isExist = os.path.exists(path_prefix)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path_prefix)
        os.makedirs("{}/Models".format(path_prefix))
        os.makedirs("{}/Evaluation Results".format(path_prefix))

    latent_dimensions = source_tech.shape[1]
    source_tech = source_tech.to_numpy()
    source_tech = source_tech.astype(float)
    print("Source Technology: ", np.shape(source_tech))
    target_tech = target_tech.to_numpy()
    target_tech = target_tech.astype(float)
    print("Target Technology: ", np.shape(target_tech))

    source_target_numpy = np.concatenate((source_tech, target_tech), axis=1)
    source_target_tensor = torch.tensor(source_target_numpy).float()  # Convert to tensor

    # Define data loader
    data_loader = DataLoader(source_target_tensor, batch_size=batch_size, shuffle=False, num_workers=0)

    print("=======================================================================")
    print("Training first generation")
    print("=======================================================================")
    generator = GeneratorNet(input_dim=latent_dimensions, output_dim=latent_dimensions)
    discriminator = DiscriminatorNet(input_dim=latent_dimensions)
    techs = [source_tech_name, target_tech_name]

    aggregate_topo_error = []

    for random_seed in range(num_iterations):
        # Set random seed for model uniqueness and reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        tf.random.set_seed(random_seed)
        path_suffix = "{}_MODEL_{}".format(core_suffix, str(random_seed))
        trained_generator = train(generator, discriminator, data_loader,
                                  total_epochs, checkpoint_epoch,
                                  learning_rate, techs, path_prefix, path_suffix)

        Evaluation_Columns = ["Model", "Source", "Target", "Iteration", "Epoch",
                              "Source to Projected Topology Loss"]

        evaluation_results = []
        topo_errors = []

        for epoch in epochs:

            print("#########################################################")
            print("#########################################################")
            print("")
            print("Evaluating for epoch: ", epoch)
            path = "{}/Models/{}_to_{}_Generator_{}_{}.pt".format(path_prefix,
                                                                  source_tech_name,
                                                                  target_tech_name,
                                                                  epoch,
                                                                  path_suffix)
            print("Model path: ", path)

            # Project source into target space
            model = torch.load(path)  # Load trained model
            source_to_target = torch.empty(0)

            source_tech = torch.tensor(source_tech).float()
            for tensor in source_tech:
                projected_tensor = model(tensor).reshape(1, 8)
                source_to_target = torch.cat((source_to_target, projected_tensor))

            source_to_target = source_to_target.detach().numpy()
            print("source data shape: ", source_to_target.shape)
            source_projected_numpy = np.concatenate((source_tech, source_to_target), axis=1)
            source_projected_tensor = torch.tensor(source_projected_numpy).float()  # Convert to tensor
            # Define data loader
            data_loader_source_projected = DataLoader(source_projected_tensor, batch_size=topology_batch_size,
                                                      shuffle=False, num_workers=0)

            # Evaluate topology between source and projected
            total_topo_loss_source_projected = 0
            for n_batch, data in enumerate(data_loader_source_projected):
                first_dim = data.shape[0]
                source_batch = tf.slice(data, [0, 0], [first_dim, 8])
                source_to_target_batch = tf.slice(data, [0, 8], [first_dim, 8])

                # Convert source and target from eager tensor to native tensor
                source_to_target_batch = source_to_target_batch.numpy()
                source_to_target_batch = torch.tensor(source_to_target_batch)
                source_batch = source_batch.numpy()
                source_batch = torch.tensor(source_batch)

                topo_error = compute_topological_error(source_batch, source_to_target_batch)
                total_topo_loss_source_projected += topo_error.item()

            evaluation_results.append(["GAN", source_tech_name, target_tech_name, path_suffix, epoch,
                                       total_topo_loss_source_projected])
            if epoch > 500:
                topo_errors.append(total_topo_loss_source_projected)
        mean_topo_error = sum(topo_errors) / len(topo_errors)
        aggregate_topo_error.append([random_seed, mean_topo_error])

        # Save evaluation results
        evaluation_results = pd.DataFrame(evaluation_results, columns=Evaluation_Columns)
        evaluation_results.to_csv(
            path_or_buf="{}/Evaluation Results/Topological Assessment {}.csv".format(path_prefix, path_suffix),
            header=True, index=False)
    # </editor-fold>

    aggregate_topo_error = pd.DataFrame(data=aggregate_topo_error, columns=["Seed", "Mean Error"]).sort_values(
        by="Mean Error", ascending=True)
    print("Recommended model number for second generation of training: ")
    print(aggregate_topo_error.head(1))
    aggregate_topo_error.to_csv(
        path_or_buf="{}/Evaluation Results/Aggregate Topological Assessment First Generation.csv".format(path_prefix),
        header=True, index=False)

    model_01 = aggregate_topo_error.head(1)["Seed"].values[0]
    model_02 = aggregate_topo_error.head(1)["Seed"].values[0]
    epoch_01 = 1000
    epoch_02 = 1000
    random_seed = 1
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    tf.random.set_seed(random_seed)
    load_path_prefix = "Results"  # The first generation models will be loaded the "Models" folder in this directory
    ensemble_path_prefix = "Results"  # Results from this generation will be stored in this directory


    model_01_name = "{}_to_{}_Generator_{}_{}_MODEL_{}".format(source_tech_name, target_tech_name, epoch_01, core_suffix, model_01)
    model_02_name = "{}_to_{}_Generator_{}_{}_MODEL_{}".format(source_tech_name, target_tech_name, epoch_02, core_suffix, model_02)

    path_01 = "{}/Models/{}.pt".format(load_path_prefix, model_01_name)
    path_02 = "{}/Models/{}.pt".format(load_path_prefix, model_02_name)

    # Load models and create ensemble Generator Net
    generator_1 = torch.load(path_01)
    generator_2 = torch.load(path_02)
    generator_ensemble = GeneratorNet(input_dim=latent_dimensions, output_dim=latent_dimensions)

    beta = 0.5  # The mixing factor
    params1 = generator_1.named_parameters()
    params2 = generator_2.named_parameters()
    params_ensemble = generator_ensemble.named_parameters()

    dict_params1 = dict(params1)
    dict_params2 = dict(params2)
    dict_params_ensemble = dict(params_ensemble)

    for name1 in dict_params_ensemble:
        dict_params_ensemble[name1].data.copy_(beta * dict_params1[name1].data + (1 - beta) * dict_params2[name1].data)

    generator_ensemble.load_state_dict(dict_params_ensemble)

    discriminator = DiscriminatorNet(input_dim=latent_dimensions)
    techs = [source_tech_name, target_tech_name]
    path_suffix = "Generation02_{}_{}_{}".format(model_01, model_02, str(random_seed))

    print("=======================================================================")
    print("Training second generation")
    print("=======================================================================")

    trained_generator = train(generator_ensemble, discriminator, data_loader,
                              total_epochs, checkpoint_epoch,
                              learning_rate, techs, ensemble_path_prefix, path_suffix)

    Topo_Evaluation_Columns = ["Model", "Source", "Target", "Iteration", "Epoch",
                               "Source to Projected Topology Loss"]

    Evaluation_Columns = ["Model", "Source", "Target", "Epoch", "k",
                          "Percentage sub cell_type matches",
                          "Percentage cell type matches"]

    Topo_evaluation_results = []

    evaluation_results = []

    for epoch in epochs:

        print("#########################################################")
        print("#########################################################")
        print("")
        print("Evaluating for epoch: ", epoch)
        path = "{}/Models/{}_to_{}_Generator_{}_{}.pt".format(ensemble_path_prefix,
                                                              source_tech_name,
                                                              target_tech_name,
                                                              epoch,
                                                              path_suffix)
        print("Model path: ", path)

        # Project source into target space
        model = torch.load(path)  # Load trained model
        source_to_target = torch.empty(0)

        source_tech = torch.tensor(source_tech).float()
        for tensor in source_tech:
            projected_tensor = model(tensor).reshape(1, 8)
            source_to_target = torch.cat((source_to_target, projected_tensor))

        source_to_target = source_to_target.detach().numpy()
        print("source data shape: ", source_to_target.shape)
        source_projected_numpy = np.concatenate((source_tech, source_to_target), axis=1)
        source_projected_tensor = torch.tensor(source_projected_numpy).float()  # Convert to tensor
        # Define data loader
        data_loader_source_projected = DataLoader(source_projected_tensor, batch_size=topology_batch_size,
                                                  shuffle=False, num_workers=0)

        # Evaluate topology between source and projected
        total_topo_loss_source_projected = 0
        for n_batch, data in enumerate(data_loader_source_projected):
            first_dim = data.shape[0]
            source_batch = tf.slice(data, [0, 0], [first_dim, 8])
            source_to_target_batch = tf.slice(data, [0, 8], [first_dim, 8])

            # Convert source and target from eager tensor to native tensor
            source_to_target_batch = source_to_target_batch.numpy()
            source_to_target_batch = torch.tensor(source_to_target_batch)
            source_batch = source_batch.numpy()
            source_batch = torch.tensor(source_batch)

            topo_error = compute_topological_error(source_batch, source_to_target_batch)
            total_topo_loss_source_projected += topo_error.item()

        Topo_evaluation_results.append(["GAN", source_tech_name, target_tech_name, path_suffix, epoch,
                                        total_topo_loss_source_projected])

    # Save evaluation results
    Topo_evaluation_results = pd.DataFrame(Topo_evaluation_results, columns=Topo_Evaluation_Columns)
    Topo_evaluation_results.to_csv(
        path_or_buf="{}/Evaluation Results/Topological Assessment {}.csv".format(ensemble_path_prefix, path_suffix),
        header=True, index=False)

    for epoch in epochs:

        print("#########################################################")
        print("#########################################################")
        print("")
        print("Evaluating for epoch: ", epoch)
        path = "{}/Models/{}_to_{}_Generator_{}_{}.pt".format(ensemble_path_prefix,
                                                              source_tech_name,
                                                              target_tech_name,
                                                              epoch,
                                                              path_suffix)
        print("Model path: ", path)

        # Project source into target space
        model = torch.load(path)  # Load trained model
        source_to_target = torch.empty(0)

        source_tech = torch.tensor(source_tech).float()
        for tensor in source_tech:
            projected_tensor = model(tensor).reshape(1, 8)
            source_to_target = torch.cat((source_to_target, projected_tensor))

        source_to_target = source_to_target.detach()
        print("source data shape: ", source_to_target.shape)

        # Convert to dataframes
        source_indices = list(Annotation_Data_Source.index.values)
        source_to_target = pd.DataFrame(source_to_target, index=source_indices).astype("float")
        if epoch == 1000:
            source_aligned = source_to_target
        target_indices = list(Annotation_Data_Target.index.values)
        target_tech = pd.DataFrame(target_tech, index=target_indices).astype("float")


        # Annotate data

        source_to_target_annotated = source_to_target.merge(Annotation_Data_Source, how='inner', left_index=True,
                                                            right_index=True)
        target_tech_annotated = target_tech.merge(Annotation_Data_Target, how='inner', left_index=True,
                                                  right_index=True)

        # k_array = [1, 5, 10, 15, 20, 50, 100]
        k_array = [5]
        print("Total cells: ", len(target_tech))
        for k in k_array:
            print("############################################################")
            print("")
            print("Evaluating for k = ", k)
            print("")
            results = evaluate_first_k_neighbors(source_to_target, source_to_target_annotated, target_tech,
                                                 target_tech_annotated, k)
            print('')
            print(results)

            evaluation_results.append(["GAN", source_tech_name, target_tech_name, epoch, k,
                                       results["Percentage sub cell_type matches"],
                                       results["Percentage cell type matches"]])

    # Save evaluation results
    evaluation_results = pd.DataFrame(evaluation_results, columns=Evaluation_Columns)
    evaluation_results.to_csv(
        path_or_buf="{}/Evaluation Results/Evaluation Results_{}.csv".format(ensemble_path_prefix, path_suffix),
        header=True, index=False)

    return source_aligned








