# Import modules
from copy import deepcopy
from sklearn.model_selection import train_test_split
from src.models.approx_based import TopologicallyRegularizedAutoencoder
import matplotlib.pyplot as plt
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device ="cpu"

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import random
from Manifold_Alignment_GAN import GeneratorNet, DiscriminatorNet, train
from src.models.Assess_Topology import compute_topological_error
from src.models.submodules import VAE_PBMC, AE_PBMC
import os

def prepare_manifold_data(Manifold_Data, batch_size):
    all_samples = list(Manifold_Data.index.values)
    Manifold_Data = Manifold_Data.to_numpy()
    Manifold_Data = Manifold_Data.astype(float)
    Manifold_Data_tensor = torch.tensor(Manifold_Data).float() # Convert to tensor

    AE_train, AE_val = train_test_split(Manifold_Data_tensor,test_size=0.20, random_state = 42)

    dataloaders = {'train': DataLoader(AE_train, batch_size=batch_size, shuffle=True, num_workers=0),
                    'val': DataLoader(AE_val, batch_size=batch_size, shuffle=True, num_workers=0)
                    }

    print("Train data: ", AE_train.shape)
    print("Val data: ", AE_val.shape)

    return Manifold_Data, all_samples, dataloaders

def plot_training(x, train_losses, val_losses):
    plt.plot(x, train_losses, label="Training loss")
    plt.plot(x, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def train_AE(n_epochs, my_model, learning_rate, weight_decay,
          early_stop_threshold, _rnd, grad_threshold, dataloaders):

    model = my_model

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate,
        weight_decay=weight_decay)

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
            loss = model(data.float())

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
            loss = model(data.float())
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

def get_TopoAE_Embeddings(Manifold_Data, batch_size, autoencoder_model, AE_arch, topology_regulariser_coefficient, initial_LR):
    Manifold_Data, all_samples, dataloaders = prepare_manifold_data(Manifold_Data, batch_size)

    my_model = TopologicallyRegularizedAutoencoder(lam=topology_regulariser_coefficient,
                                                   autoencoder_model=autoencoder_model,
                                                   ae_kwargs={"arch": AE_arch},
                                                   toposig_kwargs={"match_edges": "symmetric"})
    trained_model = train_AE(n_epochs=200, my_model=my_model, learning_rate=initial_LR,
                                 weight_decay=1e-05, early_stop_threshold=50, _rnd=42,
                                 grad_threshold=50, dataloaders=dataloaders)

    data_loader = DataLoader(Manifold_Data, batch_size=1)
    latent_data = []
    latent_data = torch.empty(0)
    for index, original_sample in enumerate(data_loader):
        latent_sample = trained_model.encode(original_sample.float())
        latent_data = torch.cat((latent_data, latent_sample))
    latent_data = latent_data.detach()

    latent_data = latent_data.numpy()
    latent_data = pd.DataFrame(latent_data, index=all_samples)

    return latent_data

def get_AE_Embeddings(Manifold_Data, batch_size, model_type, AE_arch, initial_LR):
    Manifold_Data, all_samples, dataloaders = prepare_manifold_data(Manifold_Data, batch_size)

    if model_type == "Variational":
        my_model = VAE_PBMC(arch = AE_arch)
    elif model_type == "regular":
        my_model = AE_PBMC(arch = AE_arch)

    trained_model = train_AE(n_epochs=200, my_model=my_model, learning_rate=initial_LR,
                                 weight_decay=1e-05, early_stop_threshold=50, _rnd=42,
                                 grad_threshold=50, dataloaders=dataloaders)

    data_loader = DataLoader(Manifold_Data, batch_size=1)
    latent_data = []
    latent_data = torch.empty(0)
    for index, original_sample in enumerate(data_loader):
        if model_type == "Variational":
            latent_sample = trained_model.get_z(original_sample.float())
        elif model_type == "regular":
            latent_sample = trained_model.encode(original_sample.float())
        latent_data = torch.cat((latent_data, latent_sample))
    latent_data = latent_data.detach()

    latent_data = latent_data.numpy()
    latent_data = pd.DataFrame(latent_data, index=all_samples)

    return latent_data

def run_scTopoGAN(source_tech, target_tech, source_tech_name, target_tech_name, batch_size, 
                  topology_batch_size, total_epochs, num_iterations, checkpoint_epoch, 
                  g_learning_rate, d_learning_rate, path_prefix):
    epochs = [500, 600, 700, 800, 900, 1000]
    core_suffix = "TopoGAN_Generation01"
    isExist = os.path.exists(path_prefix)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path_prefix)
        os.makedirs("{}/Models".format(path_prefix))
        os.makedirs("{}/Evaluation Results".format(path_prefix))

    source_indices = list(source_tech.index.values)
    latent_dimensions = source_tech.shape[1]
    source_tech = source_tech.to_numpy()
    source_tech = source_tech.astype(float)
    print("Source Technology: ", np.shape(source_tech))
    target_tech = target_tech.to_numpy()
    target_tech = target_tech.astype(float)
    print("Target Technology: ", np.shape(target_tech))

    print("=======================================================================")
    print("Training first generation")
    print("=======================================================================")
    techs = [source_tech_name, target_tech_name]

    aggregate_topo_error = []

    for random_seed in range(num_iterations):
        # Set random seed for model uniqueness and reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        #tf.random.set_seed(random_seed)
        path_suffix = "{}_MODEL_{}".format(core_suffix, str(random_seed))
        
        generator = GeneratorNet(input_dim=latent_dimensions, output_dim=latent_dimensions)
        discriminator = DiscriminatorNet(input_dim=latent_dimensions)
        
        generator = generator.to("cuda:0")
        discriminator = discriminator.to("cuda:0")
        
        generator.train()
        discriminator.train()
    
        trained_generator = train(generator, discriminator, batch_size, 
                                source_tech, target_tech, total_epochs, g_learning_rate,
                                d_learning_rate, checkpoint_epoch, techs, path_prefix, path_suffix)

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
            model = torch.load(path, map_location="cpu")  # Load trained model
            model.eval()
            source_to_target = torch.empty(0)

            source_tech = torch.tensor(source_tech).float()
            data_loader = DataLoader(source_tech, batch_size=1)
            for index, original_sample in enumerate(data_loader):
                projected_tensor = model(original_sample.float()).reshape(1, latent_dimensions)
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
                source_batch = data[0:first_dim,0:latent_dimensions]
                source_to_target_batch = data[0:first_dim,latent_dimensions:(latent_dimensions*2)]
                # source_batch = tf.slice(data, [0, 0], [first_dim, latent_dimensions])
                # source_to_target_batch = tf.slice(data, [0, latent_dimensions], [first_dim, latent_dimensions])

                # Convert source and target from eager tensor to native tensor
                source_to_target_batch = source_to_target_batch.numpy()
                source_to_target_batch = torch.tensor(source_to_target_batch)
                source_batch = source_batch.numpy()
                source_batch = torch.tensor(source_batch)

                topo_error = compute_topological_error(source_batch, source_to_target_batch)
                total_topo_loss_source_projected += topo_error.item()

            evaluation_results.append(["GAN", source_tech_name, target_tech_name, path_suffix, epoch,
                                       total_topo_loss_source_projected])
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
    epoch_01 = 1000
    random_seed = 1
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    #tf.random.set_seed(random_seed)

    model_01_name = "{}_to_{}_Generator_{}_{}_MODEL_{}".format(source_tech_name, target_tech_name, epoch_01, core_suffix, model_01)

    path_01 = "{}/Models/{}.pt".format(path_prefix, model_01_name)

    # Load model
    generator_1 = torch.load(path_01, map_location="cuda:0")

    discriminator = DiscriminatorNet(input_dim=latent_dimensions)
    
    discriminator = discriminator.to("cuda:0")
    
    generator_1.train()
    discriminator.train()
    techs = [source_tech_name, target_tech_name]
    path_suffix = "Generation02_{}_{}".format(model_01, str(random_seed))

    print("=======================================================================")
    print("Training second generation")
    print("=======================================================================")

    trained_generator = train(generator_1, discriminator, batch_size, 
                                source_tech, target_tech, total_epochs, g_learning_rate,
                                d_learning_rate, checkpoint_epoch, techs, path_prefix, path_suffix)
    
    trained_generator.to("cpu")
    trained_generator.eval()
    
    source_aligned = torch.empty(0)
    source_tech = torch.tensor(source_tech).float()
    data_loader = DataLoader(source_tech, batch_size=1)
    for index, original_sample in enumerate(data_loader):
        projected_tensor = trained_generator(original_sample.float()).reshape(1, latent_dimensions)
        source_aligned = torch.cat((source_aligned, projected_tensor))
    
    source_aligned = source_aligned.detach()
    source_aligned = pd.DataFrame(source_aligned, index=source_indices).astype("float")
    
    return source_aligned

