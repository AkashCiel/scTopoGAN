"""Wasserstein Generative Adversarial Network while projecting source into target space"""
random_seed = 1
# Import modules
import numpy as np
np.random.seed(random_seed)
import torch
torch.manual_seed(random_seed)
import random
random.seed(random_seed)

from torch import nn, optim
import tensorflow as tf
tf.random.set_seed(random_seed)
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt

# Define function to initialise weights
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.02)

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self, input_dim, output_dim):
        super(GeneratorNet, self).__init__()
        n_features = input_dim
        n_out = output_dim

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 100), nn.ReLU())

        self.hidden1 = nn.Sequential(
            nn.Linear(100, 40), nn.ReLU())

        self.hidden2 = nn.Sequential(
            nn.Linear(40, 20), nn.ReLU())

        self.out = nn.Sequential(
            nn.Linear(20, n_out))

        # Initialise all weights
        """
        self.hidden0.apply(weights_init)
        self.hidden1.apply(weights_init)
        self.hidden2.apply(weights_init)
        self.out.apply(weights_init)
        """

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

class DiscriminatorNet(torch.nn.Module):
    """
    A four hidden-layer discriminative neural network
    """

    def __init__(self, input_dim):
        super(DiscriminatorNet, self).__init__()
        n_features = input_dim
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 200),
            nn.LeakyReLU(0.2), nn.Dropout(0.3))

        self.hidden1 = nn.Sequential(
            nn.Linear(200, 100),
            nn.LeakyReLU(0.2), nn.Dropout(0.3))

        self.hidden2 = nn.Sequential(
            nn.Linear(100, 40),
            nn.LeakyReLU(0.2), nn.Dropout(0.3))

        self.hidden3 = nn.Sequential(
            nn.Linear(40, 20),
            nn.LeakyReLU(0.2), nn.Dropout(0.3))

        self.out = nn.Sequential(
            torch.nn.Linear(20, n_out))

        # Initialise all weights

        self.hidden0.apply(weights_init)
        self.hidden1.apply(weights_init)
        self.hidden2.apply(weights_init)
        self.hidden3.apply(weights_init)
        self.out.apply(weights_init)

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x

# Define functions to generate arrays of 0s or 1s
def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

# Define function to train the discriminator
def train_discriminator(optimizer, discriminator, real_data, fake_data):

    print("Discriminator")
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    prediction_fake = discriminator(fake_data)

    discriminator_error = -(torch.mean(prediction_real) - torch.mean(prediction_fake))
    discriminator_error.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Clamp discriminator weights
    for p in discriminator.parameters():
        p.data.clamp_(-0.01, 0.01)
    # Return error and predictions for real and fake inputs
    return discriminator_error, prediction_real, prediction_fake

# Define function to train the generator
def train_generator(optimizer, prediction):
    print("Generator")
    # Reset gradients
    optimizer.zero_grad()
    # Calculate error and backpropagate
    error = -torch.mean(prediction)
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

def train(generator, discriminator,
          data_loader,
          num_epochs, checkpoint_epoch,
          learning_rate, techs, path_prefix, path_suffix):
    # Define optimisers for Discriminator and Generator
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=learning_rate)
    g_optimizer = optim.RMSprop(generator.parameters(), lr=learning_rate)
    D_Losses = []
    G_Losses = []
    for epoch in range(num_epochs):
        d_loss = 0
        g_loss = 0
        print("Current Epoch: ", epoch)
        for n_batch, data in enumerate(data_loader):
            first_dim = data.shape[0]
            source = tf.slice(data, [0, 0], [first_dim, 8])
            target = tf.slice(data, [0, 8], [first_dim, 8])

            # Convert source and target from eager tensor to native tensor
            source = source.numpy()
            source = torch.tensor(source)

            target = target.numpy()
            target = torch.tensor(target)

            # 1. Train Discriminator
            real_data = target
            # Generate fake data and detach
            # (so gradients are not calculated for generator)
            fake_data = generator(source).detach()

            # Train Discriminator
            for _ in range(1):
                d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, discriminator, real_data, fake_data)
                d_loss += d_error.item()

            # 2. Train Generator
            # Generate fake data
            for _ in range(1):
                fake_data = generator(source)
                prediction = discriminator(fake_data)
                # Train G
                g_error = train_generator(g_optimizer, prediction)
                g_loss += g_error.item()

        d_loss = d_loss / n_batch
        g_loss = g_loss / n_batch

        D_Losses.append(d_loss)
        G_Losses.append(g_loss)

        x = range(epoch + 1)
        if ((epoch % checkpoint_epoch) == 0):
            intermediate_path = "{}/{}_to_{}_Generator_{}_{}_{}.pt".format(
                path_prefix, techs[0], techs[1], epoch, learning_rate, path_suffix)
            torch.save(generator, intermediate_path)
            plt.plot(x, D_Losses, label="Discriminator loss")
            plt.plot(x, G_Losses, label="Generator loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

    return generator