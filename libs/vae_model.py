from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import tensor
from torch import nn
import torch as t
from abc import abstractmethod

class BaseVAE(nn.Module):


    def __init__(self) -> None:
        super(BaseVAE, self).__init__()


    def encode(self, input: tensor) -> List[tensor]:
        raise NotImplementedError


    def decode(self, input: tensor) -> Any:
        raise NotImplementedError


    def sample(self, batch_size: int, current_device: int, **kwargs) -> tensor:
        raise RuntimeWarning()


    def generate(self, x: tensor, **kwargs) -> tensor:
        raise NotImplementedError


    @abstractmethod
    def forward(self, *inputs: tensor) -> tensor:
        pass


    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> tensor:
        pass


class VAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: tensor) -> List[tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (tensor) Input tensor to encoder [N x C x H x W]
        :return: (tensor) List of latent codes
        """
        result = self.encoder(input)
        result = t.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: tensor) -> tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (tensor) [B x D]
        :return: (tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: tensor, logvar: tensor) -> tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (tensor) [B x D]
        """
        std = t.exp(0.5 * logvar)
        eps = t.randn_like(std)
        return eps * std + mu

    def forward(self, input: tensor, **kwargs) -> List[tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = t.mean(-0.5 * t.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (tensor)
        """
        z = t.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: tensor, **kwargs) -> tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (tensor) [B x C x H x W]
        :return: (tensor) [B x C x H x W]
        """

        return self.forward(x)[0]