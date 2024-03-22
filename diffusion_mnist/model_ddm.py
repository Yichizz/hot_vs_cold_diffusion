"""!@file model_ddm.py
@brief Model architecture for the Defading Diffusion Model (DDM) for MNIST dataset.

@details Defading Diffusion Model (DDM) is a special case of Generalized Diffusion Model (GDM) in paper by Bansal et al., (2022).
The image is degraded by progressively greying-out pixels over time (Inpainting). We use a 2D Gaussian distribution with random center and constant variance to create masks.
This file contains the scheduler, CNN architecture and loss function for DDM architecture.
@author Yichi Zhang (yz870) on 20/03/2024
"""

import typing
from typing import Tuple
from scipy.stats import multivariate_normal
import numpy as np
import torch
import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure


def mask_t(variance: float = 1, T: int = 10, input_shape: Tuple[int, int] = (28, 28)) -> np.ndarray:
    """!@brief Create a mask for time step T.

    @param variance: variance of the 2D Gaussian distribution.
    @param T: time step.
    @param input_shape: input image shape.

    @return mask_t: mask for time step T.
    """
    # create a mask for time step T
    mask_t = np.ones(input_shape)
    x = np.arange(0, input_shape[0])
    y = np.arange(0, input_shape[1])
    xx, yy = np.meshgrid(x, y)
    for t in range(T):
        # create a 2D gaussian distribution
        mean = np.random.rand(2) * input_shape[0]
        cov = np.eye(2) * variance # identity matrix
        gaussian = multivariate_normal(mean=mean, cov=cov)

        # evaluate the gaussian at each pixel
        z_i = gaussian.pdf(np.dstack((xx, yy)))
        # normalize such that the peak has value 1
        z_i = (z_i - z_i.min()) / (z_i.max() - z_i.min())
        # z_i then has center 1 (white) and edge 0 (black)
        # but we want the center to be grey (0.5) and edge 1 (white)
        # z_i = 0.5 + (1 - z_i) * 0.5
        mask_t *= (1 - z_i)
    return mask_t

def ddm_schedules(variance: float = 1, T: int = 10, input_shape: Tuple[int, int] = (28, 28)) -> dict:
    """!@brief Create degradation for all time steps.

    @param variance: variance of the 2D Gaussian distribution.
    @param T: number of time steps.
    @param input_shape: input image shape.

    @return masks: masks for all time steps.
    """
    # create a tensor of shape (T, input_shape[0], input_shape[1])
    masks = torch.zeros((T, input_shape[0], input_shape[1]))
    for t in range(T):
        mask = mask_t(T=t)
        masks[t] = torch.from_numpy(mask)

    return {"mask_t": masks}

# we use the same CNN architecture as the original ddpm model
class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        expected_shape,
        act=nn.GELU,
        kernel_size=7,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.LayerNorm((out_channels, *expected_shape)), 
            act()
        )

    def forward(self, x):
        return self.net(x)
    
class CNN(nn.Module):
    def __init__(
        self,
        in_channels,
        expected_shape=(28, 28),
        n_hidden=(64, 128, 64),
        kernel_size=7,
        last_kernel_size=3,
        time_embeddings=16, 
        act=nn.GELU,
    ) -> None:
        super().__init__()
        last = in_channels

        self.blocks = nn.ModuleList()
        for hidden in n_hidden:
            self.blocks.append(
                CNNBlock(
                    last,
                    hidden,
                    expected_shape=expected_shape,
                    kernel_size=kernel_size,
                    act=act,
                )
            )
            last = hidden

        # The final layer, we use a regular Conv2d to get the
        # correct scale and shape (and avoid applying the activation)
        self.blocks.append(
            nn.Conv2d(
                last,
                in_channels,
                last_kernel_size,
                padding=last_kernel_size // 2,
            )
        )

        ## This part is literally just to put the single scalar "t" into the CNN
        ## in a nice, high-dimensional way:
        self.time_embed = nn.Sequential(
            nn.Linear(time_embeddings * 2, 128), act(),
            nn.Linear(128, 128), act(),
            nn.Linear(128, 128), act(),
            nn.Linear(128, n_hidden[0]),
        )
        frequencies = torch.tensor(
            [0] + [2 * np.pi * 1.5**i for i in range(time_embeddings - 1)]
        ) # 1.5 is a hyperparameter means that the frequency of the time encoding increases by 1.5x each time
        self.register_buffer("frequencies", frequencies)

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        phases = torch.concat(
            (
                torch.sin(t[:, None] * self.frequencies[None, :]), # has shape (batch, time_embeddings)
                torch.cos(t[:, None] * self.frequencies[None, :]) - 1, # has shape (batch, time_embeddings)
            ),
            dim=1,
        ) # has shape (batch, time_embeddings * 2)

        return self.time_embed(phases)[:, :, None, None] # has shape (batch, n_hidden[0], 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Shapes of input:
        #    x: (batch, chan, height, width)
        #    t: (batch,)

        embed = self.blocks[0](x)
        # ^ (batch, n_hidden[0], height, width)

        # Add information about time along the diffusion process
        #  (Providing this information by superimposing in latent space)
        embed += self.time_encoding(t) 
        #         ^ (batch, n_hidden[0], 1, 1) - thus, broadcasting
        #           to the entire spatial domain

        for block in self.blocks[1:]:
            embed = block(embed)

        return embed
    
# now, we define the DDM class
class DDM(nn.Module):
    """!@brief Defading Diffusion Model (DDM) for MNIST dataset.

    @details Defading Diffusion Model (DDM) is a special case of Generalized Diffusion Model (GDM) in paper by Bansal et al., (2022).
    The image is degraded by progressively greying-out pixels over time (Inpainting). We use a 2D Gaussian distribution with random center and constant variance to create masks.
    This class describes the DDM model architecture and loss function.

    @param gt: ground truth model.
    @param variance_schedular: variance of the 2D Gaussian distribution.
    @param n_T: number of time steps.
    @param expected_shape: input image shape.
    @param criterion: loss function for the model.

    @func __init__: Initialize the DDM model.
    @func degrade: Degrade the input image.
    @func forward: Forward pass of the DDM model.
    @func sample: Generate samples from the DDM model.

    @return DDM: DDM model.
    """
    def __init__(
        self,
        gt,
        variance_schedular: float,
        n_T: int,
        expected_shape: Tuple[int, int],
        criterion: nn.Module = nn.L1Loss(),
    ) -> None:
        super().__init__()

        self.gt = gt

        noise_schedule = ddm_schedules(variance=variance_schedular, T = n_T, input_shape=expected_shape)

        # `register_buffer` will track these tensors for device placement, but
        # not store them as model parameters. This is useful for constants.
        self.register_buffer("mask_t", noise_schedule["mask_t"])
        self.mask_t

        self.n_T = n_T
        self.criterion = criterion
        self.ssim = StructuralSimilarityIndexMeasure(kernel_size=3)
        self.psnr = PeakSignalNoiseRatio()

    def degrade(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # now, our z_t is the input image x, multiplied by the mask at time t
        masks = self.mask_t[t-1].unsqueeze(1).to(x.device) # (batch, 1, 28, 28)
        z_t = x * masks
        return z_t
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # random time step with shape (batch,)
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)           
        z_t = self.degrade(x, t)
        # return all the masks at all random time steps
        noise = self.mask_t[t-1].unsqueeze(1).to(x.device)
        
        # we train the model to find the noise (inpainting)
        pred_noise = self.gt(z_t, t / self.n_T)
        loss = self.criterion(pred_noise, noise) 
        ssim_val = self.ssim(pred_noise, noise)
        psnr_val = self.psnr(pred_noise, noise)

        return loss, ssim_val, psnr_val

    def sample(self, n_sample: int, size, original_sample, device) -> torch.Tensor:
        """Algorithm 2 in Bansal et al., (2022)"""

        # we need degraded samples to sample
        # we degrade the original sample with n_T masks
        assert original_sample.shape[0] >= n_sample, "original_sample must have at least n_sample samples"
        assert original_sample.shape[1:] == size, "original_sample must have the same size as size"
        z = original_sample[:n_sample, :] # z has shape (n_sample, *size)#
        z_t = z * self.mask_t[self.n_T-1].to(device)

        _one = torch.ones(n_sample, device=device)

        for t in range(self.n_T, 0, -1): 
            # since the , we need to use t-1 and t-2
            # the restoration is z_t - pred_noise
            z_0 = z_t - self.gt(z_t, (t/self.n_T) * _one)
            z_t = z_t - z_0 * self.mask_t[t-1].to(device) + z_0 * self.mask_t[t-2].to(device)

        return z_t