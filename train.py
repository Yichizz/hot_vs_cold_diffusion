# run on the command line: python train.py num_epochs 
import torch
import sys
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from ddpm_mnist import engine, model_builder, utils

# Visualize the training loss
# utils.visualize_training_loss(losses)

def main():
    NUM_EPOCHS = 2

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True) # 1000/128 = 7 batches (drop last)

    gt = model_builder.CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU)
    model = model_builder.DDPM(gt, betas=(1e-4, 0.02), n_T=1000)

    optim = torch.optim.Adam(model.parameters(), lr=2e-4)

    accelerator = Accelerator()

    # We wrap our model, optimizer, and dataloaders with `accelerator.prepare`,
    # which lets HuggingFace's Accelerate handle the device placement and gradient accumulation.
    ddpm, optim, dataloader = accelerator.prepare(model, optim, dataloader)

    # Start training with help from engine.py
    losses = engine.train(model=model,
                dataloader = dataloader,
                optimizer=optim,
                epochs=NUM_EPOCHS,
                device = accelerator.device,
                return_loss = True,
                save_model = False,
                sampling_dir = 'contents') # specify the directory to save the samples

if __name__ == "__main__":
    main()