# training function for ddpm
import numpy as np
import torch
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

def train(model, dataloader, optimizer, epochs, device, sampling_dir, return_loss=True, save_model = True):
    """Train a model on a dataset."""
    losses = []
    n_epoch = epochs
    
    for i in range(n_epoch):
        model.train()
        pbar = tqdm(dataloader)  # Wrap our loop with a visual progress bar
        for x, _ in pbar:
            optimizer.zero_grad()

            loss = model(x)

            loss.backward()
            # ^Technically should be `accelerator.backward(loss)` but not necessary for local training

            losses.append(loss.item())
            avg_loss = np.average(losses[min(len(losses)-100, 0):])
            pbar.set_description(f"epoch {i+1}/{n_epoch}, loss: {avg_loss:.3f}")

            optimizer.step()

        model.eval()
        with torch.no_grad():
            xh = model.sample(16, (1, 28, 28), device)  # Can get device explicitly with `accelerator.device`
            grid = make_grid(xh, nrow=4)

            # Save samples to `./contents` directory
            save_image(grid, sampling_dir + f"/ddpm_sample_{i:04d}.png")

            # save model
            if save_model:
                torch.save(model.state_dict(), f"./ddpm_mnist.pth")
    if return_loss:
        return losses
