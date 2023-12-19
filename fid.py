from pytorch_fid.inception import InceptionV3
import torch
from torch import Tensor
from typing import Callable
from utils import generate_images
from data import load_dataset_and_make_dataloaders

gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if gpu else "cpu")


# wrapper class as feature_extractor taken from https://pytorch.org/ignite/generated/ignite.metrics.FID.html
class WrapperInceptionV3(torch.nn.Module):
    def __init__(self, fid_incv3: torch.nn.Module):
        super().__init__()
        self.fid_incv3 = fid_incv3

    @torch.no_grad()
    def forward(self, x: Tensor):
        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]
        return y


# pure pytorch implementation taken from https://www.reddit.com/r/MachineLearning/comments/12hv2u6/d_a_better_way_to_compute_the_fr%C3%A9chet_inception/
def calculate_frechet_distance(x: Tensor, y: Tensor) -> Tensor:
    sigma_x, sigma_y = torch.cov(x.T), torch.cov(y.T)
    a = (x.mean(axis=0) - y.mean(axis=0)).square().sum()
    b = sigma_x.trace() + sigma_y.trace()
    c = torch.linalg.eigvals(sigma_x @ sigma_y + 1e-12).sqrt().real.sum(dim=-1)
    return a + b - 2 * c


def calculate_fid(y_gen: Tensor, y_data: Tensor) -> Tensor:
    if y_gen.shape[1] != 3:
        y_gen = y_gen * torch.ones(1, 3, 1, 1, device=device)

    if y_data.shape[1] != 3:
        y_data = y_data * torch.ones(1, 3, 1, 1, device=device)

    # pytorch_fid model
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    wrapper_model = WrapperInceptionV3(model)
    wrapper_model.eval()

    y_gen_latent = wrapper_model(y_gen)
    y_data_latent = wrapper_model(y_data)

    return calculate_frechet_distance(y_gen_latent, y_data_latent)


def get_fid(n_samples: int, sigmas: Tensor, D: Callable, device: str = "cpu") -> Tensor:
    dl, info = load_dataset_and_make_dataloaders(
        dataset_name="FashionMNIST",
        root_dir="data",  # choose the directory to store the data
        batch_size=n_samples,
        num_workers=0,  # you can use more workers if you see the GPU is waiting for the batches
        pin_memory=gpu,  # use pin memory if you're planning to move the data to GPU
    )

    for y, label in dl.train:
        y_data = y
        break

    y_gen = generate_images(
        n_samples, y_data.shape[1:], sigmas, D, include_steps=False, device=device
    )

    return calculate_fid(y_gen, y_data.to(device))
