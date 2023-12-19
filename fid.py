from pytorch_fid.inception import InceptionV3
import torch

gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if gpu else "cpu")


# wrapper class as feature_extractor taken from https://pytorch.org/ignite/generated/ignite.metrics.FID.html
class WrapperInceptionV3(torch.nn.Module):
    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3

    @torch.no_grad()
    def forward(self, x):
        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]
        return y


# pure pytorch implementation taken from
def calculate_frechet_distance(x, y):
    sigma_x, sigma_y = torch.cov(x.T), torch.cov(y.T)
    a = (x.mean(axis=0) - y.mean(axis=0)).square().sum()
    b = sigma_x.trace() + sigma_y.trace()
    c = torch.linalg.eigvals(sigma_x @ sigma_y + 1e-12).sqrt().real.sum(dim=-1)
    return a + b - 2 * c


def calculate_fid(y_gen, y_data):
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
