import matplotlib.pyplot as plt
from data import load_dataset_and_make_dataloaders
import torch
from tqdm import tqdm


def sample_sigma(
    n: int,
    loc: float = -1.2,
    scale: float = 1.2,
    sigma_min: float = 2e-3,
    sigma_max: float = 80,
    device: str = "cpu",
) -> torch.Tensor:
    return (
        (torch.randn(n, device=device) * scale + loc).exp().clip(sigma_min, sigma_max)
    )


def build_sigma_schedule(
    steps: int, rho: int = 7, sigma_min: float = 2e-3, sigma_max: float = 80
) -> torch.Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (
        max_inv_rho + torch.linspace(0, 1, steps) * (min_inv_rho - max_inv_rho)
    ) ** rho
    return sigmas


def generate_images(
    n_imgs: int,
    img_shape: tuple,
    sigmas: list,
    D: torch.nn.Module,
    include_steps: bool,
    device: str = "cpu",
) -> torch.Tensor:
    x = (
        torch.randn(n_imgs, *img_shape, device=device) * sigmas[0]
    )  # Initialize with pure gaussian noise ~ N(0, sigmas[0])

    if include_steps:
        denoised_steps = torch.empty(len(sigmas) + 1, *x.shape, device=device)
        denoised_steps[0] = x

    for i, sigma in enumerate(sigmas):
        with torch.no_grad():
            x_denoised = D(x, torch.Tensor([sigma]).to(device))
            # Where D(x, sigma) = cskip(sigma) * x + cout(sigma) * F(cin(sigma) * x, cnoise(sigma))
            # and F(.,.) is your neural network

        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        d = (x - x_denoised) / sigma

        x = x + d * (sigma_next - sigma)  # Perform one step of Euler's method

        if include_steps:
            denoised_steps[i + 1] = x

    if include_steps:
        return denoised_steps
    else:
        return x


def noising_process(imgs: torch.Tensor, sigmas: torch.Tensor):
    img_dims = imgs.ndim * [
        None,
    ]
    n_sigma = len(sigmas)

    noisy_imgs = (
        torch.randn(n_sigma, *imgs.shape) * sigmas[:, *img_dims] + imgs[None, ...]
    )
    return noisy_imgs


def plot_imgs(
    imgs: torch.Tensor, sigmas: torch.Tensor, plot_name: str, plot_title: str
):
    n_sigma, n_img = imgs.shape[:2]

    fig, ax = plt.subplots(n_img, n_sigma, figsize=(1.5 * n_sigma, 1.5 * n_img + 1))
    fig.suptitle(plot_title, fontsize=18)
    for i in range(n_img):
        for j in range(n_sigma):
            if i == 0:
                ax[i, j].set_title(r"$\sigma$ = " + f"{sigmas[-j-1]: .2f}")
            ax[i, j].imshow(imgs[-j - 1, i].squeeze().cpu(), cmap="Greys_r")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    plt.savefig(f"images/{plot_name}.png")


def denoiser(F, c_in, c_out, c_skip, c_noise):
    def D(x, sigma):
        return c_skip(sigma[:, None, None, None]) * x + c_out(
            sigma[:, None, None, None]
        ) * F(c_in(sigma[:, None, None, None]) * x, c_noise(sigma))

    return D


def train_ddpm(
    model,
    c_funcs,
    N,
    nb_epochs,
    lr,
    batch_size,
    optim="Adam",
    dataset_name="FashionMNIST",
    root_dir="data",
):
    gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if gpu else "cpu")

    F = model.to(device).forward
    c_in, c_out, c_skip, c_noise = c_funcs

    def D(x, sigma):
        return c_skip(sigma[:, None, None, None]) * x + c_out(
            sigma[:, None, None, None]
        ) * F(c_in(sigma[:, None, None, None]) * x, c_noise(sigma))

    criterion = torch.nn.MSELoss().to(device)
    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr)
    elif optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr)
    else:
        print("Choose 'optim' from 'Adam' or 'SGD'. ")
        return D, [], []

    train_losses = []
    val_losses = []

    dl, info = load_dataset_and_make_dataloaders(
        dataset_name=dataset_name,
        root_dir=root_dir,  # choose the directory to store the data
        batch_size=batch_size,
        num_workers=0,  # you can use more workers if you see the GPU is waiting for the batches
        pin_memory=gpu,  # use pin memory if you're planning to move the data to GPU
    )

    for _ in tqdm(range(nb_epochs)):
        train_loss = []
        for y, _ in dl.train:
            sigma = sample_sigma(y.shape[0])[:, None, None, None].to(device)
            y = y.to(device)
            x = N(y, sigma)

            output = F(c_in(sigma) * x, c_noise(sigma[:, 0, 0, 0]))
            target = (y - c_skip(sigma) * x) / c_out(sigma)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_losses.append(torch.Tensor(train_loss).mean())

        val_loss = []
        for y, _ in dl.valid:
            sigma = sample_sigma(y.shape[0])[:, None, None, None].to(device)
            y = y.to(device)
            x = N(y, sigma)

            output = F(c_in(sigma) * x, c_noise(sigma[:, 0, 0, 0]))
            target = (y - c_skip(sigma) * x) / c_out(sigma)

            val_loss.append(criterion(output, target).item())

        val_losses.append(torch.Tensor(val_loss).mean())

    return D, train_losses, val_losses
