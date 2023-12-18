import matplotlib.pyplot as plt
import torch


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
