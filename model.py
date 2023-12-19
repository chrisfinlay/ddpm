import torch
import torch.nn as nn
import torch.nn.functional as F


class Model1(nn.Module):
    def __init__(
        self,
        image_channels: int,
        nb_channels: int,
        num_blocks: int,
        cond_channels: int,
    ) -> None:
        super().__init__()
        self.noise_emb = NoiseEmbedding1(cond_channels)
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock1(nb_channels) for _ in range(num_blocks)]
        )
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        cond = self.noise_emb(c_noise)  # TODO: not used yet
        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x)
        return self.conv_out(x)


class NoiseEmbedding1(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer("weight", torch.randn(1, cond_channels // 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 1
        f = 2 * torch.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class ResidualBlock1(nn.Module):
    def __init__(self, nb_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(nb_channels)
        self.conv1 = nn.Conv2d(
            nb_channels, nb_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(
            nb_channels, nb_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self.norm1(x)))
        y = self.conv2(F.relu(self.norm2(y)))
        return x + y


###############################################################################


class Model2(nn.Module):
    def __init__(
        self,
        image_channels: int,
        nb_channels: int,
        num_blocks: int,
        cond_channels: int,
    ) -> None:
        super().__init__()
        self.noise_emb = NoiseEmbedding1(cond_channels)
        self.noise_fc1 = nn.Linear(cond_channels, nb_channels)
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock1(nb_channels) for _ in range(num_blocks)]
        )
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        cond = self.noise_emb(c_noise)  # TODO: not used yet
        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x) + F.relu(self.noise_fc1(cond)).unsqueeze(2).unsqueeze(2)
        return self.conv_out(x)


class Model3(nn.Module):
    def __init__(
        self,
        image_channels: int,
        nb_channels: int,
        num_blocks: int,
        cond_channels: int,
    ) -> None:
        super().__init__()
        self.noise_emb = NoiseEmbedding1(cond_channels)
        self.noise_fc1 = nn.Linear(cond_channels, 2 * nb_channels)
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock2(nb_channels) for _ in range(num_blocks)]
        )
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        cond = self.noise_emb(c_noise)  # TODO: not used yet
        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x, self.noise_fc1(cond).unsqueeze(2).unsqueeze(2))
        return self.conv_out(x)


class ResidualBlock2(nn.Module):
    def __init__(self, nb_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(nb_channels, affine=False)
        self.conv1 = nn.Conv2d(
            nb_channels, nb_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.BatchNorm2d(nb_channels, affine=False)
        self.conv2 = nn.Conv2d(
            nb_channels, nb_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor, noise_emb: torch.Tensor) -> torch.Tensor:
        nb_channels = self.norm1.num_features
        y = self.conv1(
            F.relu(
                self.norm1(x) * noise_emb[:, :nb_channels] + noise_emb[:, nb_channels:]
            )
        )
        y = self.conv2(
            F.relu(
                self.norm2(y) * noise_emb[:, :nb_channels] + noise_emb[:, nb_channels:]
            )
        )
        return x + y
