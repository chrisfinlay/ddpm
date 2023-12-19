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
    def __init__(self, nb_channels: int, init_zero: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(nb_channels)
        self.conv1 = nn.Conv2d(
            nb_channels, nb_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(
            nb_channels, nb_channels, kernel_size=3, stride=1, padding=1
        )
        if init_zero:
            torch.nn.init.zeros_(self.conv2.weight)

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
            x = block(x) + F.relu(self.noise_fc1(cond)).unsqueeze(2).unsqueeze(3)
        return self.conv_out(x)


class Model22(nn.Module):
    def __init__(
        self,
        image_size: int,
        image_channels: int,
        nb_channels: int,
        num_blocks: int,
        cond_channels: int,
        init_zero: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.noise_emb = NoiseEmbedding1(cond_channels)
        self.noise_fc1 = nn.Linear(cond_channels, image_size**2)
        self.conv_in = nn.Conv2d(
            image_channels + 1, nb_channels, kernel_size=3, padding=1
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock1(nb_channels, init_zero) for _ in range(num_blocks)]
        )
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)
        if init_zero:
            torch.nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        cond = self.noise_fc1(self.noise_emb(c_noise))  # TODO: not used yet
        x = self.conv_in(
            torch.concat(
                [
                    noisy_input,
                    cond.reshape(-1, 1, self.image_size, self.image_size)
                    * torch.ones(*noisy_input.shape, device=noisy_input.device),
                ],
                axis=1,
            )
        )
        for block in self.blocks:
            x = block(x)
        return self.conv_out(x)


class Model3(nn.Module):
    def __init__(
        self,
        image_channels: int,
        nb_channels: int,
        num_blocks: int,
        cond_channels: int,
        init_zero: bool = False,
    ) -> None:
        super().__init__()
        self.noise_emb = NoiseEmbedding1(cond_channels)
        self.noise_fc1 = nn.Linear(cond_channels, 2 * nb_channels)
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock2(nb_channels, init_zero) for _ in range(num_blocks)]
        )
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)
        if init_zero:
            torch.nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        cond = self.noise_emb(c_noise)  # TODO: not used yet
        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x, self.noise_fc1(cond).unsqueeze(2).unsqueeze(2))
        return self.conv_out(x)


class ResidualBlock2(nn.Module):
    def __init__(self, nb_channels: int, init_zero: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(nb_channels, affine=False)
        self.conv1 = nn.Conv2d(
            nb_channels, nb_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.BatchNorm2d(nb_channels, affine=False)
        self.conv2 = nn.Conv2d(
            nb_channels, nb_channels, kernel_size=3, stride=1, padding=1
        )
        if init_zero:
            torch.nn.init.zeros_(self.conv2.weight)

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


class NoiseEmbedding2(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        assert image_size % 2 == 0
        self.register_buffer("weight", torch.randn(1, image_size**2 // 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 1
        f = 2 * torch.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class Unet1(nn.Module):
    def __init__(self, image_size: int, channels: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.noise_emb = NoiseEmbedding2(image_size)
        self.noise_fc = nn.Linear(image_size**2, image_size**2)
        self.norm1 = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(2, channels, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels * 2, kernel_size=3, stride=2, padding=1
        )
        self.norm3 = nn.BatchNorm2d(channels * 2)
        self.conv3 = nn.Conv2d(
            channels * 2, channels * 4, kernel_size=3, stride=2, padding=1
        )
        self.norm4 = nn.BatchNorm2d(channels * 4)
        self.fc1 = nn.Linear(
            channels * 4 * (image_size // 8) ** 2, channels * 4 * (image_size // 8) ** 2
        )
        self.inorm1 = nn.BatchNorm2d(channels * 4)
        self.iconv1 = nn.ConvTranspose2d(
            channels * 4, channels * 2, kernel_size=2, stride=2
        )
        self.inorm2 = nn.BatchNorm2d(channels * 2)
        self.iconv2 = nn.ConvTranspose2d(
            channels * 2, channels, kernel_size=2, stride=2
        )
        self.inorm3 = nn.BatchNorm2d(channels)
        self.iconv3 = nn.ConvTranspose2d(channels, 2, kernel_size=2, stride=2)
        self.inorm4 = nn.BatchNorm2d(2)
        self.conv_out = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, cond):
        cond = self.noise_fc(self.noise_emb(cond))
        x0 = torch.concat(
            [
                x,
                cond.reshape((-1, 1, self.image_size, self.image_size))
                * torch.ones(*x.shape, device=x.device),
            ],
            axis=1,
        )
        x1 = self.conv1(F.relu(self.norm1(x0)))
        x2 = self.conv2(F.relu(self.norm2(x1)))
        x3 = self.conv3(F.relu(self.norm3(x2)))
        x4 = F.relu(
            self.fc1(
                F.relu(self.norm4(x3)).reshape(
                    (-1, self.channels * 4 * (self.image_size // 8) ** 2)
                )
            )
        )
        x4 = self.iconv1(
            F.relu(
                self.inorm1(
                    x4.reshape(
                        (
                            -1,
                            self.channels * 4,
                            self.image_size // 8,
                            self.image_size // 8,
                        )
                    )
                )
            )
            + x3
        )
        x4 = self.iconv2(F.relu(self.inorm2(x4)) + x2)
        x4 = self.iconv3(F.relu(self.inorm3(x4)) + x1)
        x4 = self.conv_out(F.relu(self.inorm4(x4)))
        return x4
