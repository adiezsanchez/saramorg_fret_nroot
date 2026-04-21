from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


def _number_of_features_per_level(
    init_channel_number: int, num_levels: int
) -> list[int]:
    return [init_channel_number * 2**k for k in range(num_levels)]


def _create_conv(
    in_channels,
    out_channels,
    kernel_size,
    order,
    num_groups,
    padding,
    is3d,
):
    assert "c" in order, "Conv layer MUST be present"
    assert order[0] not in "rle", "Non-linearity cannot be the first operation"

    modules = []
    for i, char in enumerate(order):
        if char == "r":
            modules.append(("relu", nn.ReLU(inplace=True)))
        elif char == "l":
            modules.append(("leaky_relu", nn.LeakyReLU(inplace=True)))
        elif char == "e":
            modules.append(("elu", nn.ELU(inplace=True)))
        elif char == "c":
            bias = not ("g" in order or "b" in order)
            conv = nn.Conv3d if is3d else nn.Conv2d
            modules.append(
                (
                    "conv",
                    conv(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=padding,
                        bias=bias,
                    ),
                )
            )
        elif char == "g":
            is_before_conv = i < order.index("c")
            num_channels = in_channels if is_before_conv else out_channels
            if num_channels < num_groups:
                num_groups = 1
            assert num_channels % num_groups == 0
            modules.append(
                (
                    "groupnorm",
                    nn.GroupNorm(
                        num_groups=num_groups, num_channels=num_channels
                    ),
                )
            )
        elif char == "b":
            bn = nn.BatchNorm3d if is3d else nn.BatchNorm2d
            is_before_conv = i < order.index("c")
            modules.append(
                ("batchnorm", bn(in_channels if is_before_conv else out_channels))
            )
        else:
            raise ValueError(f"Unsupported layer type '{char}'")
    return modules


class SingleConv(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        order="gcr",
        num_groups=8,
        padding=1,
        is3d=True,
    ):
        super().__init__()
        for name, module in _create_conv(
            in_channels,
            out_channels,
            kernel_size,
            order,
            num_groups,
            padding,
            is3d,
        ):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        encoder,
        kernel_size=3,
        order="gcr",
        num_groups=8,
        padding=1,
        is3d=True,
    ):
        super().__init__()
        if encoder:
            conv1_in = in_channels
            conv1_out = max(out_channels // 2, in_channels)
            conv2_in, conv2_out = conv1_out, out_channels
        else:
            conv1_in, conv1_out = in_channels, out_channels
            conv2_in, conv2_out = out_channels, out_channels

        self.add_module(
            "single_conv1",
            SingleConv(
                conv1_in,
                conv1_out,
                kernel_size,
                order,
                num_groups,
                padding,
                is3d,
            ),
        )
        self.add_module(
            "single_conv2",
            SingleConv(
                conv2_in,
                conv2_out,
                kernel_size,
                order,
                num_groups,
                padding,
                is3d,
            ),
        )


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        apply_pooling=True,
        pool_kernel_size=2,
        pool_type="max",
        conv_layer_order="gcr",
        num_groups=8,
        padding=1,
        is3d=True,
    ):
        super().__init__()
        if apply_pooling:
            if pool_type == "max":
                self.pooling = (
                    nn.MaxPool3d(pool_kernel_size)
                    if is3d
                    else nn.MaxPool2d(pool_kernel_size)
                )
            else:
                self.pooling = (
                    nn.AvgPool3d(pool_kernel_size)
                    if is3d
                    else nn.AvgPool2d(pool_kernel_size)
                )
        else:
            self.pooling = None

        self.basic_module = DoubleConv(
            in_channels,
            out_channels,
            encoder=True,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            padding=padding,
            is3d=is3d,
        )

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        return self.basic_module(x)


class InterpolateUpsampling(nn.Module):
    def __init__(self, mode="nearest"):
        super().__init__()
        self.upsample = partial(F.interpolate, mode=mode)

    def forward(self, encoder_features, x):
        return self.upsample(x, size=encoder_features.size()[2:])


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        conv_layer_order="gcr",
        num_groups=8,
        mode="nearest",
        padding=1,
        upsample=True,
        is3d=True,
    ):
        super().__init__()
        self.upsampling = (
            InterpolateUpsampling(mode=mode) if upsample else nn.Identity()
        )
        self.basic_module = DoubleConv(
            in_channels,
            out_channels,
            encoder=False,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            padding=padding,
            is3d=is3d,
        )

    def forward(self, encoder_features, x):
        if isinstance(self.upsampling, InterpolateUpsampling):
            x = self.upsampling(encoder_features, x)
        x = torch.cat((encoder_features, x), dim=1)
        return self.basic_module(x)


class AbstractUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        final_sigmoid: bool,
        f_maps: int | list[int] | tuple[int, ...] = 64,
        layer_order: str = "gcr",
        num_groups: int = 8,
        num_levels: int = 4,
        is_segmentation: bool = True,
        conv_kernel_size: int = 3,
        pool_kernel_size: int = 2,
        conv_padding: int = 1,
        is3d: bool = True,
    ):
        super().__init__()

        if isinstance(f_maps, int):
            f_maps = _number_of_features_per_level(f_maps, num_levels=num_levels)
        f_maps = list(f_maps)

        self.encoders = nn.ModuleList()
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(
                    in_channels,
                    out_feature_num,
                    apply_pooling=False,
                    conv_kernel_size=conv_kernel_size,
                    conv_layer_order=layer_order,
                    num_groups=num_groups,
                    padding=conv_padding,
                    is3d=is3d,
                )
            else:
                encoder = Encoder(
                    f_maps[i - 1],
                    out_feature_num,
                    apply_pooling=True,
                    pool_kernel_size=pool_kernel_size,
                    conv_kernel_size=conv_kernel_size,
                    conv_layer_order=layer_order,
                    num_groups=num_groups,
                    padding=conv_padding,
                    is3d=is3d,
                )
            self.encoders.append(encoder)

        self.decoders = nn.ModuleList()
        rev_f = list(reversed(f_maps))
        for i in range(len(rev_f) - 1):
            self.decoders.append(
                Decoder(
                    rev_f[i] + rev_f[i + 1],
                    rev_f[i + 1],
                    conv_kernel_size=conv_kernel_size,
                    conv_layer_order=layer_order,
                    num_groups=num_groups,
                    padding=conv_padding,
                    is3d=is3d,
                )
            )

        self.final_conv = (
            nn.Conv3d(f_maps[0], out_channels, 1)
            if is3d
            else nn.Conv2d(f_maps[0], out_channels, 1)
        )

        if is_segmentation:
            self.final_activation = nn.Sigmoid() if final_sigmoid else nn.Softmax(dim=1)
        else:
            self.final_activation = None

    def forward(self, x):
        enc_feats = []
        for enc in self.encoders:
            x = enc(x)
            enc_feats.insert(0, x)

        enc_feats = enc_feats[1:]
        for dec, feat in zip(self.decoders, enc_feats):
            x = dec(feat, x)

        x = self.final_conv(x)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x


class UNet3D(AbstractUNet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        final_sigmoid: bool = True,
        f_maps: int | list[int] | tuple[int, ...] = 64,
        layer_order: str = "gcr",
        num_groups: int = 8,
        num_levels: int = 4,
        is_segmentation: bool = True,
        conv_padding: int = 1,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            is3d=True,
        )


class UNet2D(AbstractUNet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        final_sigmoid: bool = True,
        f_maps: int | list[int] | tuple[int, ...] = 64,
        layer_order: str = "gcr",
        num_groups: int = 8,
        num_levels: int = 4,
        is_segmentation: bool = True,
        conv_padding: int = 1,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            is3d=False,
        )
