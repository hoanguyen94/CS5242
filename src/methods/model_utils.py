import inspect
import time

import torch.nn as nn
import torch
import torchvision
from torch.nn import functional as F
import argparse
import copy
from typing import Dict, List, Tuple, Optional

# Optional FLOPs (handled gracefully if not installed)
try:
    from thop import profile
    THOP_AVAILABLE = True
except Exception:
    THOP_AVAILABLE = False


# ──────────────────────────────────────────────
# ResNet-18 from Scratch
# ──────────────────────────────────────────────

class BasicResidualBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet model from scratch."""
    def __init__(self, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicResidualBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicResidualBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicResidualBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicResidualBlock, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicResidualBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# ──────────────────────────────────────────────
# ConvNeXt-Tiny from Scratch
# ──────────────────────────────────────────────

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block.
    Args:
        dim (int): Number of input channels.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

class ConvNeXt(nn.Module):
    """ ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
    """
    def __init__(self, in_chans=3, num_classes=100,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ourblock(nn.Module):
    """ ConvNeXt Block.
    Args:
        dim (int): Number of input channels.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv

        # self.dwconv = nn.Conv2d(
        #     dim, 
        #     dim, 
        #     kernel_size=3, 
        #     padding=3,      # needed to preserve spatial size
        #     dilation=3, 
        #     groups=dim      # still depthwise
        # )

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

class ournet(nn.Module):
    """ ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
    """
    def __init__(self, in_chans=3, num_classes=100,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    # nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=1),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x





class ourblock_inception(nn.Module):
    """ ConvNeXt Block.
    Args:
        dim (int): Number of input channels.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwcon1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.dwcon2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim) # depthwise conv
        self.dwcon3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim) # depthwise conv
        # self.mp = nn.MaxPool2d(kernel_size=3, pading=2, stride=1) # S = (s- k + 2p)/stride +1
        self.conv1 = nn.Conv2d(dim*4, dim, kernel_size=1, padding=0) # pointwise/1x1 convs
        # self.dwconv = nn.Conv2d(
        #     dim, 
        #     dim, 
        #     kernel_size=3, 
        #     padding=3,      # needed to preserve spatial size
        #     dilation=3, 
        #     groups=dim      # still depthwise
        # )

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x3 = self.dwconv3(x)
        x = self.conv1(torch.concat([x1, x2, x3, x], dim=1))

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

class ournetv2(nn.Module):
    """ ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
    """
    def __init__(self, in_chans=3, num_classes=100,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    # nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=1),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ourblock_inception(dim=dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ──────────────────────────────────────────────
# Model Profiling
# ──────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def try_flops(
    model: nn.Module,
    img_size: int = 224,
    device: torch.device = torch.device("cpu"),
) -> Optional[float]:
    """Returns GFLOPs, or None if thop is not installed."""
    if not THOP_AVAILABLE:
        return None
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    model_copy = copy.deepcopy(model).to(device)
    model_copy.eval()
    flops, _ = profile(model_copy, inputs=(dummy,), verbose=False)
    return flops / 1e9


# ──────────────────────────────────────────────
# Sanity Check / Profiling
# ──────────────────────────────────────────────

def run_sanity_check(backbone: str, img_size: int, device: torch.device):
    """
    Builds a model, profiles its parameters and FLOPs, and performs a
    forward pass with a random tensor to check for errors.
    """
    
    print("─" * 80)
    print("Running Sanity Check: Model Profiling")
    if backbone == "resnet18_scratch":
        model = ResNet([2, 2, 2, 2], num_classes=100)
    if backbone == "convnext_tiny_scratch":
        model = ConvNeXt(num_classes=100)
    if backbone == "ournet":
        # model = ournet(num_classes=100, depths=[2, 2, 2, 2], dims=[96, 192, 384, 768]) 
        model = ournet(num_classes=100, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384]) 
    model = model.to(device)
    model.eval()

    batch_size = 4  # Use a small batch size for the sanity check
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)

    print(f"Profiling model '{backbone}' with input size {dummy_input.shape}...")
    print(f"  - Parameters: {count_params(model) / 1e6:.2f}M")
    gflops = try_flops(model, img_size=img_size, device=device)
    print(f"  - GFLOPs: {gflops:.2f}" if gflops is not None else "  - GFLOPs: thop not installed")
    
    # --- Inference time measurement ---
    warmup_iters = 10
    timing_iters = 100
    
    with torch.no_grad():
        output = model(dummy_input)
        # Warmup runs
        for _ in range(warmup_iters):
            _ = model(dummy_input)

        # Timing runs
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(timing_iters):
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_end = time.time()

    total_time_s = t_end - t_start
    time_per_batch_ms = (total_time_s / timing_iters) * 1000
    time_per_image_ms = time_per_batch_ms / batch_size
    print(f"  - Inference time: {time_per_image_ms:.3f} ms/image ({batch_size} images/batch)")
    print(f"  - Output shape for a batch: {output.shape}")
    print("Sanity check complete. Exiting.")
    print("─" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Model utility script for sanity checks.")
    p.add_argument("--backbone",   default="resnet18_scratch",
                   choices=["convnext_tiny", "resnet18", "resnet34", "resnet50",
                            "efficientnet_b0", "efficientnet_b1", "resnet18_scratch",
                            "convnext_tiny_scratch", "ournet"])
    p.add_argument("--img_size",   type=int,   default=32)
    p.add_argument("--use_gpu",    action="store_true")
    args = p.parse_args()

    # Late import to avoid circular dependency issues if this file is imported
    dev =  torch.device("cuda") if args.use_gpu and torch.cuda.is_available() else torch.device("cpu")


    run_sanity_check(
        backbone=args.backbone,
        img_size=args.img_size,
        device=dev
    )