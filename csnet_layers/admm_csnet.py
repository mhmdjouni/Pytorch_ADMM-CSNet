import torch.nn as nn

from csnet_layers.auxiliary import (
    ConvolutionLayer1,
    NonlinearLayer,
    ConvolutionLayer2,
)
from csnet_layers.multiplier import MultipleOriginalLayer, MultipleUpdateLayer
from csnet_layers.reconstruction import (
    ReconstructionOriginalLayer,
    ReconstructionUpdateLayer,
    ReconstructionFinalLayer,
)
from utils.fftc import *
import torch


class ADMMCSNetLayer(nn.Module):
    def __init__(
        self,
        mask,
        in_channels: int = 1,
        out_channels: int = 128,
        kernel_size: int = 5,
    ):
        """
        Args:

        """
        super(ADMMCSNetLayer, self).__init__()

        self.rho = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.mask = mask

        self.re_org_layer = ReconstructionOriginalLayer(self.rho, self.mask)
        self.conv1_layer = ConvolutionLayer1(
            in_channels, out_channels, kernel_size
        )
        self.nonlinear_layer = NonlinearLayer()
        self.conv2_layer = ConvolutionLayer2(
            out_channels, in_channels, kernel_size
        )
        self.min_layer = MinusLayer()
        self.multiple_org_layer = MultipleOriginalLayer(self.gamma)
        self.re_update_layer = ReconstructionUpdateLayer(self.rho, self.mask)
        self.add_layer = AdditionalLayer()
        self.multiple_update_layer = MultipleUpdateLayer(self.gamma)
        self.re_final_layer = ReconstructionFinalLayer(self.rho, self.mask)

        layers = []

        layers.append(self.re_org_layer)
        layers.append(self.conv1_layer)
        layers.append(self.nonlinear_layer)
        layers.append(self.conv2_layer)
        layers.append(self.min_layer)
        layers.append(self.multiple_org_layer)

        for i in range(8):
            layers.append(self.re_update_layer)
            layers.append(self.add_layer)
            layers.append(self.conv1_layer)
            layers.append(self.nonlinear_layer)
            layers.append(self.conv2_layer)
            layers.append(self.min_layer)
            layers.append(self.multiple_update_layer)

        layers.append(self.re_update_layer)
        layers.append(self.add_layer)
        layers.append(self.conv1_layer)
        layers.append(self.nonlinear_layer)
        layers.append(self.conv2_layer)
        layers.append(self.min_layer)
        layers.append(self.multiple_update_layer)

        layers.append(self.re_final_layer)

        self.cs_net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1_layer.conv.weight = torch.nn.init.normal_(
            self.conv1_layer.conv.weight, mean=0, std=1
        )
        self.conv2_layer.conv.weight = torch.nn.init.normal_(
            self.conv2_layer.conv.weight, mean=0, std=1
        )
        self.conv1_layer.conv.weight.data = (
            self.conv1_layer.conv.weight.data * 0.025
        )
        self.conv2_layer.conv.weight.data = (
            self.conv2_layer.conv.weight.data * 0.025
        )

    def forward(self, x):
        y = torch.mul(x, self.mask)
        x = self.cs_net(y)
        x = torch.fft.ifft2(y + (1 - self.mask) * torch.fft.fft2(x))
        return x


# minus layer
class MinusLayer(nn.Module):
    def __init__(self):
        super(MinusLayer, self).__init__()

    def forward(self, x):
        minus_input = x["conv1_input"]
        conv2_output = x["conv2_output"]
        output = torch.sub(minus_input, conv2_output)
        x["minus_output"] = output
        return x


# additional layer
class AdditionalLayer(nn.Module):
    def __init__(self):
        super(AdditionalLayer, self).__init__()

    def forward(self, x):
        mid_output = x["re_mid_output"]
        multi_output = x["multi_output"]
        output = torch.add(mid_output, multi_output)
        x["conv1_input"] = output
        return x
