import torch
import torch.nn as nn

import torchpwl


# convolution layer
class ConvolutionLayer1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvolutionLayer1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
            stride=1,
            dilation=1,
            bias=True,
        )

    def forward(self, x):
        conv1_input = x["conv1_input"]
        real = self.conv(conv1_input.real)
        imag = self.conv(conv1_input.imag)
        output = torch.complex(real, imag)
        x["conv1_output"] = output
        return x


# convolution layer
class ConvolutionLayer2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvolutionLayer2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
            stride=1,
            dilation=1,
            bias=True,
        )

    def forward(self, x):
        nonlinear_output = x["nonlinear_output"]
        real = self.conv(nonlinear_output.real)
        imag = self.conv(nonlinear_output.imag)
        output = torch.complex(real, imag)

        x["conv2_output"] = output
        return x


# nonlinear layer
class NonlinearLayer(nn.Module):
    def __init__(self):
        super(NonlinearLayer, self).__init__()
        self.pwl = torchpwl.PWL(num_channels=128, num_breakpoints=101)

    def forward(self, x):
        conv1_output = x["conv1_output"]
        y_real = self.pwl(conv1_output.real)
        y_imag = self.pwl(conv1_output.imag)
        output = torch.complex(y_real, y_imag)
        x["nonlinear_output"] = output
        return x
