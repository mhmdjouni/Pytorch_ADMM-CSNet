import torch
import torch.nn as nn


# multiple original layer
class MultipleOriginalLayer(nn.Module):
    def __init__(self, gamma):
        super(MultipleOriginalLayer, self).__init__()
        self.gamma = gamma

    def forward(self, x):
        org_output = x["conv1_input"]
        minus_output = x["minus_output"]
        output = torch.mul(self.gamma, torch.sub(org_output, minus_output))
        x["multi_output"] = output
        return x


# multiple middle layer
class MultipleUpdateLayer(nn.Module):
    def __init__(self, gamma):
        super(MultipleUpdateLayer, self).__init__()
        self.gamma = gamma

    def forward(self, x):
        multiple_output = x["multi_output"]
        re_mid_output = x["re_mid_output"]
        minus_output = x["minus_output"]
        output = torch.add(
            multiple_output,
            torch.mul(self.gamma, torch.sub(re_mid_output, minus_output)),
        )
        x["multi_output"] = output
        return x
