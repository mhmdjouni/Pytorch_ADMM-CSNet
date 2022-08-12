import torch
import torch.nn as nn


# reconstruction original layers
class ReconstructionOriginalLayer(nn.Module):
    def __init__(self, rho, mask):
        super(ReconstructionOriginalLayer, self).__init__()
        self.rho = rho
        self.mask = mask

    def forward(self, x):
        mask = self.mask
        denom = torch.add(mask.cuda(), self.rho)
        a = 1e-6
        value = torch.full(denom.size(), a).cuda()
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)

        orig_output2 = torch.mul(x, orig_output1)
        orig_output3 = torch.fft.ifft2(orig_output2)
        # define data dict
        cs_data = dict()
        cs_data["input"] = x
        cs_data["conv1_input"] = orig_output3
        return cs_data


# reconstruction middle layers
class ReconstructionUpdateLayer(nn.Module):
    def __init__(self, rho, mask):
        super(ReconstructionUpdateLayer, self).__init__()
        self.rho = rho
        self.mask = mask

    def forward(self, x):
        minus_output = x["minus_output"]
        multiple_output = x["multi_output"]
        inputt = x["input"]
        mask = self.mask
        number = torch.add(
            inputt,
            self.rho
            * torch.fft.fft2(torch.sub(minus_output, multiple_output)),
        )
        denom = torch.add(mask.cuda(), self.rho)
        a = 1e-6
        value = torch.full(denom.size(), a).cuda()
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)
        orig_output2 = torch.mul(number, orig_output1)
        orig_output3 = torch.fft.ifft2(orig_output2)
        x["re_mid_output"] = orig_output3
        return x


# reconstruction final layers
class ReconstructionFinalLayer(nn.Module):
    def __init__(self, rho, mask):
        super(ReconstructionFinalLayer, self).__init__()
        self.rho = rho
        self.mask = mask

    def forward(self, x):
        minus_output = x["minus_output"]
        multiple_output = x["multi_output"]
        inputt = x["input"]
        mask = self.mask
        number = torch.add(
            inputt,
            self.rho
            * torch.fft.fft2(torch.sub(minus_output, multiple_output)),
        )
        denom = torch.add(mask.cuda(), self.rho)
        a = 1e-6
        value = torch.full(denom.size(), a).cuda()
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)
        orig_output2 = torch.mul(number, orig_output1)
        orig_output3 = torch.fft.ifft2(orig_output2)
        x["re_final_output"] = orig_output3
        return x["re_final_output"]
