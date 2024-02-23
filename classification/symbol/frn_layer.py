import torch
from torch import nn

class FRNLayer2d(nn.Module):
    """
    Filter Response Normalization
    Thresholded Linear Unit
    """

    def __init__(self,c, eps=1e-6):
        super(FRNLayer2d, self).__init__()
        self.register_parameter("gamma", torch.nn.Parameter(torch.ones(size=(1,c,1,1))))
        self.register_parameter("beta", torch.nn.Parameter(torch.zeros(size=(1,c,1,1))))
        self.register_parameter("tau", torch.nn.Parameter(torch.zeros(size=(1,c,1,1))))
        self.register_buffer("eps",torch.tensor(eps))

    def forward(self, x):
        mu2 = torch.mean(torch.square(x),dim=(2,3),keepdim=True)
        x = x * torch.rsqrt(mu2 + torch.abs(self.eps))
        return torch.maximum(self.gamma * x + self.beta, self.tau)


if __name__ == "__main__":
    C = 32
    layer = FRNLayer2d(C)
    layer.train()
    x = torch.rand(size=(2,C, 64, 128))
    y = layer(x)
    loss = torch.mean(y) - torch.mean(x)
    loss.backward()
    print(x.shape,y.shape)
    print(x.mean(), x.std(), y.mean(), y.std())
    print("done")