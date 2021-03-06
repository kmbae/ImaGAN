import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

class GeneratorCNN(nn.Module):
    def __init__(self, input_channel, output_channel, conv_dims, deconv_dims, num_gpu):
        super(GeneratorCNN, self).__init__()
        self.num_gpu = num_gpu
        self.layers = []

        prev_dim = conv_dims[0]
        self.layers.append(nn.Conv2d(input_channel, prev_dim, 4, 2, 1, bias=False))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        for out_dim in conv_dims[1:]:
            self.layers.append(nn.Conv2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = out_dim

        for out_dim in deconv_dims:
            self.layers.append(nn.ConvTranspose2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = out_dim

        self.layers.append(nn.ConvTranspose2d(prev_dim, output_channel, 4, 2, 1, bias=False))
        self.layers.append(nn.Sigmoid())#nn.Tanh())

        self.layer_module = nn.ModuleList(self.layers)

    def main(self, x, y=None):
        if not y==None:
            out = torch.cat([x, y], dim=1)
        else:
            out = x
        for layer in self.layer_module:
            out = layer(out)
        return out

    def forward(self, x, y=None):
        return self.main(x, y)

class GeneratorCNN_g(nn.Module):
    def __init__(self, input_channel, output_channel, conv_dims, deconv_dims, num_gpu):
        super(GeneratorCNN_g, self).__init__()
        self.num_gpu = num_gpu
        self.layers = []

        prev_dim = conv_dims[0]
        self.layers.append(nn.Conv2d(input_channel, prev_dim, 4, 2, 1, bias=False))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        for out_dim in conv_dims[1:]:
            self.layers.append(nn.Conv2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = out_dim

        for out_dim in deconv_dims:
            self.layers.append(nn.ConvTranspose2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = out_dim

        self.layers.append(nn.ConvTranspose2d(prev_dim, output_channel, 4, 2, 1, bias=False))
        self.layers.append(nn.Sigmoid())#nn.Tanh())

        self.layer_module = nn.ModuleList(self.layers)

    def main(self, x, y):
        out = torch.cat([x, y], dim=1)
        for layer in self.layer_module:
            out = layer(out)
        return out

    def forward(self, x, y):
        return self.main(x, y)

class DiscriminatorCNN(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_dims, num_gpu):
        super(DiscriminatorCNN, self).__init__()
        self.num_gpu = num_gpu
        self.layers = []

        prev_dim = hidden_dims[0]
        self.layers.append(nn.Conv2d(input_channel, prev_dim, 4, 2, 1, bias=False))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        for out_dim in hidden_dims[1:]:
            self.layers.append(nn.Conv2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = out_dim

        self.layers.append(nn.Conv2d(prev_dim, output_channel, 4, 1, 0, bias=False))
        self.layers.append(nn.Sigmoid())

        self.layer_module = nn.ModuleList(self.layers)

    def main(self, x, y=None):
        if not y==None:
            out = torch.cat([x, y], dim=1)
        else:
            out = x
        for layer in self.layer_module:
            out = layer(out)
        return out.view(out.size(0), -1)

    def forward(self, x, y=None):
        return self.main(x,y)


class DiscriminatorCNN_f(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_dims, num_gpu):
        super(DiscriminatorCNN_f, self).__init__()
        self.num_gpu = num_gpu
        self.layers = []

        prev_dim = hidden_dims[0]
        self.layers.append(nn.Conv2d(input_channel, prev_dim, 4, 2, 1, bias=False))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        for out_dim in hidden_dims[1:]:
            self.layers.append(nn.Conv2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = out_dim

        self.layers.append(nn.Conv2d(prev_dim, output_channel, 4, 1, 0, bias=False))
        self.layers.append(nn.Sigmoid())

        self.layer_module = nn.ModuleList(self.layers)

    def main(self, x, y):
        out = torch.cat([x, y], dim=1)
        for layer in self.layer_module:
            out = layer(out)
        return out.view(out.size(0), -1)

    def forward(self, x, y):
        return self.main(x,y)

class GeneratorFC(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims):
        super(GeneratorFC, self).__init__()
        self.layers = []

        prev_dim = input_size
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, output_size))

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out

class DiscriminatorFC(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims):
        super(DiscriminatorFC, self).__init__()
        self.layers = []

        prev_dim = input_size
        for idx, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = hidden_dim

        self.layers.append(nn.Linear(prev_dim, output_size))
        self.layers.append(nn.Sigmoid())

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out.view(-1, 1)
