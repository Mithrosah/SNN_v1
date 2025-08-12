import torch
import torch.nn as nn

import layers
from transform import Transform

class SMNIST(layers.Slayer, nn.Module):
    def __init__(self, polarize=False, seq_len=1024):
        super().__init__()

        self.polarize = polarize
        self.trans = Transform(seq_len)

        self.flatten = nn.Flatten()
        self.fc1 = layers.SLinear(729, 243, polarize=polarize)
        self.actv1 = layers.SActv(0)
        self.fc2 = layers.SLinear(243, 81, polarize=polarize)
        self.actv2 = layers.SActv(0)
        self.fc3 = layers.SLinear(81, 10, polarize=polarize)
        self.dropout = nn.Dropout(0.2)

    def set_kk(self, kknew):
        self.fc1.set_kk(kknew)
        self.fc2.set_kk(kknew)
        self.fc3.set_kk(kknew)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.actv1(self.fc1(x)))
        x = self.dropout(self.actv2(self.fc2(x)))
        x = self.fc3(x)
        if self.fc3.summation:
            return x
        else:
            return torch.sum(x, dim=-1)

    def prepare_Sforward(self, trans):
        # must be called before Sforward is called. once will do.
        for layer in [self.fc1, self.actv1, self.fc2, self.actv2, self.fc3]:
            layer.prepare_Sforward(trans)

    def Sforward(self, x):
        stream = self.trans.f2s(self.flatten(x))
        stream = self.actv1.Sforward(self.fc1.Sforward(stream))
        stream = self.actv2.Sforward(self.fc2.Sforward(stream))
        stream = self.fc3.Sforward(stream)
        out = self.trans.s2f(stream)
        return out

class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(729, 243)
        self.actv1 = nn.Tanh()
        self.fc2 = nn.Linear(243, 81)
        self.actv2 = nn.Tanh()
        self.fc3 = nn.Linear(81, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.actv1(self.fc1(x)))
        x = self.dropout(self.actv2(self.fc2(x)))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    model = SMNIST()
    x = torch.rand(4, 1, 27, 27)*2 - 1

    # with torch.no_grad():
    #     for param in model.parameters():
    #         param.clamp_(-1, 1)

    model.prepare_Sforward(model.trans)
    print(model.Sforward(x).shape)