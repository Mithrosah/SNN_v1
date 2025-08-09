import torch
import torch.nn as nn

import layers

class SMNIST(layers.Slayer, nn.Module):
    def __init__(self, seq_len=1024):
        super().__init__(seq_len)

        self.flatten = nn.Flatten()
        self.fc1 = layers.SLinear(729, 243, seq_len=seq_len)
        self.actv1 = layers.SActv(0, seq_len=seq_len)
        self.fc2 = layers.SLinear(243, 81, seq_len=seq_len)
        self.actv2 = layers.SActv(0, seq_len=seq_len)
        self.fc3 = layers.SLinear(81, 10, seq_len=seq_len)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.actv1(self.fc1(x)))
        x = self.dropout(self.actv2(self.fc2(x)))
        x = self.fc3(x)
        if self.fc3.summation:
            return x
        else:
            return torch.sum(x, dim=-1)

    def Sforward(self, x):
        stream = self.trans.f2s(self.flatten(x))
        stream = self.actv1.Sforward(self.fc1.Sforward(stream))
        stream = self.actv2.Sforward(self.fc2.Sforward(stream))
        stream = self.fc3.Sforward(stream)
        out = model.trans.s2f(stream)
        return out

class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # 98.19%
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
    x = torch.rand(4, 1, 27, 27)
    s = model.trans.f2s(x)
    print(model.Sforward(s).shape)