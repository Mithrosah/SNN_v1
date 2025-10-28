import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def get_kk(self):
        if self.polarize:
            assert self.fc1.kk.item() == self.fc2.kk.item() == self.fc3.kk.item(), 'kk of fc1,2,3 mismatch'
            return self.fc1.kk.item()
        else:
            raise AttributeError('get_kk() can only be called when polarize is set to True')

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


class SMNIST_CNN(layers.Slayer, nn.Module):
    def __init__(self, polarize=False, seq_len=1024):
        super().__init__()
        self.polarize = polarize
        self.trans = Transform(seq_len)

        self.conv1 = layers.SConv2d(in_channels=1, out_channels=9, kernel_size=3, stride=1, padding=1,
                                    polarize=polarize)
        self.pool1 = layers.SAvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = layers.SConv2d(in_channels=9, out_channels=27, kernel_size=3, stride=1, padding=1,
                                    polarize=polarize)
        self.pool2 = layers.SAvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = layers.SConv2d(in_channels=27, out_channels=81, kernel_size=3, stride=1, padding=1,
                                    polarize=polarize)
        self.pool3 = layers.SAvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4 = layers.SConv2d(in_channels=81, out_channels=243, kernel_size=3, stride=2, padding=1,
                                    polarize=polarize)
        self.pool4 = layers.SAvgPool2d(kernel_size=3, stride=2, padding=1)
        self.fc = layers.SLinear(243, 10, polarize=polarize)

    def set_kk(self, kknew):
        self.conv1.set_kk(kknew)
        self.conv2.set_kk(kknew)
        self.conv3.set_kk(kknew)
        self.conv4.set_kk(kknew)
        self.fc.set_kk(kknew)

    def get_kk(self):
        if self.polarize:
            return self.conv1.kk.item()
        else:
            raise AttributeError('get_kk() can only be called when polarize is set to True')

    def forward(self, x):
        # x = torch.sgn(x)
        x = torch.tanh(x)
        x = self.pool1(F.dropout2d(self.conv1(x), p=0.0, training=self.training))
        x = self.pool2(F.dropout2d(self.conv2(x), p=0.0, training=self.training))
        x = self.pool3(F.dropout2d(self.conv3(x), p=0.0, training=self.training))
        x = self.pool4(F.dropout2d(self.conv4(x), p=0.0, training=self.training))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def shift(self, stream):
        N, C, H, W, num_ints = stream.shape
        stream = stream.reshape(N, -1, num_ints)
        shifted = self.trans.circular_shift(stream)
        shifted = shifted.reshape(N, C, H, W, num_ints)
        return shifted

    def prepare_Sforward(self, trans):
        # must be called before Sforward is called. once will do.
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.fc]:
            layer.prepare_Sforward(trans)

    def Sforward(self, x):
        # x = torch.sgn(x)
        x = torch.tanh(x)

        # stream = self.trans.f2s(x)
        # stream = self.pool1.Sforward(self.conv1.Sforward(stream))
        # stream = self.pool2.Sforward(self.conv2.Sforward(stream))
        # stream = self.pool3.Sforward(self.conv3.Sforward(stream))
        # stream = self.pool4.Sforward(self.conv4.Sforward(stream))
        # stream = stream.squeeze(2).squeeze(2).to(torch.int64)
        # stream = self.fc.Sforward(stream)
        # out = self.trans.s2f(stream)

        stream = self.trans.f2s(x)
        stream = self.conv1.Sforward(stream)
        stream = self.shift(stream)
        stream = self.pool1.Sforward(stream)
        stream = self.shift(stream)

        stream = self.conv2.Sforward(stream)
        stream = self.shift(stream)
        stream = self.pool2.Sforward(stream)
        stream = self.shift(stream)

        stream = self.conv3.Sforward(stream)
        stream = self.shift(stream)
        stream = self.pool3.Sforward(stream)
        stream = self.shift(stream)

        stream = self.conv4.Sforward(stream)
        stream = self.shift(stream)
        stream = self.pool4.Sforward(stream)
        stream = self.shift(stream)

        stream = stream.squeeze(2).squeeze(2).to(torch.int64)
        stream = self.fc.Sforward(stream)
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