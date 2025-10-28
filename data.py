import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def mnist_loader(path=r'./dataset', batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        transforms.Resize((27, 27)),
        ])
    train_ds = torchvision.datasets.MNIST(root = path,
                                          train = True,
                                          transform = transform,
                                          download = False)

    valid_ds = torchvision.datasets.MNIST(root = path,
                                          train = False,
                                          transform = transform,
                                          download = False)

    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    valid_dl = DataLoader(valid_ds, batch_size = batch_size, shuffle = False)
    return train_dl, valid_dl

if __name__ == '__main__':
    train_dl, valid_dl = mnist_loader(path=r'D:\ProgramMe\MLDatasets', batch_size=32)
    x, y = next(iter(train_dl))
    print(x.shape)
    print(y.shape)