import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def mnist_loader(path=r'./dataset'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_ds = torchvision.datasets.MNIST(root = path,
                                          train = True,
                                          transform = transform,
                                          download = False)

    valid_ds = torchvision.datasets.MNIST(root = path,
                                          train = False,
                                          transform = transform,
                                          download = False)

    train_dl = DataLoader(train_ds, batch_size = 32, shuffle = True)
    valid_dl = DataLoader(valid_ds, batch_size = 32, shuffle = False)
    return train_dl, valid_dl
