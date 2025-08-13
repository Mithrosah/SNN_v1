import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm
import argparse

from model import SMNIST, MNIST
from data import mnist_loader
from train import seed_everything, train, evaluate


class Polarize():
    def __init__(self, model, accelerator, train_dl, valid_dl, optimizer, loss_fn):
        assert model.polarize, "model.polarize must be set to True for polarization"
        self.accelerator = accelerator
        self.model, self.train_dl, self.valid_dl, self.optimizer = accelerator.prepare(model, train_dl, valid_dl, optimizer)
        self.loss_fn = loss_fn
        self.epoch_counts = 0

    @staticmethod
    def mean_abs(model):
        assert model.polarize, "model.polarize must be set to True for mean_abs info"
        abs_means = []
        for p in model.parameters():
            if p.requires_grad:
                p = torch.tanh(model.module.get_kk() * p)
                abs_means.append(p.abs().mean())
        return torch.stack(abs_means).mean()

    def polarize(self, r = 1.5, pretrain_epochs = 50, subepochs = 15):
        self.accelerator.print(f"Pretrain stage, kk = {self.model.module.get_kk():.2f}")
        for epoch in range(pretrain_epochs):
            train_loss, train_acc = train(self.model, self.train_dl, self.optimizer, self.loss_fn, self.accelerator, epoch)
            valid_loss, valid_acc = evaluate(self.model, self.valid_dl, self.loss_fn, self.accelerator, epoch)
            mean_abs = self.mean_abs(self.model)
            self.epoch_counts += 1
            self.accelerator.print(f'epoch {epoch+1:03d}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, '
                                   f'train_acc: {train_acc:.4f}, valid_acc: {valid_acc:.4f}, mean_abs: {mean_abs:.4f}')
        self.accelerator.print()

        self.accelerator.print(f"Polarization stage, starting to push kk at ratio {r:.2f}\n")
        while True:
            self.optimizer.state.clear()
            self.model.module.set_kk(self.model.module.get_kk() * r)
            self.accelerator.print(f"kk is now {self.model.module.get_kk()}")
            for epoch in range(subepochs):
                self.epoch_counts += 1
                train_loss, train_acc = train(self.model, self.train_dl, self.optimizer, self.loss_fn, self.accelerator, self.epoch_counts)
                valid_loss, valid_acc = evaluate(self.model, self.valid_dl, self.loss_fn, self.accelerator, self.epoch_counts)
                mean_abs = self.mean_abs(self.model)
                self.accelerator.print(
                    f'epoch {epoch + 1:03d}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, '
                    f'train_acc: {train_acc:.4f}, valid_acc: {valid_acc:.4f}, mean_abs: {mean_abs:.4f}')
            if self.model.module.get_kk() >= 1000:
                print("\nPolarization complete")
                break

if __name__ == '__main__':
    seed_everything(42)
    model = SMNIST(polarize=True)
    train_dl, valid_dl = mnist_loader()
    accelerator = Accelerator()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    polarizer = Polarize(model, accelerator, train_dl, valid_dl, optimizer, loss_fn)
    polarizer.polarize()


    # run:
    # accelerate launch --num_processes=2 polarize.py
