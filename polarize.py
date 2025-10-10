import torch
import torch.nn as nn
from accelerate import Accelerator

from model import SMNIST
from data import mnist_loader
from train import train, evaluate
from utils import seed_everything, CustomScheduler

class Polarize():
    def __init__(self, model, accelerator, train_dl, valid_dl, optimizer, loss_fn, pretrain_checkpoint=None):
        assert model.polarize, "model.polarize must be set to True for polarization"
        if pretrain_checkpoint is not None:
            model.load_state_dict(torch.load(pretrain_checkpoint), strict=True)
            self.pretrained = True
        else:
            self.pretrained = False

        self.accelerator = accelerator
        self.model, self.train_dl, self.valid_dl, self.optimizer = accelerator.prepare(model, train_dl, valid_dl, optimizer)
        self.loss_fn = loss_fn
        self.epoch_counts = 0
        self.actv1 = self.model.module.actv1.repeats
        self.actv2 = self.model.module.actv2.repeats

    @staticmethod
    def mean_abs(model):
        return torch.mean(torch.cat([p.abs().view(-1) for p in model.parameters() if p.requires_grad]))
        # abs_means = []
        # for p in model.parameters():
        #     if p.requires_grad:
        #         if model.module.polarize:
        #             p = torch.tanh(model.module.get_kk() * p)
        #         abs_means.append(p.abs().mean())
        # return torch.stack(abs_means).mean()

    def evaluate_wihout_activation(self, epoch):
        with torch.no_grad():
            self.model.module.actv1.repeats = 0
            self.model.module.actv2.repeats = 0
            _, valid_acc = evaluate(self.model, self.valid_dl, self.loss_fn, self.accelerator, epoch)
            self.model.module.actv1.repeats = self.actv1
            self.model.module.actv2.repeats = self.actv2
        return valid_acc

    def pretrain(self, epochs = 60):
        if self.pretrained:
            return
        self.accelerator.print(f"Pretrain stage, kk = {self.model.module.get_kk():.2f}")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = CustomScheduler(optimizer)
        for epoch in range(epochs):
            train_loss, train_acc = train(self.model, self.train_dl, optimizer, self.loss_fn, self.accelerator, epoch)
            valid_loss, valid_acc = evaluate(self.model, self.valid_dl, self.loss_fn, self.accelerator, epoch)
            valid_acc2 = self.evaluate_wihout_activation(epoch)
            scheduler.step()
            mean_abs = Polarize.mean_abs(self.model)
            self.epoch_counts += 1
            self.accelerator.print(f'epoch {epoch+1:03d}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, '
                                   f'train_acc: {train_acc:.4f}, valid_acc: {valid_acc:.4f}/{valid_acc2:.4f}, mean_abs: {mean_abs:.4f}')
        self.accelerator.print()
        state_dict = accelerator.get_state_dict(self.model)
        torch.save(state_dict, f'checkpoint/polarize/pretrain.pth')

    def polarize(self, r, subepochs, threshold):
        self.accelerator.print(f"Polarization stage, starting to push kk at ratio {r:.2f}\n")

        while True:
            self.optimizer.state.clear()
            self.model.module.set_kk(self.model.module.get_kk() * r)
            self.accelerator.print(f"kk is now {self.model.module.get_kk()}")
            for epoch in range(subepochs):
                train_loss, train_acc = train(self.model, self.train_dl, self.optimizer, self.loss_fn, self.accelerator, self.epoch_counts)
                valid_loss, valid_acc = evaluate(self.model, self.valid_dl, self.loss_fn, self.accelerator, self.epoch_counts)
                valid_acc2 = self.evaluate_wihout_activation(self.epoch_counts)
                mean_abs = self.mean_abs(self.model)
                self.accelerator.print(
                    f'epoch {epoch + 1:03d}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, '
                    f'train_acc: {train_acc:.4f}, valid_acc: {valid_acc:.4f}/{valid_acc2:.4f}, mean_abs: {mean_abs:.4f}')
                self.epoch_counts += 1
            if self.model.module.get_kk() >= threshold:
                self.accelerator.print("\nPolarization complete")
                state_dict = accelerator.get_state_dict(self.model)
                torch.save(state_dict, f'checkpoint/polarize/polarized.pth')
                break

    def main(self, r=1.5, subepochs=15, threshold=100):
        self.pretrain()
        self.polarize(r, subepochs, threshold)


if __name__ == '__main__':
    seed_everything(42)
    model = SMNIST(polarize=True)
    train_dl, valid_dl = mnist_loader()
    accelerator = Accelerator()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    polarizer = Polarize(model, accelerator, train_dl, valid_dl, optimizer, loss_fn,
                         pretrain_checkpoint='checkpoint/polarize/pretrained.pth')
    polarizer.main()


    # run:
    # accelerate launch --num_processes=2 polarize.py
