import os
import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm

from model import SMNIST, MNIST
from data import mnist_loader
from utils import seed_everything


def train(model, train_dl, optimizer, loss_fn, accelerator, epoch):
    model.train()
    total_correct = torch.tensor(0.0, device=accelerator.device)
    total_seen = torch.tensor(0.0, device=accelerator.device)
    total_loss = torch.tensor(0.0, device=accelerator.device)

    for data, target in tqdm(train_dl, desc=f'Training Epoch {epoch + 1:02d}', disable=not accelerator.is_local_main_process):
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_fn(pred, target)
        accelerator.backward(loss)
        optimizer.step()

        batch_size = torch.tensor(len(target), device=accelerator.device, dtype=torch.float32)
        correct = (pred.argmax(dim=1) == target).float().sum()

        total_correct += correct
        total_seen += batch_size
        total_loss += loss.detach() * batch_size

    total_correct = accelerator.gather_for_metrics(total_correct)
    total_seen = accelerator.gather_for_metrics(total_seen)
    total_loss = accelerator.gather_for_metrics(total_loss)

    acc = total_correct.sum().item() / total_seen.sum().item()
    running_loss = total_loss.sum().item() / total_seen.sum().item()
    return running_loss, acc


def evaluate(model, valid_dl, loss_fn, accelerator, epoch):
    model.eval()
    total_correct = torch.tensor(0.0, device=accelerator.device)
    total_seen = torch.tensor(0.0, device=accelerator.device)
    total_loss = torch.tensor(0.0, device=accelerator.device)

    with torch.no_grad():
        for data, target in tqdm(valid_dl, desc=f'Validation Epoch {epoch + 1:02d}', disable=not accelerator.is_local_main_process):
            pred = model(data)
            loss = loss_fn(pred, target)

            batch_size = torch.tensor(len(target), device=accelerator.device, dtype=torch.float32)
            correct = (pred.argmax(dim=1) == target).float().sum()

            total_correct += correct
            total_seen += batch_size
            total_loss += loss.detach() * batch_size

    total_correct = accelerator.gather_for_metrics(total_correct)
    total_seen = accelerator.gather_for_metrics(total_seen)
    total_loss = accelerator.gather_for_metrics(total_loss)

    acc = total_correct.sum().item() / total_seen.sum().item()
    running_loss = total_loss.sum().item() / total_seen.sum().item()
    return running_loss, acc


def main(model, train_dl, valid_dl, optimizer, loss_fn, accelerator, num_epochs):
    model, optimizer, train_dl, valid_dl = accelerator.prepare(model, optimizer, train_dl, valid_dl)

    highest = 0.0
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_dl, optimizer, loss_fn, accelerator, epoch)
        valid_loss, valid_acc = evaluate(model, valid_dl, loss_fn, accelerator, epoch)

        accelerator.print(
            f'Epoch {epoch + 1}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, '
            f'train_acc: {train_acc:.4f}, valid_acc: {valid_acc:.4f}'
        )

        if accelerator.is_main_process and valid_acc > highest:
            highest = valid_acc
            state_dict = accelerator.get_state_dict(model)
            torch.save(state_dict, f'checkpoint/model.pth')
    accelerator.wait_for_everyone()

if __name__ == '__main__':
    seed_everything(42)
    accelerator = Accelerator()
    model = SMNIST()
    train_dl, valid_dl = mnist_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs('./checkpoint', exist_ok=True)
    main(model, train_dl, valid_dl, optimizer, loss_fn, accelerator, num_epochs=40)
    # run:
    # accelerate launch --num_processes=2 train.py

