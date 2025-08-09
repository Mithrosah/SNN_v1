import os
import random
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm

from model import SMNIST
from data import mnist_loader

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, train_dl, optimizer, loss_fn, accelerator, epoch):
    model.train()
    running_loss = 0.0
    acc = 0.0
    for data, target in tqdm(train_dl, desc=f'Training Epoch {epoch + 1:02d}'):
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_fn(pred, target)
        accelerator.backward(loss)
        optimizer.step()
        running_loss += loss.item() * len(target)
        is_correct = (torch.argmax(pred, dim=1) == target).float()
        acc += is_correct.sum().cpu()
    running_loss /= len(train_dl.dataset)
    acc /= len(train_dl.dataset)
    return running_loss, acc

def evaluate(model, valid_dl, loss_fn, epoch):
    model.eval()
    running_loss = 0.0
    acc = 0.0
    with torch.no_grad():
        for data, target in tqdm(valid_dl, desc=f'Validation Epoch {epoch + 1:02d}'):
            pred = model(data)
            loss = loss_fn(pred, target)
            running_loss += loss.item() * len(target)
            is_correct = (torch.argmax(pred, dim=1) == target).float()
            acc += is_correct.sum().cpu()
    running_loss /= len(valid_dl.dataset)
    acc /= len(valid_dl.dataset)
    return running_loss, acc


def main(model, train_dl, valid_dl, optimizer, loss_fn, accelerator, num_epochs):
    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )

    highest = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_dl, optimizer, loss_fn, accelerator, epoch)
        valid_loss, valid_acc = evaluate(model, valid_dl, loss_fn, epoch)
        print(f'Epoch {epoch + 1}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, '
              f'train_acc: {train_acc:.4f}, valid_acc: {valid_acc:.4f}')
        if valid_acc > highest:
            highest = valid_acc
            torch.save(model.state_dict(), f'checkpoint/acc_{highest}.pth')

if __name__ == '__main__':
    accelerator = Accelerator()
    model = SMNIST()
    train_dl, valid_dl = mnist_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    main(model, train_dl, valid_dl, optimizer, loss_fn, accelerator, num_epochs=40)
    # run:
    # accelerate launch --num_processes=2 train.py

