import torch
import sys
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm

from model import SMNIST, SMNIST_CNN
from data import mnist_loader
from utils import seed_everything, load_checkpoint


def Sevaluate(checkpoint_path, valid_dl, seq_len, accelerator):
    model = SMNIST_CNN(polarize=True, seq_len=seq_len)
    model.load_state_dict(torch.load(checkpoint_path), strict=True)

    model, valid_dl = accelerator.prepare(model, valid_dl)

    from polarize import Polarize
    mean_abs = Polarize.get_mean_abs(model)
    accelerator.print(f'mean_abs: {mean_abs:.4f}')

    model.module.prepare_Sforward(model.module.trans)

    model.eval()
    total_correct = torch.tensor(0.0, device=accelerator.device)
    total_seen = torch.tensor(0.0, device=accelerator.device)

    with torch.no_grad():
        for data, target in tqdm(valid_dl, desc=f'Evaluating using bit stream',
                                 disable=not accelerator.is_local_main_process, file=sys.stderr):
            pred = model.module.Sforward(data)

            batch_size = torch.tensor(len(target), device=accelerator.device, dtype=torch.float32)
            correct = (pred.argmax(dim=1) == target).float().sum()

            total_correct += correct
            total_seen += batch_size

    total_correct = accelerator.gather_for_metrics(total_correct)
    total_seen = accelerator.gather_for_metrics(total_seen)

    acc = total_correct.sum().item() / total_seen.sum().item()
    accelerator.print(f"length: {seq_len}, valid accuracy: {acc:.4f}")
    return


def evaluate(checkpoint_path, valid_dl, loss_fn, accelerator):
    model = SMNIST_CNN(polarize=True)
    model.load_state_dict(torch.load(checkpoint_path), strict=True)

    model, valid_dl = accelerator.prepare(model, valid_dl)

    from polarize import Polarize
    mean_abs = Polarize.get_mean_abs(model)
    accelerator.print(f'mean_abs: {mean_abs:.4f}')

    model.eval()
    total_correct = torch.tensor(0.0, device=accelerator.device)
    total_seen = torch.tensor(0.0, device=accelerator.device)
    total_loss = torch.tensor(0.0, device=accelerator.device)

    with torch.no_grad():
        for data, target in tqdm(valid_dl, desc=f'Evaluating', disable=not accelerator.is_local_main_process, file=sys.stderr):
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
    accelerator.print(f"valid loss: {running_loss:.4f}, valid accuracy: {acc:.4f}")
    return


if __name__ == '__main__':
    seed_everything(42)
    _, valid_dl = mnist_loader(batch_size=4)
    accelerator = Accelerator()
    checkpoint_path = './checkpoint/dev_test.pth'

    # for length in [32, 64, 128, 256, 512, 1024, 2048]:
        # Sevaluate(checkpoint_path, valid_dl, length, accelerator)

    Sevaluate(checkpoint_path, valid_dl, 128, accelerator)

    # loss_fn = CrossEntropyLossWithTemperature(temperature=0.35)
    # evaluate(checkpoint_path, valid_dl, loss_fn, accelerator)

    # run:
    # accelerate launch --num_processes=2 test.py


