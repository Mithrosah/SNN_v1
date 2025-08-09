import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm

from model import SMNIST
from data import mnist_loader
from train import seed_everything

def Sevaluate(checkpoint_path, valid_dl, seq_len):
    accelerator = Accelerator()
    model = SMNIST(seq_len)
    model.load_state_dict(torch.load(checkpoint_path), strict=True)

    model, valid_dl = accelerator.prepare(model, valid_dl)

    model.eval()
    total_correct = torch.tensor(0.0, device=accelerator.device)
    total_seen = torch.tensor(0.0, device=accelerator.device)
    with torch.no_grad():
        for data, target in tqdm(valid_dl, desc=f'Evaluating using bit stream ',
                                 disable=not accelerator.is_local_main_process):
            pred = (model.module if hasattr(model, "module") else model).Sforward(data)
            batch_size = torch.tensor(len(target), device=accelerator.device, dtype=torch.float32)
            correct = (pred.argmax(dim=1) == target).float().sum()

            total_correct += correct
            total_seen += batch_size

    total_correct = accelerator.gather_for_metrics(total_correct)
    total_seen = accelerator.gather_for_metrics(total_seen)

    acc = total_correct.sum().item() / total_seen.sum().item()
    print("valid accuracy:", acc)
    return acc

if __name__ == '__main__':
    seed_everything(42)
    _, valid_dl = mnist_loader()
    Sevaluate('./checkpoint/best_model.pth', valid_dl, 1024)