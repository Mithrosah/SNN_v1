import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(path, model, mode):
    assert mode in ['p2np', 'p2p', 'np2np'], "mode must be 'p2np' or 'p2p' or 'np2np'"
    state_dict = torch.load(path)

    if mode == 'p2np':
        kk_layers = set()
        for key in state_dict.keys():
            if key.endswith('.kk'):
                layer_name = key[:-3]
                kk_layers.add(layer_name)

        for layer_name in kk_layers:
            weight_key = f"{layer_name}.weight"
            kk_key = f"{layer_name}.kk"

            if weight_key in state_dict and kk_key in state_dict:
                layer = model
                for attr in layer_name.split('.'):
                    layer = getattr(layer, attr)

                original_weight = state_dict[weight_key]
                kk_value = state_dict[kk_key]

                layer.weight.data = torch.tanh(original_weight * kk_value)

                # print(f"processed {layer_name}")
            else:
                print(f"Warning: weight or kk is missing in {layer_name} ")

        filtered_state_dict = {}
        for key, value in state_dict.items():
            if not (key.endswith('.kk') or
                    (key.endswith('.weight') and key[:-7] in kk_layers)):
                filtered_state_dict[key] = value

        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=True)


class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs=30, decay_epochs=30,
                 initial_lr=1e-2, final_lr=1e-3, decay='linear', last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay = decay
        self.total_epochs = warmup_epochs + decay_epochs
        super(CustomScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.initial_lr for _ in self.base_lrs]
        elif self.last_epoch < self.total_epochs:
            decay_ratio = (self.last_epoch - self.warmup_epochs) / self.decay_epochs
            if self.decay == 'linear':
                current_lr = self.initial_lr - (self.initial_lr - self.final_lr) * decay_ratio
            elif self.decay == 'exponential':
                current_lr = self.initial_lr * (self.final_lr / self.initial_lr) ** decay_ratio
            else:
                raise ValueError("Decay mode must be either 'linear' or 'exponential'")
            return [current_lr for _ in self.base_lrs]
        else:
            return [self.final_lr for _ in self.base_lrs]


class CrossEntropyLossWithTemperature(nn.Module):
    def __init__(self, temperature=1.0, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, logits, targets):
        scaled_logits = logits / self.temperature
        loss = F.cross_entropy(scaled_logits, targets, reduction=self.reduction)
        return loss
