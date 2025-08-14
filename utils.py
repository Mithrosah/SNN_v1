import os
import random
import numpy as np
import torch


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

                print(f"processed {layer_name}")
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