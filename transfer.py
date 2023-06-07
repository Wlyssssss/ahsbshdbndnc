from omegaconf import OmegaConf
from scripts.rendertext_tool import Render_Text, load_model_from_config
import torch

# cfg = OmegaConf.load("other_configs/config_ema.yaml")
# model = load_model_from_config(cfg, "model_states.pt", verbose=True)
# model = load_model_from_config(cfg, "mp_rank_00_model_states.pt", verbose=True)

cfg = OmegaConf.load("other_configs/config_ema_unlock.yaml")
epoch_idx = 39
model = load_model_from_config(cfg, "epoch={:0>6d}.ckpt".format(epoch_idx), verbose=True)

from pytorch_lightning.callbacks import ModelCheckpoint
with model.ema_scope("store ema weights"):
    model_sd = model.state_dict()
    store_sd = {}
    for key in model_sd:
        if "ema" in key:
            continue
        store_sd[key] = model_sd[key]
    file_content = {
        'state_dict': store_sd
    }
    torch.save(file_content, f"textcaps5K_epoch_{epoch_idx+1}_model_wo_ema.ckpt")
    print("has stored the transfered ckpt.")
print("trial ends!")
