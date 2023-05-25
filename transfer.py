from omegaconf import OmegaConf
from scripts.rendertext_tool import Render_Text, load_model_from_config
import torch
cfg = OmegaConf.load("config_cuda.yaml")
model = load_model_from_config(cfg, "model_states.pt", verbose=True)

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
    torch.save(file_content, "model_wo_ema.ckpt")
    print("has stored the transfered ckpt.")
print("trial ends!")
