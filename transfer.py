from omegaconf import OmegaConf
from scripts.rendertext_tool import Render_Text, load_model_from_config
import torch
cfg = OmegaConf.load("config_cuda.yaml")
model = load_model_from_config(cfg, "model_states.pt", verbose=True)

from pytorch_lightning.callbacks import ModelCheckpoint
with model.ema_scope("store ema weights"):
    file_content = {
        'state_dict': model.state_dict()
    }
    torch.save(file_content, "model.ckpt")
    print("has stored the transfered ckpt.")
print("trial ends!")
