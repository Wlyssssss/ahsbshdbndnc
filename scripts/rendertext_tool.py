from cldm.ddim_hacked import DDIMSampler
import torch
from annotator.render_images import render_text_image_custom
from pytorch_lightning import seed_everything
# save_memory = False
# from cldm.hack import disable_verbosity
# disable_verbosity()
import random
import einops
import numpy as np
from ldm.util import instantiate_from_config
from cldm.model import load_state_dict
from torchvision.transforms import ToTensor
from contextlib import nullcontext

def load_model_from_config(cfg, ckpt, verbose=False, not_use_ckpt=False):

    # if "model_ema.input_blocks10in_layers0weight" not in sd:
    #     print("missing model_ema.input_blocks10in_layers0weight. set use_ema as False")
    #     cfg.model.params.use_ema = False 
    model = instantiate_from_config(cfg.model)

    if ckpt.endswith("model_states.pt"):
        sd = torch.load(ckpt, map_location='cpu')["module"]
    else:
        sd = load_state_dict(ckpt, location='cpu')
   
    keys_ = list(sd.keys())[:]
    for k in keys_:
        if k.startswith("module."):
            nk = k[7:]
            sd[nk] = sd[k]
            del sd[k]

    if not not_use_ckpt:
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys: {}".format(len(m)))
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys: {}".format(len(u)))
            print(u)

    if torch.cuda.is_available():
        model.cuda() 
    model.eval()
    return model

def load_model_ckpt(model, ckpt, verbose=True):
    map_location = "cpu" if not torch.cuda.is_available() else "cuda"
    print("checkpoint map location:", map_location)
    if ckpt.endswith("model_states.pt"):
        sd = torch.load(ckpt, map_location=map_location)["module"]
    else:
        sd = load_state_dict(ckpt, location=map_location)
   
    keys_ = list(sd.keys())[:]
    for k in keys_:
        if k.startswith("module."):
            nk = k[7:]
            sd[nk] = sd[k]
            del sd[k]

    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys: {}".format(len(m)))
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys: {}".format(len(u)))
        print(u)
    model.eval()
    return model

class Render_Text:
    def __init__(self, 
        model,
        precision_scope=nullcontext,
        transform=ToTensor(),
        save_memory = False,
        ):
        self.model = model
        self.precision_scope = precision_scope
        self.transform = transform
        self.ddim_sampler = DDIMSampler(model)
        self.save_memory = save_memory
        
    # process multiple groups of rendered text for building demo
    def process_multi(self, 
            rendered_txt_values, shared_prompt,  
            width_values, ratio_values,  
            top_left_x_values, top_left_y_values, 
            yaw_values, num_rows_values,
            shared_num_samples, shared_image_resolution, 
            shared_ddim_steps, shared_guess_mode, 
            shared_strength, shared_scale, shared_seed, 
            shared_eta, shared_a_prompt, shared_n_prompt,
            only_show_rendered_image=False
            ):
        if shared_seed == -1:
            shared_seed = random.randint(0, 65535)
        seed_everything(shared_seed)
        with torch.no_grad(), \
            self.precision_scope("cuda"), \
            self.model.ema_scope("Sampling on Benchmark Prompts"):
            print("rendered txt:", str(rendered_txt_values), "[t]")
            render_none = len([1 for rendered_txt in rendered_txt_values if rendered_txt != ""]) == 0
            if render_none:
            # if rendered_txt_values == "":
                control = None
                if only_show_rendered_image:
                    return [None]
            else:
                def format_bboxes(width_values, ratio_values, top_left_x_values, top_left_y_values, yaw_values):
                    bboxes = []
                    for width, ratio, top_left_x, top_left_y, yaw in zip(width_values, ratio_values, top_left_x_values, top_left_y_values, yaw_values):
                        bbox = {
                            "width": width,
                            "ratio": ratio,
                            # "height": height,
                            "top_left_x": top_left_x,
                            "top_left_y": top_left_y,
                            "yaw": yaw
                            }
                        bboxes.append(bbox)
                    return bboxes
                
                whiteboard_img = render_text_image_custom(
                    (shared_image_resolution, shared_image_resolution),
                    format_bboxes(width_values, ratio_values, top_left_x_values, top_left_y_values, yaw_values),
                    rendered_txt_values,
                    num_rows_values
                    )
                whiteboard_img = whiteboard_img.convert("RGB")
                
                if only_show_rendered_image:
                    return [whiteboard_img]
                
                control = self.transform(whiteboard_img.copy())
                if torch.cuda.is_available():
                    control = control.cuda()
                control = torch.stack([control for _ in range(shared_num_samples)], dim=0)
                control = control.clone()
                control = [control]
                
            H, W = shared_image_resolution, shared_image_resolution

            # if shared_seed == -1:
            #     shared_seed = random.randint(0, 65535)
            # seed_everything(shared_seed)

            if torch.cuda.is_available() and self.save_memory:
                print("low_vram_shift: is_diffusing", False)
                self.model.low_vram_shift(is_diffusing=False)

            print("control is None: {}".format(control is None))
            if shared_prompt.endswith("."):
                if shared_a_prompt == "":
                    c_prompt = shared_prompt
                else:
                    c_prompt = shared_prompt + " " + shared_a_prompt
            elif shared_prompt.endswith(","):
                if shared_a_prompt == "":
                    c_prompt = shared_prompt[:-1] + "."
                else:
                    c_prompt = shared_prompt + " " + shared_a_prompt
            else:
                if shared_a_prompt == "":
                    c_prompt = shared_prompt + "."
                else:
                    c_prompt = shared_prompt + ", " + shared_a_prompt

            # cond_c_cross = self.model.get_learned_conditioning([shared_prompt + ', ' + shared_a_prompt] * shared_num_samples)
            cond_c_cross = self.model.get_learned_conditioning([c_prompt] * shared_num_samples)
            print("prompt:", c_prompt)
            un_cond_cross = self.model.get_learned_conditioning([shared_n_prompt] * shared_num_samples)
            
            if torch.cuda.is_available() and self.save_memory:
                print("low_vram_shift: is_diffusing", True)
                self.model.low_vram_shift(is_diffusing=True)

            cond = {"c_concat": control, "c_crossattn": [cond_c_cross] if not isinstance(cond_c_cross, list) else cond_c_cross}
            un_cond = {"c_concat": None if shared_guess_mode else control, "c_crossattn": [un_cond_cross] if not isinstance(un_cond_cross, list) else un_cond_cross}
            shape = (4, H // 8, W // 8)

            if not self.model.learnable_conscale:
                self.model.control_scales = [shared_strength * (0.825 ** float(12 - i)) for i in range(13)] if shared_guess_mode else ([shared_strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            else:
                print("learned control scale: {}".format(str(self.model.control_scales)))
            samples, intermediates = self.ddim_sampler.sample(shared_ddim_steps, shared_num_samples,
                                                        shape, cond, verbose=False, eta=shared_eta,
                                                        unconditional_guidance_scale=shared_scale,
                                                        unconditional_conditioning=un_cond)
            if torch.cuda.is_available() and self.save_memory:
                print("low_vram_shift: is_diffusing", False)
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(shared_num_samples)]
        # if rendered_txt_values != "":
        if not render_none:
            return [whiteboard_img] + results
        else:
            return results
        