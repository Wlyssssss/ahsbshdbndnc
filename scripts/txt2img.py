import argparse, os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

torch.set_grad_enabled(False)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False, not_use_ckpt=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    if "model_ema.diffusion_modelinput_blocks00weight" not in sd:
        config.model.params.use_ema = False 
    model = instantiate_from_config(config.model)

    if not not_use_ckpt:
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys: {}".format(len(m)))
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys: {}".format(len(u)))
            print(u)

    model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of an astronaut riding a triceratops",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=3,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each prompt in file this often",
    )
    parser.add_argument(
        "--ckpt_folder",
        type=str,
        help="paths to checkpoints of model, if specified, use the checkpoints in the folder",
    )
    parser.add_argument(
        "--max_num_prompts",
        type=int,
        default=None,
        help="max num of the used prompts",
    )

    parser.add_argument(
        "--not_use_ckpt",
        action='store_true',
        help="whether to not use the ckpt",
    )
    parser.add_argument(
        "--spell_prompt_type",
        type=int,
        default=1,
        help="1: A sign with the word 'xxx' written on it; 2: A sign that says 'xxx'",
    )
    parser.add_argument(
        "--update",
        action='store_true',
        help="whether to update the existing generated images",
    )
    parser.add_argument(
        "--grams",
        type=int,
        default=1,
        help="How many grams (words or symbols) to form the to-be-rendered text (used for DrawSpelling Benchmark)",
    )
    parser.add_argument(
        "--save_form",
        type=str,
        help="the form of the saved images, png or pdf",
        # choices=["full", "autocast"],
        default="png"
    )
    parser.add_argument(
        "--verbose_all_prompts",
        action='store_true',
        help="whether to verbose all the prompts to the log",
    )
    return parser
    # opt = parser.parse_args()
    # return opt


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def main(opt):
    seed_everything(opt.seed)

    # batch_size = opt.n_samples
    # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        print("the prompt is {}".format(prompt))
        assert prompt is not None
        batch_size = opt.n_samples if opt.n_samples>0 else 1
        data = [batch_size * [prompt]]
        outpath = os.path.join(
            opt.outdir,
            opt.prompt,
            os.path.splitext(os.path.basename(opt.ckpt))[0]
        )
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            if "gram" in os.path.basename(opt.from_file):
                data = [item.split("\t")[0] for item in data]
            if opt.grams > 1:
                data = [" ".join(data[i:i + opt.grams]) for i in range(0, len(data), opt.grams)]
            if "DrawText_Spelling" in os.path.basename(opt.from_file) or "gram" in os.path.basename(opt.from_file):
                if opt.spell_prompt_type == 1:
                    data = ['A sign with the word "{}" written on it'.format(line.strip()) for line in data]
                elif opt.spell_prompt_type == 2:
                    data = ["A sign that says '{}'".format(line.strip()) for line in data]
                elif opt.spell_prompt_type == 20:
                    data = ['A sign that says "{}"'.format(line.strip()) for line in data]
                elif opt.spell_prompt_type == 3:
                    data = ["A whiteboard that says '{}'".format(line.strip()) for line in data]
                elif opt.spell_prompt_type == 30:
                    data = ['A whiteboard that says "{}"'.format(line.strip()) for line in data]
                else:
                    print("Only five types of prompt templates are supported currently")
                    raise ValueError
                if opt.verbose_all_prompts:
                    show_num = opt.max_num_prompts if (opt.max_num_prompts is not None and opt.max_num_prompts >0) else 10
                    for i in range(show_num):
                        print("embed the word into the prompt template for {} Benchmark: {}".format(
                            os.path.basename(opt.from_file), data[i])
                        )
                else:  
                    print("embed the word into the prompt template for {} Benchmark: e.g., {}".format(
                        os.path.basename(opt.from_file), data[0])
                        )
            if opt.max_num_prompts is not None and opt.max_num_prompts >0:
                print("only use {} prompts to test the model".format(opt.max_num_prompts))
                data = data[:opt.max_num_prompts]
            data = [p for p in data for i in range(opt.repeat)]
            batch_size = opt.n_samples if opt.n_samples>0 else len(data)
            data = list(chunk(data, batch_size))
        outpath = os.path.join(
            opt.outdir,
            os.path.splitext(os.path.basename(opt.from_file))[0] 
            + ("_{}_{}_gram".format(opt.spell_prompt_type, opt.grams) if "DrawText_Spelling" in os.path.basename(opt.from_file) else ""),
            os.path.splitext(os.path.basename(opt.ckpt))[0]
        )
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if os.path.exists(outpath):
        if not opt.update:
            print("{} already exists and we will not update it".format(outpath))
            return
        else:
            print("{} already exists but we will update it".format(outpath))
    os.makedirs(outpath, exist_ok=True)
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    save_form = opt.save_form
    sample_count = 0
    sample_limit = 15 #20 #10

    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}", verbose=True, not_use_ckpt=opt.not_use_ckpt)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    elif opt.dpm:
        # DPM-Solver
        sampler = DPMSolverSampler(model)
    else:
        sampler = DDIMSampler(model)
    
    # os.makedirs(opt.outdir, exist_ok=True)
    # outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    
    print("precison strategy: {}".format(opt.precision))
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad(), \
        precision_scope("cuda"), \
        model.ema_scope("Sampling on Benchmark Prompts"):
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    batch_size_real = len(prompts)
                    if opt.scale != 1.0: # classifier-free guidance
                        uc = model.get_learned_conditioning(batch_size_real * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    # prompt
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples, _ = sampler.sample(S=opt.steps,
                                                     conditioning=c,
                                                     batch_size=batch_size_real, #opt.n_samples,
                                                     shape=shape,
                                                     verbose=False, #False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code)

                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    # from [-1,1] to [0,1]
                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img = put_watermark(img, wm_encoder)
                        img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1
                        sample_count += 1
                    if len(x_samples) != batch_size: #opt.n_samples:
                        x_samples = torch.concat(
                            [x_samples, torch.ones(
                                (batch_size - len(x_samples), ) + x_samples.shape[1:]
                            ).to(x_samples.device)], dim=0
                        )
                    all_samples.append(x_samples)
                if sample_count >= sample_limit and len(all_samples):
                    grid_count = save_imgs_as_grid(all_samples, n_rows, wm_encoder, outpath, grid_count, save_form=save_form)
                    all_samples = []
                    sample_count = 0

            if len(all_samples):
                grid_count = save_imgs_as_grid(all_samples, n_rows, wm_encoder, outpath, grid_count, save_form=save_form)
                
    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

def save_imgs_as_grid(all_samples, n_rows, wm_encoder, outpath, grid_count, save_form="png"):
    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_rows)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    grid = Image.fromarray(grid.astype(np.uint8))
    grid = put_watermark(grid, wm_encoder)
    grid.save(os.path.join(outpath, f'grid-{grid_count:04}.{save_form}'))
    grid_count += 1
    return grid_count

if __name__ == "__main__":
    import os
    from glob import glob
    if not os.path.basename(os.getcwd()) == "stablediffusion":
        os.chdir(os.path.join(os.getcwd(), "stablediffusion"))
        print(os.getcwd()) 
    parser = parse_args()
    opt = parser.parse_args()
    # ckpt_list = ["epoch=000047-step=000148999.ckpt"]
    # ckpt_list = ["epoch=000005-step=000015999.ckpt"]
    ckpt_list = [
        "epoch=000000-step=000000999.ckpt",
        "epoch=000004-step=000012999.ckpt",
        "epoch=000007-step=000024999.ckpt",
        "epoch=000012-step=000037999.ckpt",
        "epoch=000015-step=000048999.ckpt",
        "epoch=000016-step=000050999.ckpt",
        "epoch=000020-step=000062999.ckpt",
        "epoch=000023-step=000074999.ckpt",
        "epoch=000027-step=000086999.ckpt",
        "epoch=000031-step=000097999.ckpt",
        "epoch=000031-step=000099999.ckpt", 
        "epoch=000032-step=000100999.ckpt",
        "epoch=000039-step=000124999.ckpt",
        "epoch=000047-step=000149999.ckpt", 
        "epoch=000063-step=000199999.ckpt"
        ]
    # ckpt_list = ["epoch=000005-step=000003999.ckpt", "epoch=000007-step=000004999.ckpt"]
    # ckpt_list = ["epoch=000007-step=000009999.ckpt", "epoch=000000-step=000000999.ckpt", "epoch=000014-step=000019999.ckpt"]
    if opt.ckpt_folder is not None:
        for ckpt in glob(opt.ckpt_folder + "/*.ckpt"):
            if os.path.basename(ckpt) not in ckpt_list:
                continue
            opt.ckpt = ckpt
            try:
                main(opt)
            except:
                continue
    else:
        try:
            main(opt)
        except:
            raise ValueError
