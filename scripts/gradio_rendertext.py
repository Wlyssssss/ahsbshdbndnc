
from cldm.model import load_state_dict
from cldm.ddim_hacked import DDIMSampler
from ldm.util import instantiate_from_config
import os
from omegaconf import OmegaConf
import argparse, os
from torchvision.transforms import ToTensor
from torch import autocast
from contextlib import nullcontext
from scripts.rendertext_tool import Render_Text, load_model_from_config
# def load_model_from_config(cfg, ckpt, verbose=False, not_use_ckpt=False):
#     sd = load_state_dict(ckpt, location='cpu')

#     if "model_ema.input_blocks10in_layers0weight" not in sd:
#         cfg.model.params.use_ema = False 
#     model = instantiate_from_config(cfg.model)

#     if not not_use_ckpt:
#         m, u = model.load_state_dict(sd, strict=False)
#         if len(m) > 0 and verbose:
#             print("missing keys: {}".format(len(m)))
#             print(m)
#         if len(u) > 0 and verbose:
#             print("unexpected keys: {}".format(len(u)))
#             print(u)

#     model.cuda()
#     model.eval()
#     return model




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/stable-diffusion/textcaps_cldm_v20.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--hint_range_m11",
        action="store_true",
        help="the range of the hint image ([-1, 1])",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="full" #"autocast"
    )
    parser.add_argument(
        "--not_use_ckpt",
        action="store_true",
        help="not to use the ckpt",
    )
    parser.add_argument(
        "--build_demo",
        action="store_true",
        help="whether to build the demo",
    )
    parser.add_argument(
        "--sep_prompt",
        action="store_true",
        help="whether to sep the prompt",
    )
    parser.add_argument(
        "--spell_prompt_type",
        type=int,
        default=1,
        help="1: A sign with the word 'xxx' written on it; 2: A sign that says 'xxx'",
    )
    parser.add_argument(
        "--max_num_prompts",
        type=int,
        default=None,
        help="max num of the used prompts",
    )
    parser.add_argument(
        "--grams",
        type=int,
        default=1,
        help="How many grams (words or symbols) to form the to-be-rendered text (used for DrawSpelling Benchmark)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a sign that says 'Stable Diffusion'",
        help="the prompt"
    )
    parser.add_argument(
        "--rendered_txt",
        type=str,
        nargs="?",
        default="Stable Diffusion",
        help="the text to render"
    )
    parser.add_argument(
        "--uncond_glycon_img",
        action="store_true",
        help="whether to set glyph embedding as None while using unconditional conditioning",
    )
    parser.add_argument(
        "--deepspeed_ckpt",
        action="store_true",
        help="whether to use deepspeed while training",
    )
    parser.add_argument(
        "--glyph_img_size",
        type=int,
        default=256,
        help="the size of input images of the glyph image encoder",
    )
    parser.add_argument(
        "--uncond_glyph_image_type",
        type=str,
        default="white",
        help="the type of rendered glyph images as unconditional conditions while using classifier-free guidance"
    )
    parser.add_argument(
        "--remove_txt_in_prompt",
        action="store_true",
        help="whether to remove text in the prompt",
    )
    parser.add_argument(
        "--replace_token",
        type=str,
        default="",
        help="the token used to replace"
    )
    return parser

if not os.path.basename(os.getcwd()) == "stablediffusion":
    os.chdir(os.path.join(os.getcwd(), "stablediffusion"))
    print(os.getcwd()) 
parser = parse_args()
opt = parser.parse_args()

if opt.deepspeed_ckpt:
    assert os.path.isdir(opt.ckpt)
    opt.ckpt = os.path.join(opt.ckpt, "checkpoint", "mp_rank_00_model_states.pt")
    assert os.path.exists(opt.ckpt)

cfg = OmegaConf.load(f"{opt.cfg}")
model = load_model_from_config(cfg, f"{opt.ckpt}", verbose=True, not_use_ckpt=opt.not_use_ckpt)
hint_range_m11 = opt.hint_range_m11
sep_prompt = opt.sep_prompt

ddim_sampler = DDIMSampler(model)
precision_scope = autocast if opt.precision == "autocast" else nullcontext
trans = ToTensor()
render_tool = Render_Text(
        model, precision_scope,
        trans,
        hint_range_m11,
        sep_prompt,
        uncond_glycon_img= cfg.uncond_glycon_img if hasattr(cfg, "uncond_glycon_img") else opt.uncond_glycon_img,
        glyph_control_proc_config= cfg.glyph_control_proc_config if hasattr(cfg, "glyph_control_proc_config") else None,
        glyph_img_size = opt.glyph_img_size,
        uncond_glyph_image_type = cfg.uncond_glyph_image_type if hasattr(cfg, "uncond_glyph_image_type") else opt.uncond_glyph_image_type,
        remove_txt_in_prompt = cfg.remove_txt_in_prompt if hasattr(cfg, "remove_txt_in_prompt") else opt.remove_txt_in_prompt,
        replace_token = cfg.replace_token if hasattr(cfg, "replace_token") else opt.replace_token,
        )


if opt.build_demo:
    import gradio as gr
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown("## Control Stable Diffusion with Glyph Images")
        with gr.Row():
            with gr.Column():
                # input_image = gr.Image(source='upload', type="numpy")
                rendered_txt = gr.Textbox(label="rendered_txt")
                prompt = gr.Textbox(label="Prompt")
                if sep_prompt:
                    prompt_2 = gr.Textbox(label="Prompt_ControlNet")
                else:
                    prompt_2 = gr.Number(value = 0, visible = False) #None #""
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    width = gr.Slider(label="bbox_width", minimum=0., maximum=1, value=0.3, step=0.01)
                    # height = gr.Slider(label="bbox_height", minimum=0., maximum=1, value=0.2, step=0.01)
                    ratio = gr.Slider(label="bbox_width_height_ratio", minimum=0., maximum=5, value=0., step=0.02)
                    top_left_x = gr.Slider(label="bbox_top_left_x", minimum=0., maximum=1, value=0.5, step=0.01)
                    top_left_y = gr.Slider(label="bbox_top_left_y", minimum=0., maximum=1, value=0.5, step=0.01)
                    yaw = gr.Slider(label="bbox_yaw", minimum=-180, maximum=180, value=0, step=5)
                    num_rows =  gr.Slider(label="num_rows", minimum=1, maximum=4, value=1, step=1)
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                    image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                    strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                    guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                    # low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                    # high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                    ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                    scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                    seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                    eta = gr.Number(label="eta (DDIM)", value=0.0)
                    a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                    n_prompt = gr.Textbox(label="Negative Prompt",
                                        value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
            with gr.Column():
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
        ips = [
            rendered_txt, prompt, 
            width, ratio, # height, 
            top_left_x, top_left_y, yaw, num_rows,
            a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, 
            prompt_2
            ]
        run_button.click(fn=render_tool.process, inputs=ips, outputs=[result_gallery])
        # run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


    block.launch(server_name='0.0.0.0', share=True)
else:
    import easyocr
    reader = easyocr.Reader(['en'])
    # num_samples = 1
    # rendered_txt = "happy"
    # prompt = "A sign that says 'happy'"

    num_samples = opt.num_samples
    print("the num of samples is {}".format(num_samples))
    if not opt.from_file:
        prompts = [opt.prompt]
        data = [opt.rendered_txt]
        print("the prompt is {}".format(prompts))
        print("the rendered_txt is {}".format(data))
        assert prompts is not None
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
                    prompts = ['A sign with the word "{}" written on it'.format(line.strip()) for line in data]
                elif opt.spell_prompt_type == 2:
                    prompts = ["A sign that says '{}'".format(line.strip()) for line in data]
                elif opt.spell_prompt_type == 20:
                    prompts = ['A sign that says "{}"'.format(line.strip()) for line in data]
                elif opt.spell_prompt_type == 3:
                    prompts = ["A whiteboard that says '{}'".format(line.strip()) for line in data]
                elif opt.spell_prompt_type == 30:
                    prompts = ['A whiteboard that says "{}"'.format(line.strip()) for line in data]
                else:
                    print("Only five types of prompt templates are supported currently")
                    raise ValueError
                # if opt.verbose_all_prompts:
                #     show_num = opt.max_num_prompts if (opt.max_num_prompts is not None and opt.max_num_prompts >0) else 10
                #     for i in range(show_num):
                #         print("embed the word into the prompt template for {} Benchmark: {}".format(
                #             os.path.basename(opt.from_file), data[i])
                #         )
                # else:  
                #     print("embed the word into the prompt template for {} Benchmark: e.g., {}".format(
                #         os.path.basename(opt.from_file), data[0])
                #         )
            if opt.max_num_prompts is not None and opt.max_num_prompts >0:
                print("only use {} prompts to test the model".format(opt.max_num_prompts))
                data = data[:opt.max_num_prompts]
                prompts = prompts[:opt.max_num_prompts]

    width, ratio, top_left_x, top_left_y, yaw, num_rows = 0.3, 0, 0.5, 0.5, 0, 1
    image_resolution = 512
    strength = 1
    guess_mode = False
    ddim_steps = 20
    scale = 9.0
    seed = 1945923867
    eta = 0
    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
   
    all_results_list = []
    for i in range(len(data)):
        ips = (
        data[i], prompts[i], 
        width, ratio, top_left_x, top_left_y, yaw, num_rows,
        a_prompt, n_prompt, 
        num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta
        )
        all_results = render_tool.process(*ips) #process(*ips)
        all_results_list.extend(all_results[1:] if data[i] != "" else all_results)
    all_ocr_info = []
    for image_array in all_results_list:
        ocr_result = reader.readtext(image_array)
        all_ocr_info.append(ocr_result)

