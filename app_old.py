from cldm.ddim_hacked import DDIMSampler
import math
from omegaconf import OmegaConf
from scripts.rendertext_tool import Render_Text, load_model_from_config
import gradio as gr  
import os
def process_multi_wrapper(rendered_txt_0, rendered_txt_1, rendered_txt_2, rendered_txt_3,
                            shared_prompt,  
                            width_0, width_1, width_2, width_3,  
                            ratio_0, ratio_1, ratio_2, ratio_3,  
                            top_left_x_0, top_left_x_1, top_left_x_2, top_left_x_3,  
                            top_left_y_0, top_left_y_1, top_left_y_2, top_left_y_3,  
                            yaw_0, yaw_1, yaw_2, yaw_3,  
                            num_rows_0, num_rows_1, num_rows_2, num_rows_3,  
                            shared_num_samples, shared_image_resolution,  
                            shared_ddim_steps, shared_guess_mode,  
                            shared_strength, shared_scale, shared_seed,  
                            shared_eta, shared_a_prompt, shared_n_prompt):  
    
    rendered_txt_values = [rendered_txt_0, rendered_txt_1, rendered_txt_2, rendered_txt_3]  
    width_values = [width_0, width_1, width_2, width_3]  
    ratio_values = [ratio_0, ratio_1, ratio_2, ratio_3]  
    top_left_x_values = [top_left_x_0, top_left_x_1, top_left_x_2, top_left_x_3]  
    top_left_y_values = [top_left_y_0, top_left_y_1, top_left_y_2, top_left_y_3]  
    yaw_values = [yaw_0, yaw_1, yaw_2, yaw_3]  
    num_rows_values = [num_rows_0, num_rows_1, num_rows_2, num_rows_3]  
  
    return render_tool.process_multi(rendered_txt_values, shared_prompt,  
                                     width_values, ratio_values,  
                                     top_left_x_values, top_left_y_values,  
                                     yaw_values, num_rows_values,  
                                     shared_num_samples, shared_image_resolution,  
                                     shared_ddim_steps, shared_guess_mode,  
                                     shared_strength, shared_scale, shared_seed,  
                                     shared_eta, shared_a_prompt, shared_n_prompt 
                                    ) 
     
def process_multi_wrapper_only_show_rendered(rendered_txt_0, rendered_txt_1, rendered_txt_2, rendered_txt_3,
                            shared_prompt,  
                            width_0, width_1, width_2, width_3,  
                            ratio_0, ratio_1, ratio_2, ratio_3,  
                            top_left_x_0, top_left_x_1, top_left_x_2, top_left_x_3,  
                            top_left_y_0, top_left_y_1, top_left_y_2, top_left_y_3,  
                            yaw_0, yaw_1, yaw_2, yaw_3,  
                            num_rows_0, num_rows_1, num_rows_2, num_rows_3,  
                            shared_num_samples, shared_image_resolution,  
                            shared_ddim_steps, shared_guess_mode,  
                            shared_strength, shared_scale, shared_seed,  
                            shared_eta, shared_a_prompt, shared_n_prompt):  
    
    rendered_txt_values = [rendered_txt_0, rendered_txt_1, rendered_txt_2, rendered_txt_3]  
    width_values = [width_0, width_1, width_2, width_3]  
    ratio_values = [ratio_0, ratio_1, ratio_2, ratio_3]  
    top_left_x_values = [top_left_x_0, top_left_x_1, top_left_x_2, top_left_x_3]  
    top_left_y_values = [top_left_y_0, top_left_y_1, top_left_y_2, top_left_y_3]  
    yaw_values = [yaw_0, yaw_1, yaw_2, yaw_3]  
    num_rows_values = [num_rows_0, num_rows_1, num_rows_2, num_rows_3]  
  
    return render_tool.process_multi(rendered_txt_values, shared_prompt,  
                                     width_values, ratio_values,  
                                     top_left_x_values, top_left_y_values,  
                                     yaw_values, num_rows_values,  
                                     shared_num_samples, shared_image_resolution,  
                                     shared_ddim_steps, shared_guess_mode,  
                                     shared_strength, shared_scale, shared_seed,  
                                     shared_eta, shared_a_prompt, shared_n_prompt, 
                                     only_show_rendered_image=True)  


cfg = OmegaConf.load("config.yaml")
model = load_model_from_config(cfg, "model_wo_ema.ckpt", verbose=True)
# model = load_model_from_config(cfg, "model_states.pt", verbose=True)
# model = load_model_from_config(cfg, "model.ckpt", verbose=True)

ddim_sampler = DDIMSampler(model)
render_tool = Render_Text(model)


# description = """
# #  <center>Expedit-SAM (Expedite Segment Anything Model without any training)</center>
# Github link: [Link](https://github.com/Expedit-LargeScale-Vision-Transformer/Expedit-SAM)
# You can select the speed mode you want to use from the "Speed Mode" dropdown menu and click "Run" to segment the image you uploaded to the "Input Image" box.
# Points per side is a hyper-parameter that controls the number of points used to generate the segmentation masks. The higher the number, the more accurate the segmentation masks will be, but the slower the inference speed will be. The default value is 12.
# """

description = """
## Control Stable Diffusion with Glyph Images
"""

SPACE_ID = os.getenv('SPACE_ID')
if SPACE_ID is not None:
    # description += f'\n<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. < a href=" ">< img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></ a></p >'
    description += f'\n<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'

block = gr.Blocks().queue()  

with block:  
    with gr.Row():  
        gr.Markdown(description)  
        only_show_rendered_image = gr.Number(value=1, visible=False)
        
    with gr.Column():  
            
        with gr.Row(): 
            for i in range(4):  
                with gr.Column():  
                    exec(f"""rendered_txt_{i} = gr.Textbox(label=f"Render Text {i+1}")""")
                    
                    with gr.Accordion(f"Advanced options {i+1}", open=False):  
                        exec(f"""width_{i} = gr.Slider(label="Bbox Width", minimum=0., maximum=1, value=0.3, step=0.01)  """)
                        exec(f"""ratio_{i} = gr.Slider(label="Bbox_width_height_ratio", minimum=0., maximum=5, value=0., step=0.02, visible=False)  """)
                        exec(f"""top_left_x_{i} = gr.Slider(label="Bbox Top Left x", minimum=0., maximum=1, value={0.35 - 0.25 * math.cos(math.pi * i)}, step=0.01)  """)
                        exec(f"""top_left_y_{i} = gr.Slider(label="Bbox Top Left y", minimum=0., maximum=1, value={0.1 if i < 2 else 0.6}, step=0.01)  """)
                        exec(f"""yaw_{i} = gr.Slider(label="Bbox Yaw", minimum=-180, maximum=180, value=0, step=5) """)
                        # exec(f"""num_rows_{i} = gr.Slider(label="num_rows", minimum=1, maximum=4, value=1, step=1, visible=False)  """)
                        exec(f"""num_rows_{i} = gr.Slider(label="num_rows", minimum=1, maximum=4, value=1, step=1)  """)
        
        with gr.Row(): 
            with gr.Column():
                shared_prompt = gr.Textbox(label="Shared Prompt")
                with gr.Row():
                    run_button = gr.Button(value="Run")
                    show_render_button = gr.Button(value="Only Rendered")
            
            with gr.Accordion("Shared Advanced options", open=False):  
                with gr.Row():
                    shared_num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)  
                    shared_image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64, visible=False)  
                    shared_strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01, visible=False)  
                    shared_guess_mode = gr.Checkbox(label='Guess Mode', value=False, visible=False)  
                    shared_seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                with gr.Row():
                    shared_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)  
                    shared_ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)    
                    shared_eta = gr.Number(label="eta (DDIM)", value=0.0, visible=False)  
                with gr.Row():
                    shared_a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')  
                    shared_n_prompt = gr.Textbox(label="Negative Prompt",  
                                            value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality') 
        
        with gr.Row(): 
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')  
    
    run_button.click(fn=process_multi_wrapper,  
                inputs=[rendered_txt_0, rendered_txt_1, rendered_txt_2, rendered_txt_3,
                        shared_prompt,  
                        width_0, width_1, width_2, width_3,  
                        ratio_0, ratio_1, ratio_2, ratio_3,  
                        top_left_x_0, top_left_x_1, top_left_x_2, top_left_x_3,  
                        top_left_y_0, top_left_y_1, top_left_y_2, top_left_y_3,  
                        yaw_0, yaw_1, yaw_2, yaw_3,  
                        num_rows_0, num_rows_1, num_rows_2, num_rows_3,  
                        shared_num_samples, shared_image_resolution,  
                        shared_ddim_steps, shared_guess_mode,  
                        shared_strength, shared_scale, shared_seed,  
                        shared_eta, shared_a_prompt, shared_n_prompt],  
                outputs=[result_gallery])  
    
    show_render_button.click(fn=process_multi_wrapper_only_show_rendered,  
                inputs=[rendered_txt_0, rendered_txt_1, rendered_txt_2, rendered_txt_3,
                        shared_prompt,  
                        width_0, width_1, width_2, width_3,  
                        ratio_0, ratio_1, ratio_2, ratio_3,  
                        top_left_x_0, top_left_x_1, top_left_x_2, top_left_x_3,  
                        top_left_y_0, top_left_y_1, top_left_y_2, top_left_y_3,  
                        yaw_0, yaw_1, yaw_2, yaw_3,  
                        num_rows_0, num_rows_1, num_rows_2, num_rows_3,  
                        shared_num_samples, shared_image_resolution,  
                        shared_ddim_steps, shared_guess_mode,  
                        shared_strength, shared_scale, shared_seed,  
                        shared_eta, shared_a_prompt, shared_n_prompt],  
                outputs=[result_gallery]) 



    block.launch()