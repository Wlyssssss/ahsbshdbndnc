from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from ldm.util import instantiate_from_config
# from datasets import load_dataset
import os
from collections import defaultdict
import cv2 
import albumentations
import random
from ldm.data.util import new_process_im_base, process_wb_im,  vqgan_process_im #, imagenet_process_im
from glob import glob
import random
import base64
from io import BytesIO
from annotator.render_images import render_glyph_image
class LaionOCRCLDataset(Dataset):
    def __init__(self,
        img_folder,
        ocr_folder,
        data_info_file, 
        max_num_samples = -1, 
        no_hint = False, 

        first_stage_key = "jpg", 
        cond_stage_key = "txt",
        control_key = "hint",
        BLIP_caption = False, #True,

        filter_ocr_data = False,
        filter_way = 0, #0, 1, 2 
        ocr_threshold = 0.5,
        ocr_area_ths = 0.1,
        max_token_num = 3,

        rendered_txt_in_caption = False,
        caption_choices = ["original", "w_rend_text", "wo_rend_text"],
        caption_drop_rates = [0.1, 0.5, 0.1],

        postprocess=None,
        new_proc_config = None,

        add_glyph_control = False,
        glyph_control_key = "centered_hint", # "arranged_hint"
        glyph_control_proc_config = None,
        # centered_glyph_folder = None,
        max_glyph_imgs_num = 0, #5,
        glyph_image_encoder_type = "CLIP",
        rm_text_from_cp = False,
        replace_token = "",
        glyph_image_drop_rate = 0,
        uncond_glyph_image_type = "white", #"whiteboard",
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        with open(data_info_file, "r") as f:
            data_infos = f.readlines()
        if max_num_samples > 0:
            data_infos = random.sample(data_infos, max_num_samples)
        self.data_infos = data_infos
        self.img_folder = img_folder
        self.ocr_folder = ocr_folder
        self.ocr_threshold = ocr_threshold
        self.no_hint = no_hint
        self.filter_ocr_data = filter_ocr_data
        self.filter_way = filter_way
        self.max_token_num = max_token_num
        self.ocr_area_ths =ocr_area_ths
        self.caption_choices = caption_choices
        self.caption_drop_rates = caption_drop_rates
        self.rendered_txt_in_caption = rendered_txt_in_caption
        self.BLIP_caption = BLIP_caption
        
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.control_key = control_key

        # centered_hint
        self.add_glyph_control = add_glyph_control #False
        self.glyph_control_key = glyph_control_key
        if self.add_glyph_control:
            if glyph_image_encoder_type not in ["CLIP", "VQGAN"]:
                print("currently not support other types of glyph image encoders")
                raise ValueError
            if glyph_control_proc_config is not None:
                self.glyph_control_proc = instantiate_from_config(glyph_control_proc_config)
            else:
                if glyph_image_encoder_type == "CLIP":
                    self.glyph_control_proc = process_wb_im(exchange_channel= True, image_transforms=[])
                elif glyph_image_encoder_type == "VQGAN":
                    self.glyph_control_proc = vqgan_process_im(augment=False, ori_preprocessor = False)
        self.glyph_image_encoder_type = glyph_image_encoder_type
        self.max_glyph_imgs_num = max_glyph_imgs_num

        # postprocess
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        # image transform
        if new_proc_config is not None:
            self.new_proc_func = instantiate_from_config(new_proc_config)
        else:
            self.new_proc_func = new_process_im_base()
        
        self.filtered_data_list = []
        self.rm_text_from_cp = rm_text_from_cp
        self.replace_token = replace_token
        self.glyph_image_drop_rate = glyph_image_drop_rate
        self.uncond_glyph_image_type = uncond_glyph_image_type


    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        data = {}
        # data info
        data_info = self.data_infos[index]
        info_split = [di.strip() for di in data_info.split("\t")]
        try:
            assert len(info_split) == 5
        except:
            print("data_info_error", len(info_split))
            return self.__getitem__(np.random.choice(self.__len__())) 
        tsv_name = info_split[2]
        
        path_split = tsv_name.split("/")
        try:
            assert len(path_split) <= 2 
        except:
            print("wrong tsv path", tsv_name)
            return self.__getitem__(np.random.choice(self.__len__())) 
        tsv_name = path_split[-1] 
        if len(path_split) == 2:
            img_folder = os.path.join(self.img_folder, path_split[0])
            ocr_folder = os.path.join(
                self.ocr_folder, 
                path_split[0].rstrip("_with_new_caption").replace("ori", "ocr")
                )
        else:
            img_folder = self.img_folder
            ocr_folder = self.ocr_folder

        file_pos = eval(info_split[3])
        idx_in_tsv = eval(info_split[4])
        img_id = "\t".join(info_split[:2])
        if self.filter_ocr_data and img_id in self.filtered_data_list:
            return self.__getitem__(np.random.choice(self.__len__())) 

        # original image
        ori_tsv_file = os.path.join(img_folder, tsv_name)
        with open(ori_tsv_file, "r") as f:
            f.seek(file_pos)
            img_info = f.readline()
        img_info_split = [di.strip() for di in img_info.split("\t")]
        try:
            assert len(img_info_split) >= 4 #=4
            assert img_id == "\t".join(img_info_split[:2])
        except:
            print("image_info_error", len(img_info_split), img_id, "\t".join(img_info_split[:2]))
            return self.__getitem__(np.random.choice(self.__len__())) 
        img_code = img_info_split[2] #[-2]
        try:
            ori_img = Image.open(BytesIO(base64.b64decode(img_code)))
        except:
            print("can't open original image: {}".format(img_id))
            return self.__getitem__(np.random.choice(self.__len__())) 
        if self.BLIP_caption:
            try:
                assert len(img_info_split) == 5
            except:
                print("caption_error", len(img_info_split), img_id, "\t".join(img_info_split[:2]), img_info_split[-1])
                return self.__getitem__(np.random.choice(self.__len__())) 
            caption_ori = img_info_split[-1]
        else:
            caption_ori = img_info_split[3]
        img_size = ori_img.size
        

        # ocr info
        name_split = os.path.splitext(tsv_name)[0].split("_")
        ocr_infos_file = os.path.join(
            ocr_folder, 
            "_".join(name_split[:-1] + ["ocr_info"] + [name_split[-1]]) + ".json"
            )
        try:
            with open(ocr_infos_file, "r") as f:
                ocr_infos = json.load(f)
        except:
            print("can't open ocr info file {}".format(ocr_infos_file))
            return self.__getitem__(np.random.choice(self.__len__())) 
                    
        try:
            ocr_info = ocr_infos[img_id]
            assert len(ocr_info) > 0
        except:
            print("the ocr info of the {} is missing in {}".format(img_id, ocr_infos_file))
            return self.__getitem__(np.random.choice(self.__len__())) 
        
        if self.filter_ocr_data and self.filter_way == 0 and len(ocr_info) > self.max_token_num:
            if img_id not in self.filtered_data_list:
                self.filtered_data_list.append(img_id)
            return self.__getitem__(np.random.choice(self.__len__()))
        
        ocr_area = 0    
        pos_info_list = []
        pos_info_tuples = []
        for info in ocr_info:
            bbox, (text, confidence) = info
            if confidence > self.ocr_threshold:
                xy_info = np.array(bbox)
                min_x, min_y = np.min(xy_info, axis = 0).astype(int)
                max_x, max_y = np.max(xy_info, axis = 0).astype(int)
                pos_info_list.append(
                    [min_x, min_y, max_x, max_y]
                )
                mean_xy = (xy_info[0] + xy_info[2]) / 2
                lf = xy_info[0, 0] # min_x
                pos_info_tuples.append((text, 0.2 * lf + mean_xy[1])) #0.15
                # ocr_txt = info[1]
                if self.filter_ocr_data and self.filter_way == 1:
                    ocr_area += np.abs(
                        np.linalg.det(
                        [xy_info[1] - xy_info[0], xy_info[3] - xy_info[0]]
                        )
                    )
        if self.filter_ocr_data and self.filter_way == 1:
            if ocr_area < self.ocr_area_ths * (img_size[0] * img_size[1]):
                if img_id not in self.filtered_data_list:
                    self.filtered_data_list.append(img_id)
                return self.__getitem__(np.random.choice(self.__len__())) 
            
        pos_info_list = np.array(pos_info_list)
        all_lf, all_up = np.min(pos_info_list[:, :2], axis = 0)
        all_rg, all_dn = np.max(pos_info_list[:, 2:], axis = 0)
        all_pos_info = [all_lf, all_up, all_rg, all_dn]
        # the third way to filter ocr data
        if self.filter_ocr_data and self.filter_way == 2:
            if (all_rg - all_lf) * (all_dn - all_up) < self.ocr_area_ths * (img_size[0] * img_size[1]):
                if img_id not in self.filtered_data_list:
                    self.filtered_data_list.append(img_id)
                return self.__getitem__(np.random.choice(self.__len__())) 

        # hint image
        if not self.no_hint:
            hint_tsv_file = os.path.join(
                ocr_folder, 
                "_".join(name_split[:-1] + ["rendered"] + [name_split[-1]]) + ".tsv"
                )
            with open(hint_tsv_file, "r") as f:
                hint_img_infos = f.readlines()
                hint_img_info = hint_img_infos[idx_in_tsv]
            hint_img_info_split = [di.strip() for di in hint_img_info.split("\t")]
            try:
                assert len(hint_img_info_split) == 3
                assert img_id == "\t".join(hint_img_info_split[:2])
            except:
                print("hint_image_info_error", len(hint_img_info_split), img_id, "\t".join(hint_img_info_split[:2]))
                return self.__getitem__(np.random.choice(self.__len__())) 
            
            hint_img_code = hint_img_info_split[-1]
            try:
                hint_img = Image.open(BytesIO(base64.b64decode(hint_img_code)))
            except:
                print("can't open hint image: {}".format(img_id))
                return self.__getitem__(np.random.choice(self.__len__()))
        else:
            hint_img = None
                # return self.__getitem__(np.random.choice(self.__len__())) 

        assert all_pos_info
        im, im_hint = self.new_proc_func(ori_img, all_pos_info, hint_img)
        
        if not self.no_hint:
            assert im_hint is not None
            data[self.control_key] = im_hint
        data[self.first_stage_key] = im

        caption_wr_text = None
        arrange_tokens = [item[0] for item in (sorted(pos_info_tuples, key=lambda x: x[1]))]
        if self.rendered_txt_in_caption:
            valid_words = " ".join(arrange_tokens)
            caption_wr_text = caption_ori + '. Words in the image: "{}"'.format(valid_words)
            # class_name = ""
            # if class_name == "":
            #     return self.__getitem__(np.random.choice(self.__len__()))
            # else:
            #     caption_wr_text = 'A {} that says "{}".'.format(
            #         class_name, valid_words
            #         )                   
        if self.add_glyph_control:
            drop_glyph_image = torch.rand(1) < self.glyph_image_drop_rate
            # if drop_glyph_image:
            #     aa = 1
            # assert self.uncond_glyph_image_type == "whiteboard"  
            # Currently only support whiteboard images as unconditional condition of glyph image embeddings
            if self.glyph_control_key == "centered_hint":
                glyphs = [rg.strip() for rg in arrange_tokens]
                if len(glyphs) == 0:
                    print("error: glyphs - None")
                    return self.__getitem__(np.random.choice(self.__len__()))
                if self.max_glyph_imgs_num > 0:
                    glyphs = glyphs[:self.max_glyph_imgs_num]
                if not drop_glyph_image:
                    glyph_images = render_glyph_image(glyphs, fill_way="tight") #"both_padding"
                    cglyph_images_procd = []
                    for cgim in glyph_images:
                        if 0 in cgim.size:
                            print("error: glyph image has ", cgim.size, arrange_tokens)
                            return self.__getitem__(np.random.choice(self.__len__()))
                        try:
                            cgim_processed = self.glyph_control_proc(cgim)
                            cglyph_images_procd.append(cgim_processed)          
                        except Exception as e:
                            print(e)
                            print("invalid glyph image", cgim.size)
                            return self.__getitem__(np.random.choice(self.__len__()))
                else:
                    cglyph_images_procd = [
                        self.glyph_control_proc(Image.new("RGB", (224, 224), self.uncond_glyph_image_type))
                    ] * len(glyphs)
                # cglyph_images_procd = [self.glyph_control_proc(cgim) for cgim in glyph_images]
            elif self.glyph_control_key == "arranged_hint":
                assert hint_img is not None
                cglyph_images_procd = [
                    self.glyph_control_proc(
                    hint_img if not drop_glyph_image else 
                    Image.new("RGB", (224, 224), self.uncond_glyph_image_type)
                    )
                ]
            else:
                print("not support glyph control keys beyond 'centered_hint' and 'arranage_hint'")
                raise ValueError
            if isinstance(cglyph_images_procd[0], torch.Tensor): 
                data[self.glyph_control_key] = torch.stack(cglyph_images_procd, dim = 0) 
            elif isinstance(cglyph_images_procd[0], np.ndarray):
                data[self.glyph_control_key] = np.stack(cglyph_images_procd, axis = 0)
            
        caption_wo_text = None 
        if self.rm_text_from_cp and self.BLIP_caption:  # only generate the caption without the rendered words in it while using BLIP captions
            # caption_wo_text = caption_ori
            # for token in arrange_tokens:
            #     caption_wo_text = caption_wo_text.replace(token, self.replace_token)
            caption_items = caption_ori.split(" ")
            lower_arrange_tokens = [tk.lower() for tk in arrange_tokens]
            caption_wo_text = []
            for cp_item in caption_items:
                if cp_item.lower() in lower_arrange_tokens:
                    if self.replace_token != "":
                        caption_wo_text.append(self.replace_token) 
                else:
                    caption_wo_text.append(cp_item)
            caption_wo_text = " ".join(caption_wo_text)
        prompt_list = []
        for i in range(len(self.caption_choices)):
            cc = self.caption_choices[i]
            if cc == "original":
                caption = caption_ori
            elif cc == "w_rend_text":
                caption = caption_wr_text if caption_wr_text is not None else caption_ori
            elif cc == "wo_rend_text":
                caption = caption_wo_text if caption_wo_text is not None else caption_ori
            
            if torch.rand(1) < self.caption_drop_rates[i]:
                caption = ""
            prompt_list.append(caption)

        data[self.cond_stage_key] = prompt_list if len(prompt_list) > 1 else prompt_list[0]

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data
