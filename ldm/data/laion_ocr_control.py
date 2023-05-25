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
from ldm.data.util import new_process_im #, imagenet_process_im
from glob import glob

class LaionOCRCLDataset(Dataset):
    def __init__(self,
        img_folder,
        no_hint = False, 
        no_caption = False,
        first_stage_key = "jpg", 
        cond_stage_key = "txt",
        control_key = "hint",
        default_caption="",
        ext = "jpg",
        img_folder_sym = "real-images",
        hint_folder_sym = "rendered-images",
        cap_ocr_folder_sym = "info",
        postprocess=None,
        return_paths=False,
        new_proc_config = None,
        random_drop_caption = False,
        drop_caption_p = 0.5,
        ocr_threshold = 0.5,
        filter_ocr_data = False,
        filter_way = 1,
        ocr_area_ths = 0.1,
        fixed_ocr_data = True,
        sep_cap_for_2b = False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        # self.root_dir = Path(img_folder)
        img_files = glob(img_folder + "/*.{}".format(ext))
        if len(img_files) == 0:
            for subfolder in os.listdir(img_folder):
                subpath = os.path.join(img_folder, subfolder)
                if img_folder_sym in subfolder and os.path.isdir(subpath):
                    img_files.extend(
                        glob(subpath + "/*.{}".format(ext))
                    )
        self.img_files = img_files 
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        # postprocess
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        # image transform
        if new_proc_config is not None:
            self.new_proc_func = instantiate_from_config(new_proc_config)
        else:
            self.new_proc_func = new_process_im()

        # caption
        self.default_caption = default_caption
    
        self.return_paths = return_paths

        self.no_hint = no_hint
        self.no_caption = no_caption
        self.control_key = control_key
        self.random_drop_caption = random_drop_caption
        self.drop_caption_p = drop_caption_p
        self.ext = ext
        self.img_folder_sym = img_folder_sym
        self.hint_folder_sym = hint_folder_sym
        self.cap_ocr_folder_sym = cap_ocr_folder_sym
        self.ocr_threshold = ocr_threshold
        self.filter_ocr_data = filter_ocr_data
        self.ocr_area_ths =ocr_area_ths
        self.fixed_ocr_data = fixed_ocr_data
        self.sep_cap_for_2b = sep_cap_for_2b
        self.filter_way = filter_way
        self.filtered_data_list = []

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        data = {}
        filename = self.img_files[index]
        if filename in self.filtered_data_list:
            return self.__getitem__(np.random.choice(self.__len__())) 
        dirname, basename = os.path.split(filename)
        # if basename == '00842_parquet_00842707.jpg':
        #     aa = 1
        root, img_folder = os.path.split(dirname)
        assert basename.endswith(self.ext) and self.img_folder_sym in img_folder
        # caption and ocr info
        names = os.path.splitext(basename)[0].split("_")
        cap_ocr_file = "_".join(names[:-1]) + ".json"
        cap_ocr_folder = img_folder.replace(self.img_folder_sym, self.cap_ocr_folder_sym) 
        cap_ocr_path = os.path.join(root, cap_ocr_folder, cap_ocr_file)
        assert os.path.isfile(cap_ocr_path)
        with open(cap_ocr_path, "r") as f:
            cap_ocr_infos = json.load(f)["ocr_data"]
            for item in cap_ocr_infos:
                if item["image_name"] == basename:
                    cap_ocr_info = item
                    break
        if self.no_caption:
            caption = self.default_caption
        else:
            try:
                caption = cap_ocr_info["caption"]
            except:
                caption = self.default_caption
        ocr_info = cap_ocr_info["ocr_info"]
        pos_info_list = []
        ocr_area = 0
        if len(ocr_info) == 0:
            print("the ocr info of the {} is missing".format(os.path.join(img_folder, basename)))
            return self.__getitem__(np.random.choice(self.__len__())) 
        for info in ocr_info:
            if info[-1] > self.ocr_threshold:
                xy_info = np.array(info[0])
                min_x, min_y = np.min(xy_info, axis = 0).astype(int)
                max_x, max_y = np.max(xy_info, axis = 0).astype(int)
                pos_info_list.append(
                    [min_x, min_y, max_x, max_y]
                )
                # ocr_txt = info[1]
                if self.filter_ocr_data and self.filter_way == 1:
                    ocr_area += np.abs(
                        np.linalg.det(
                        [xy_info[1] - xy_info[0], xy_info[3] - xy_info[0]]
                        )
                    )

        if self.filter_ocr_data and self.filter_way == 1:
            with Image.open(filename) as pic:
                img_size = pic.size
            if ocr_area < self.ocr_area_ths * (img_size[0] * img_size[1]):
                # print("the total ocr area is {}, smaller than {} of the original image size {}".format(
                #     ocr_area, self.ocr_area_ths, str(img_size)
                #     ))
                if filename not in self.filtered_data_list:
                    self.filtered_data_list.append(filename)
                return self.__getitem__(np.random.choice(self.__len__())) 

        pos_info_list = np.array(pos_info_list)
        all_lf, all_up = np.min(pos_info_list[:, :2], axis = 0)
        all_rg, all_dn = np.max(pos_info_list[:, 2:], axis = 0)
        all_pos_info = [all_lf, all_up, all_rg, all_dn]
        # another way to filter ocr data
        if self.filter_ocr_data and self.filter_way == 2:
            with Image.open(filename) as pic:
                img_size = pic.size
            if (all_rg - all_lf) * (all_dn - all_up) < self.ocr_area_ths * (img_size[0] * img_size[1]):
                # print("the total ocr area is {}, smaller than {} of the original image size {}".format(
                #     (all_rg - all_lf) * (all_dn - all_up), self.ocr_area_ths, str(img_size)
                #     ))
                if filename not in self.filtered_data_list:
                    self.filtered_data_list.append(filename)
                return self.__getitem__(np.random.choice(self.__len__())) 

        # hint 
        hint_folder = img_folder.replace(self.img_folder_sym, self.hint_folder_sym) + "-fixed" if self.fixed_ocr_data else ""  
        if not self.no_hint:
            hint_filename = os.path.join(root, hint_folder, basename)
            if not os.path.isfile(hint_filename):
                print("Hint file {} does not exist".format(hint_filename))
                return self.__getitem__(np.random.choice(self.__len__()))
        else:
            hint_filename = None

        assert all_pos_info
        im, im_hint = self.new_proc_func(filename, all_pos_info, hint_filename)
        
        if not self.no_hint:
            assert im_hint is not None
            data[self.control_key] = im_hint
        data[self.first_stage_key] = im
        
        if self.return_paths:
            data["path"] = str(filename)
        
        out_caption = caption 
        if self.random_drop_caption:
            if torch.rand(1) < self.drop_caption_p:
                out_caption = ""
                
        if not self.sep_cap_for_2b:
            data[self.cond_stage_key] = out_caption
        else:
            data[self.cond_stage_key] = [caption, out_caption]

        # if self.random_drop_caption:
        #     if torch.rand(1) < self.drop_caption_p:
        #         caption = ""
        # data[self.cond_stage_key] = caption

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data
