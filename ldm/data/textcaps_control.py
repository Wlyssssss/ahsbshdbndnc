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
from ldm.data.util import new_process_im, imagenet_process_im

class TextCapsCLDataset(Dataset):
    def __init__(self,
        img_folder,
        caption_file=None,
        image_transforms=[],
        first_stage_key = "jpg", cond_stage_key = "txt",
        OneCapPerImage = False,
        default_caption="",
        ext="jpg",
        postprocess=None,
        return_paths=False,

        filter_data=False,
        filter_words=["sign", "poster"], 

        ocr_file=None,
        no_hint = False, 
        hint_folder = None,
        control_key = "hint",
        # aug4hint = True,
        do_tutorial_proc = False,
        imagenet_proc = False,
        imagenet_proc_config = None,
        filter_ocr_tokens = False,
        do_new_proc = True,
        new_proc_config = None,
        random_drop_caption = False,
        drop_caption_p = 0.5,
        new_ocr_info = True,
        sep_cap_for_2b = False,
        rendered_txt_in_caption = False,
        filter_token_num = False,
        max_token_num = 3,
        random_drop_sd_caption = False,
        drop_sd_caption_p = 0.1,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(img_folder)
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        # postprocess
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        # image transform
        self.imagenet_proc = imagenet_proc
        self.do_new_proc = do_new_proc
        self.do_tutorial_proc = do_tutorial_proc
        # self.aug4hint = aug4hint
        if self.do_new_proc:
            if new_proc_config is not None:
                self.new_proc_func = instantiate_from_config(new_proc_config)
            else:
                self.new_proc_func = new_process_im()
        elif not self.do_tutorial_proc:
            if self.imagenet_proc:
                if imagenet_proc_config is not None:
                    self.imagenet_proc_func = instantiate_from_config(imagenet_proc_config)
                else:
                    self.imagenet_proc_func = imagenet_process_im()
                self.process_im = self.imagenet_proc_func 
            else:
                if isinstance(image_transforms, ListConfig):
                    image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
                image_transforms.extend([transforms.ToTensor(), # to be checked
                                        transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
                image_transforms = transforms.Compose(image_transforms)
                self.tform = image_transforms
                self.process_im = self.simple_process_im
        # caption
        if caption_file is not None:
            with open(caption_file, "rt") as f:
                ext = Path(caption_file).suffix.lower()
                if ext == ".json":
                    captions = json.load(f)
                # elif ext == ".jsonl":
                #     lines = f.readlines()
                #     lines = [json.loads(x) for x in lines]
                #     captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
                else:
                    raise ValueError(f"Unrecognised format: {ext}")
            self.captions = captions["data"]
            if OneCapPerImage and ocr_file is None:
                new_captions = []
                taken_images = []
                for caption_data in self.captions:
                    if caption_data["image_id"] in taken_images:
                        continue
                    else:
                        new_captions.append(caption_data)
                        taken_images.append(caption_data["image_id"])
                self.captions = new_captions
        else:
            self.captions = None

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        # Only used if there is no caption file
        self.paths = []
        for e in ext:
            self.paths.extend(list(self.root_dir.rglob(f"*.{e}")))

        self.default_caption = default_caption
        self.return_paths = return_paths
        self.filter_data = filter_data
        self.filter_words = filter_words

        self.ocr_file = ocr_file
        self.ocr_data = []
        if ocr_file is not None:
            assert self.captions is not None
            with open(ocr_file, "r") as f:
                ocrs = json.loads(f.read())
                ocr_data = ocrs['data']
            self.ocr_data = ocr_data

        self.no_hint = no_hint
        self.control_key = control_key
        self.hint_folder = None
        if not self.no_hint:
            if hint_folder is None:
                print("Warning: The folder of hint images is not provided! No hint will be used")
                self.no_hint = True
            else:
                self.hint_folder = Path(hint_folder)

        self.filter_ocr_tokens = filter_ocr_tokens
        self.random_drop_caption = random_drop_caption
        self.drop_caption_p = drop_caption_p
        self.new_ocr_info = new_ocr_info
        self.sep_cap_for_2b = sep_cap_for_2b
        self.rendered_txt_in_caption = rendered_txt_in_caption
        self.filter_token_num = filter_token_num
        self.max_token_num = max_token_num
        self.random_drop_sd_caption = random_drop_sd_caption
        self.drop_sd_caption_p = drop_sd_caption_p

    def __len__(self):
        if self.ocr_file is not None:
            return len(self.ocr_data)
        if self.captions is not None:
            # return len(self.captions.keys())
            return len(self.captions)
        else:
            return len(self.paths)

    def __getitem__(self, index):
        data = {}
        if self.ocr_file is not None:
            sample = self.ocr_data[index]
            image_id = sample["image_id"]
            ocr_tokens = sample["ocr_tokens"]
            ocr_info = sample["ocr_info"]
            chosen = image_id + ".jpg"
            filename = self.root_dir/chosen

            for d in self.captions:
                if d["image_id"] == image_id:
                    image_captions = d["reference_strs"]
                    image_classes = d["image_classes"]
                    break

            if not len(ocr_tokens) or not len(image_captions) or not len(image_classes):
                return self.__getitem__(np.random.choice(self.__len__()))
            
            if self.filter_ocr_tokens:
                tokens_state=defaultdict(list)
                for token in ocr_tokens:
                    token_info = [
                    caption for caption in image_captions if (token.lower() in caption.rstrip(".").lower().split(" ")) 
                    ]
                    tokens_state[len(token_info)].append(token.lower())

                max_n = max(tokens_state.keys())
                if max_n > 0:
                    valid_tokens = list(set(tokens_state[max_n]))
                    pos_info = dict()
                    for token in valid_tokens:
                        for item in ocr_info:
                            if item['word'].lower() == token:
                                token_box = item['bounding_box']
                                tx, ty = token_box['top_left_x'], token_box['top_left_y']
                                pos_info[token] = tx+ty
                                break
                    # arrange_tokens = list(dict(sorted(pos_info.items(), key=lambda x: x[1])).keys())
                    arrange_tokens = [item[0] for item in (sorted(pos_info.items(), key=lambda x: x[1]))]
                    valid_words = " ".join(arrange_tokens)
                    class_name = ""
                    for word in self.filter_words:
                        if word in " ".join(image_classes).lower():
                            class_name = word
                            break
                    if class_name == "":
                        return self.__getitem__(np.random.choice(self.__len__()))
                    else:
                        caption = "A {} that says '{}'.".format(
                            class_name, valid_words
                            )
                else:
                    return self.__getitem__(np.random.choice(self.__len__()))
            else:
                caption = random.choice(image_captions)
                if self.filter_data:
                    if not len([word for word in self.filter_words if word in " ".join(image_classes).lower()]):
                        return self.__getitem__(np.random.choice(self.__len__()))
                with Image.open(filename) as img:
                    im_w, im_h = img.size 
                pos_info_list = []
                pos_info_dict = dict()
                if self.filter_token_num and len(ocr_info) > self.max_token_num:
                    return self.__getitem__(np.random.choice(self.__len__()))
                for item in ocr_info:
                    token_box = item['bounding_box']
                    lf, up = token_box['top_left_x'], token_box['top_left_y']
                    w, h = token_box['width'], token_box['height']
                    if not self.new_ocr_info:
                        # old version
                        rg, dn = lf + w, up + h
                        pos_info_list.append([lf, up, rg, dn])
                    else:
                        ## fix the bug when rotation happens
                        # pos_info_dict[item["word"]] = 0.06 * lf + up
                        lf, w = int(lf * im_w), int(w * im_w)
                        up, h = int(up * im_h), int(h * im_h)
                        yaw = token_box['yaw']
                        # if yaw > 5:
                        #     aa = 1
                        tf_xy = np.array([lf, up])
                        yaw = yaw * np.pi / 180
                        rotate_mx = np.array([
                            [np.cos(yaw), -np.sin(yaw)],
                            [np.sin(yaw), np.cos(yaw)]
                        ])
                        rel_cord = np.matmul(rotate_mx, np.array(
                            [[0, 0],
                            [w, 0],
                            [0, h],
                            [w, h]]
                        ).T)
                        min_xy = np.min(rel_cord, axis = 1).astype(int) + tf_xy
                        max_xy = np.max(rel_cord, axis = 1).astype(int) + tf_xy
                        pos_info_list.append(
                            [
                            min_xy[0], min_xy[1],
                            max_xy[0], max_xy[1]
                            ]
                        )
                        mean_xy = rel_cord[:, -1] / 2 + tf_xy
                        pos_info_dict[item["word"]] = 0.2 * lf + mean_xy[1] #0.15
                pos_info_list = np.array(pos_info_list)
                all_lf, all_up = np.min(pos_info_list[:, :2], axis = 0)
                all_rg, all_dn = np.max(pos_info_list[:, 2:], axis = 0)
                all_pos_info = [all_lf, all_up, all_rg, all_dn]
                if self.rendered_txt_in_caption:
                    assert self.filter_data
                    arrange_tokens = [item[0] for item in (sorted(pos_info_dict.items(), key=lambda x: x[1]))]
                    valid_words = " ".join(arrange_tokens)
                    class_name = ""
                    for word in self.filter_words:
                        if word in " ".join(image_classes).lower():
                            class_name = word
                            break
                    if class_name == "":
                        return self.__getitem__(np.random.choice(self.__len__()))
                    else:
                        out_caption = 'A {} that says "{}".'.format(
                            class_name, valid_words
                            )

        else:
            if self.captions is not None:
                # chosen = list(self.captions.keys())[index]
                # caption = self.captions.get(chosen, None)
                caption_data = self.captions[index]
                chosen = os.path.basename(caption_data["image_path"])
                caption = caption_data["caption_str"]
                if caption is None:
                    caption = self.default_caption
                filename = self.root_dir/chosen
                image_classes = caption_data["image_classes"]
                # data[self.cond_stage_key] = caption   
            else:
                filename = self.paths[index]
                caption = self.default_caption
                image_classes = [""]
                # data[self.cond_stage_key] = self.default_caption

            if self.filter_data:
                if not len([word for word in self.filter_words if word in " ".join(image_classes).lower()]):
                    return self.__getitem__(np.random.choice(self.__len__()))                    
                # if not len([word for word in self.filter_words if word in caption.rstrip(".").lower().split(" ")]):
                #     return self.__getitem__(np.random.choice(self.__len__()))
        
        if not self.no_hint:
            hint_filename = self.hint_folder/chosen
            if not os.path.isfile(hint_filename):
                print("Hint file {} does not exist".format(hint_filename))
                return self.__getitem__(np.random.choice(self.__len__()))
        else:
            hint_filename = None

        if self.do_tutorial_proc:
            # to be aborted
            im, im_hint = self.tutorial_process_im(filename, hint_filename)
        elif self.do_new_proc:
            # recommended
            assert all_pos_info
            im, im_hint = self.new_proc_func(filename, all_pos_info, hint_filename)
        else:
            im_hint = None
            im = Image.open(filename)
            im = self.process_im(im) # not supported for the flip option for now
            if hint_filename is not None:
                im_hint = Image.open(hint_filename)
                im_hint = self.process_im(im_hint) #if self.aug4hint else self.noaug_process_im(im_hint)

        
        if not self.no_hint:
            assert im_hint is not None
            data[self.control_key] = im_hint
        data[self.first_stage_key] = im
        
        if self.return_paths:
            data["path"] = str(filename)
        
        if not self.rendered_txt_in_caption:
            out_caption = caption

        if self.random_drop_caption:
            if torch.rand(1) < self.drop_caption_p:
                out_caption = ""

        if self.random_drop_sd_caption:
            assert self.sep_cap_for_2b
            if torch.rand(1) < self.drop_sd_caption_p:
                caption = ""
                
        if not self.sep_cap_for_2b:
            data[self.cond_stage_key] = out_caption
        else:
            data[self.cond_stage_key] = [caption, out_caption]

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def simple_process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
    
    # def noaug_process_im(self, im):
    #     # To be aborted: lack consideration of different image sizes
    #     im = im.convert("RGB")
    #     im_trans = [transforms.ToTensor(), # to be checked
    #                              transforms.Lambda(lambda x: rearrange(x, 'c h w -> h w c'))]
    #     im_trans= transforms.Compose(im_trans)
    #     im = im_trans(im)
    #     return im
    
    def tutorial_process_im(self, target_filename, source_filename = None):
        # To be aborted: lack consideration of different image sizes
        target = cv2.imread(target_filename)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = (target.astype(np.float32) / 127.5) - 1.0  # Normalize target images to [-1, 1].

        if source_filename is not None:
            source = cv2.imread(source_filename)
            # Do not forget that OpenCV read images in BGR order.
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            # Normalize source images to [0, 1].
            source = source.astype(np.float32) / 255.0
        else:
            source = None
        return target, source

    


