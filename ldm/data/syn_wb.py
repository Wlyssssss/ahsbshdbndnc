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
from glob import glob
import re
from bisect import bisect_left, bisect_right
import albumentations, cv2
import time
class SynWhiteBoardDataset(Dataset):
    def __init__(self,
        img_folder,
        caption_folder,
        tsv_info_file, 
        corpus_type = "all_4gram",
        image_transforms=[], 
        first_stage_key = "jpg", 
        cond_stage_key = "txt",
        postprocess=None,
        ext = "png",
        img_class = "whiteboard",
        caption_type = "regular", # "simple" or "regular" or "full"
        lower_case = False,
        max_num = None,
        image_size = 512,
        do_padding = True,
        explict_arrangement = False,
        ) -> None:

        self.root_dir = os.path.join(Path(img_folder), corpus_type)
        self.caption_folder = caption_folder
        assert os.path.exists(self.caption_folder) and os.path.exists(tsv_info_file)
        with open(tsv_info_file, "r") as f:
            tsv_info_dict = json.loads(f.read())
        total_num = 0
        rank_list = []
        for _, value in tsv_info_dict.items():
            total_num += len(value)
            rank_list.append(total_num)
        self.rank_list = rank_list
        self.total_num = total_num if max_num is None else max_num
        self.tsv_info_dict = tsv_info_dict
        self.corpus_type = corpus_type
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        # postprocess
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        # image transform
        if isinstance(image_transforms, ListConfig):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(), # to be checked
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms
        self.ext = ext
        self.num_rank = eval((list(tsv_info_dict.keys())[0]).split("_")[-1].split(".")[0])
        self.img_class = img_class
        self.caption_type = caption_type
        self.lower_case = lower_case
        self.do_padding = do_padding
        self.image_rescaler = albumentations.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_AREA)
        self.image_size = image_size
        self.pad = albumentations.PadIfNeeded(min_height= self.image_size, min_width=self.image_size,
                                              border_mode=cv2.BORDER_CONSTANT, value= (255, 255, 255), 
                                              )
        self.explict_arrangement = explict_arrangement 

    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        pre = time.time()
        data = {}
        rank = bisect_right(self.rank_list, index)
        index_in_tsv = index - ( self.rank_list[rank-1] if rank > 0 else 0 )
        # rank = index % self.num_rank
        # index_in_tsv = index // self.num_rank
        tsv_name = "{}_{}_{}.tsv".format(
            self.corpus_type, rank, self.num_rank
        )
        with open(os.path.join(self.caption_folder, tsv_name), "r") as f:
            f.seek(
                self.tsv_info_dict[tsv_name][index_in_tsv]
            )
            caption_info = f.readline().strip()
        # print("open caption file", time.time() - pre)    
        info_list = caption_info.split("\t")
        assert len(info_list) == 5
        txt_content, font_file, arrange_, align, imagename= info_list

        # imagename= str(index) + ".{}".format(self.ext) 
        filename = os.path.join(self.root_dir, imagename)
        img_pret = time.time()
        try:
            im = Image.open(filename)
            # print("open image time", time.time() - img_pret)
        except:
            return self.__getitem__(np.random.choice(self.__len__()))
        im = self.process_im(im)
        data[self.first_stage_key] = im
        # print("img process time", time.time() - img_pret)
        if self.caption_type == "simple":
            caption = 'A {} that says {}'.format(
                self.img_class, txt_content,
            )
        else:
        # elif self.caption_type == "regular":
            font_weight = ""
            font_style = ""
            font_width = ""
            font_file = re.sub(u'\\[.*?\\]',"", font_file) # remove []
            font_list = font_file[:-4].split("-")
            if len(font_list) > 2:
                print("font file name outlier: {}".format(font_file))
                font_list = [
                    "-".join(font_list[:-1]),
                    font_list[-1]
                ]
            if len(font_list) == 2:
                font_name, font_type = font_list
                if font_type == "VF":
                    font_style = "VF"
                else:
                # font_type = re.sub(u'\\[.*?\\]',"", font_type) # remove []
                    font_tlist = re.findall("[A-Z][a-z]*", font_type)
                    if "Regular" in font_tlist:
                        font_weight = "Regular"    
                        font_style = "Regular"
                    else:
                        # style
                        if "Italic" in font_tlist:
                            font_style = "Italic"
                            font_tlist.remove("Italic")
                        elif "Oblique" in font_tlist:
                            font_style = "Oblique"
                            font_tlist.remove("Oblique")
                        elif "Cursive" in font_tlist:
                            font_style = "Cursive"
                            font_tlist.remove("Cursive")
                        elif "Book" in font_tlist:
                            font_style = "Book"
                            font_tlist.remove("Book") 
                        # width
                        if "Condensed" in font_tlist:
                            font_width = "Condensed"
                            font_tlist.remove("Condensed")   
                        # weight
                        if len(font_tlist):
                            font_weight = " ".join(font_tlist)       
                
            elif len(font_list) == 1:
                font_name = font_list[0]
                # font_name = re.sub(u'\\[.*?\\]',"", font_name) # remove []
                if "Italic" in font_name:
                    font_name = font_name.replace("Italic","")
                    font_style = "Italic"
                if "Bold" in font_name:
                    font_name = font_name.replace("Bold", "")
                    font_weight = "Bold"
            else:
                print("Invalid font file name: {}".format(font_file))
                return self.__getitem__(np.random.choice(self.__len__()))
                            # Width
            if "Condensed" in font_name:
                if "Extra" in font_name or "Semi" in font_name or "Ultra" in font_name:
                    font_name_list = re.findall("[A-Z][a-z]*", font_name)
                    font_width = " ".join(font_name_list[-2:])
                    font_name = "".join(font_name_list[:-2])
                else:
                    font_name = font_name.rstrip("Condensed")
                    font_width = "Condensed"
                # if "ExtraCondensed" in font_name:
                #     font_width = "Extra Condensed"
                # elif "SemiCondensed" in font_name:
                #     font_width = "Semi Condensed"
                # elif "UltraCondensed" in font_name:
                #     font_width = "Ultra Condensed"
                # else:
                #     font_width = "Condensed"
            caption = 'A {} that says {} written in the font of {}'.format(
                self.img_class, txt_content, font_name
            )
            addition_cond = 0
            if font_weight != "":
                font_weight = font_weight.lower() if self.lower_case else font_weight
                caption += " {} {} stroke weight".format(
                    "with" if addition_cond == 0 else "and", font_weight
                )
                addition_cond += 1
            if font_width != "":
                font_width = font_width.lower() if self.lower_case else font_width
                caption += " {} {} font width".format(
                    "with" if addition_cond == 0 else "and", font_width
                )
                addition_cond += 1
            if font_style != "":
                font_style = font_style.lower() if self.lower_case else font_style
                caption += " {} {} font style".format(
                    "with" if addition_cond == 0 else "and", font_style
                ) 
                addition_cond += 1 
            if self.caption_type == "full":
                words = txt_content.strip('"').split(" ")
                assert len(words) == 4
                frn, srn = arrange_.split("_")
                frn, srn = eval(frn), eval(srn)
                assert (frn + srn == 4 )
                if frn == 0 or srn == 0:
                    caption += '. All the words are written in the same row.'
                else:
                    if self.explict_arrangement:
                        caption += '. "{}" is written in the first row while "{}" is in the second row.'.format(
                            ' '.join(words[:frn]),
                            ' '.join(words[frn:])
                        )
                    else:
                        caption += '. The first {} written in the first row while the {} in the second row.'.format(
                            "{} words are".format(frn) if frn >1 else "word is",
                            "other {} words are".format(srn) if srn >1 else "last word is",
                        )  
                # print(caption)
        # print(caption)        
        data[self.cond_stage_key] = caption 
        # if self.captions is not None:
        #     data[self.cond_stage_key] = caption
        # else:
        #     data[self.cond_stage_key] = self.default_caption

        if self.postprocess is not None:
            data = self.postprocess(data)
        
        # print("total time", time.time() - pre)
        return data

    def process_im(self, im):
        im = im.convert("RGB")
        if self.do_padding:
            # pre = time.time()
            im = self.padding_image(im)
            # print("padding time", time.time() - pre)
        return self.tform(im)

    
    def padding_image(self, im):
        # resize 
        im = np.array(im).astype(np.uint8)
        im_rescaled = self.image_rescaler(image=im)["image"]
        # padding
        im_padded = self.pad(image=im_rescaled)["image"]
        return im_padded
        # im_out = Image.fromarray(im_padded)
        # return im_out