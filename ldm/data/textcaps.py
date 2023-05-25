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
def make_multi_folder_data(paths, caption_files=None, **kwargs):
    """Make a concat dataset from multiple folders
    Don't support captions yet

    If paths is a list, that's ok, if it's a Dict interpret it as:
    k=folder v=n_times to repeat that
    """
    list_of_paths = []
    if isinstance(paths, (Dict, DictConfig)):
        assert caption_files is None, \
            "Caption files not yet supported for repeats"
        for folder_path, repeats in paths.items():
            list_of_paths.extend([folder_path]*repeats)
        paths = list_of_paths

    if caption_files is not None:
        datasets = [TextCapsDataset(p, caption_file=c, **kwargs) for (p, c) in zip(paths, caption_files)]
    else:
        datasets = [TextCapsDataset(p, **kwargs) for p in paths]
    return torch.utils.data.ConcatDataset(datasets)

class TextCapsDataset(Dataset):
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
        if isinstance(image_transforms, ListConfig):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(), # to be checked
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms
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
                # if self.filter_data:
                #     if not len([word for word in self.filter_words if word in caption.rstrip(".").lower().split(" ")]):
                #         return self.__getitem__(np.random.choice(self.__len__()))
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
                # data[self.cond_stage_key] = caption   
            else:
                filename = self.paths[index]
                caption = self.default_caption
                # data[self.cond_stage_key] = self.default_caption

            if self.filter_data:
                if not len([word for word in self.filter_words if word in caption.rstrip(".").lower().split(" ")]):
                    return self.__getitem__(np.random.choice(self.__len__()))

        if self.return_paths:
            data["path"] = str(filename)
        im = Image.open(filename)
        im = self.process_im(im)
        data[self.first_stage_key] = im
        data[self.cond_stage_key] = caption 
        # if self.captions is not None:
        #     data[self.cond_stage_key] = caption
        # else:
        #     data[self.cond_stage_key] = self.default_caption

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)

