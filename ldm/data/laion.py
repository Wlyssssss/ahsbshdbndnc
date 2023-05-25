import os #yaml, pickle, shutil, tarfile, 
from glob import glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
# from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from torch.utils.data import Dataset #, Subset
import pandas as pd
from torchvision import transforms
from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light
from skimage import io
from tqdm import tqdm
import base64
from io import BytesIO
from ldm.data.base import Txt2ImgIterableBaseDataset
import multiprocessing as mp
from bisect import bisect_left, bisect_right
import omegaconf
import time
import json
from torch.utils.data.dataloader import _get_distributed_settings
class LAIONBase(Dataset):
    def __init__(self, img_folder, caption_folder=None, 
                 recollect_data_info = False,
                #  indices_file = None, 
                 first_stage_key = "jpg", cond_stage_key = "txt", do_flip = False,
                 size=None, degradation=None, 
                 downscale_f=4, min_crop_f=0.5, max_crop_f=1., flip_p=0.5,
                 random_crop=True):
        """
        LAION Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        # 3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.root = img_folder.split("/laion")[0]
        self.images = []
        self.texts = []
        self.paired_data = []
        self.parquet_info = {}
        # self.load_from_origin_data(caption_folder)
        # self.load_data(img_folder, caption_folder)
        # self.load_from_parquet(img_folder)
        data_info_file = os.path.join(img_folder, "data_info.json")
        # if os.path.exists(data_info_file):
        collect_data_info = True
        if not recollect_data_info:
            try:
                with open(data_info_file, "r") as f:
                    # f.write(json.dump(self.data_info))
                    self.data_info = json.loads(f.read())
                    collect_data_info = False
            except:
                print(
                    "fail to load data info from {}".format(data_info_file)
                    )  
        if collect_data_info:  
            print(
                "start to collect data info to {}".format(data_info_file)
                )      
            self.data_info = []
            self.load_data_par(img_folder)
            with open(data_info_file, "w") as f:
                f.write(json.dumps(self.data_info))
        # if indices_file is None or not os.path.exists(indices_file):
        # self.data_info = self.data_info[:50000]
        self.data_info = self.data_info[:5000]
        self.indices = range(self.__len__())
        # else:
        #     with open(indices_file, "r") as f:
        #         self.indices = [int(s.strip()) for s in f.readlines()]
        # return
        self.do_flip = do_flip
        if self.do_flip:
            self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        # self.base = self.get_base()
        assert size
        self.size = size
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
        
        # assert (size / downscale_f).is_integer()
        # self.LR_size = int(size / downscale_f)        
        # self.pil_interpolation = False # gets reset later if incase interp_op is from pillow
        # if degradation == "bsrgan":
        #     self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

        # elif degradation == "bsrgan_light":
        #     self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)

        # else:
        #     interpolation_fn = {
        #     "cv_nearest": cv2.INTER_NEAREST,
        #     "cv_bilinear": cv2.INTER_LINEAR,
        #     "cv_bicubic": cv2.INTER_CUBIC,
        #     "cv_area": cv2.INTER_AREA,
        #     "cv_lanczos": cv2.INTER_LANCZOS4,
        #     "pil_nearest": PIL.Image.NEAREST,
        #     "pil_bilinear": PIL.Image.BILINEAR,
        #     "pil_bicubic": PIL.Image.BICUBIC,
        #     "pil_box": PIL.Image.BOX,
        #     "pil_hamming": PIL.Image.HAMMING,
        #     "pil_lanczos": PIL.Image.LANCZOS,
        #     }[degradation]

        #     self.pil_interpolation = degradation.startswith("pil_")

        #     if self.pil_interpolation:
        #         self.degradation_process = partial(TF.resize, size=self.LR_size, interpolation=interpolation_fn)

        #     else:
        #         self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
        #                                                                   interpolation=interpolation_fn)

    def __len__(self):
        return len(self.data_info)
        # if len(self.images):
        #     return len(self.images)
        # elif len(self.paired_data):
        #     self.ranges = []
        #     num = 0
        #     for imgs, _ in self.paired_data:
        #         num += len(imgs)
        #         self.ranges.append(num)
        #     return num
    
    def load_from_origin_data(self, folder):
        # folder = "/home/v-yukangyang/data/stable-diffusion-v2/v-yukangyang/data/laion_aesthetics/laion_aesthetics_6.25+"
        # folder = "/home/v-yukangyang/data/stable-diffusion-v2/v-yukangyang/data/data/nl/output_part-00000/"
        valid_num = 0
        store_processed_folder = os.path.join("./data", os.path.basename(folder))
        if not os.path.exists(store_processed_folder):
            os.makedirs(store_processed_folder)
        text_list = []
        image_list = []
        image_path = os.path.join(store_processed_folder, "images.npy")
        text_path = os.path.join(store_processed_folder, "prompts.txt")
        if not os.path.exists(image_path) or not os.path.exists(text_path):
            for file_name in glob(folder + "/*"):
                if file_name.endswith(".parquet"):
                    data = pd.read_parquet(file_name)
                # elif file_name.endswith(".tsv"):
                #     data = pd.read_csv(file_name,sep='\t')
                else:
                    continue
                for idx, row in tqdm(data.iterrows()):
                    try:
                        img=  row.URL #IMAGEPATH  #assumes that the df has the column IMAGEPATH
                        txt = row.TEXT
                        image = io.imread(img)
                    except:
                        continue
                    if len(image.shape) == 2:
                        image = Image.fromarray(image)
                        image = image.convert("RGB")
                        image = np.array(image).astype(np.uint8)
                    image_list.append(image)
                    text_list.append(txt)
                    valid_num += 1
                    # if idx == 128:
                    #     break
            del data
            self.images = np.array(image_list)
            self.texts = text_list
            with open(text_path, "w") as f:
                f.writelines([text + "\n" for text in text_list])
            np.save(image_path, self.images)
        else:
            self.images = np.load(image_path,allow_pickle=True)
            with open(text_path, "r") as f:
                self.texts = [line.rstrip() for line in f.readlines()]
    
    def load_from_tsv(self, image_path, original_data):
        idx_list = []
        with open(image_path, "r") as f:
            for line_ in tqdm(f.readlines()):
                list_ = line_.split("\t")
                # if not list_[1].startswith("/"):
                #     continue
                img = list_[1]
                idx = int(list_[0])
                idx_list.append(idx)
                code_ = base64.b64decode(img) #.decode()
                image = Image.open(BytesIO(code_)).convert("RGB")
                image = np.array(image).astype(np.uint8)
                text = original_data.iloc[idx].TEXT
                self.images.append(image)
                self.texts.append(text)
    
    def load_data(self, img_folder, caption_folder):
        # par_data = pd.read_parquet(caption_folder) # faster
        for subfolder in glob(img_folder + "/*"):
            if os.path.isdir(subfolder):
                caption_path = os.path.join(
                    caption_folder,
                    os.path.basename(subfolder).lstrip("output_") + ".parquet"
                    )
                par_data = pd.read_parquet(caption_path) # faster
                # num_items = par_data.num_rows
                imgstr_list = []
                # for img_file in glob(subfolder + "/*.tsv"):
                #     # self.load_from_tsv(img_file, par_data) 
                #     with open(img_file, "r") as f:
                #         imgstr_list.extend(f.readlines())
                tsv_paths = glob(subfolder + "/*.tsv")
                def merge_(item):
                    imgstr_list.extend(item)
                def load_(path):
                    with open(path, "r") as f:
                        return f.readlines()
                p = mp.Pool(30)
                p.map_async(load_, tsv_paths, callback=merge_)
                p.close()
                p.join()
                self.paired_data.append([imgstr_list, par_data])
                del par_data

    def load_from_parquet(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        # rows = df.num_rows
        print(parquet_path + " is successfully loaded")
        valid_inds = list(df[df.jpg.notnull()].index)
        info_lists = list(zip(
            [parquet_path] * len(valid_inds), # image path
            [parquet_path] * len(valid_inds), # text path
            valid_inds
        ))
        # return (list(df.jpg), list(df.caption))
        # return (parquet_path, len(df))
        return info_lists
        # self.images = []
        # self.texts = []
        # for idx in range(rows):
        #     img = df.iloc[idx].jpg
        #     if img:
        #         image = Image.open(BytesIO(img)).convert("RGB")
        #         image = np.array(image).astype(np.uint8)
        #         text = df.iloc[idx].caption
        #         self.images.append(image)
        #         self.texts.append(text)
        # del df
    def merge_data(self, items):
        for item in items:
            #  self.images.extend(item[0])
            #  self.texts.extend(item[1])
            self.data_info.extend(item)
        # self.data_info.update(dict(items))
        

    def load_data_par(self, folder):
        # if isinstance(folder, list) or isinstance(folder, omegaconf.listconfig.ListConfig):
        #     parquet_paths = []
        #     for f_ in folder:
        #         parquet_paths.extend(glob(f_ + "/*.parquet")) 
        # else:
        #     parquet_paths = glob(folder + "/*.parquet")
        parquet_paths = []
        for root, _, files in os.walk(os.path.abspath(folder)):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_paths.append(os.path.join(root, file))
        parquet_paths = glob(folder + "/*/*.parquet")
        # parquet_paths = parquet_paths[:40]
        # for parquet_path in tqdm(parquet_paths): 
        #     df = pd.read_parquet(parquet_path)
        #     # self.images.extend(list(df.jpg))
        #     # self.texts.extend(list(df.caption))
        #     del df
        bs = 20
        iterables = [
            parquet_paths[i:i + bs] for i in range(0, len(parquet_paths), bs)
            ]
        # results = p.map_async(read_imgs,
            # ["/home/v-yukangyang/data/stable-diffusion-v2/v-yukangyang/data/data/000001.tsv", "/home/v-yukangyang/data/stable-diffusion-v2/v-yukangyang/data/data/000000.tsv"])   
        for iterable_ in tqdm(iterables):
            p = mp.Pool(20) 
            p.map_async(self.load_from_parquet, iterable_, callback=self.merge_data)
            # p.map_async(pd.read_parquet, iterable_)
            # p.join()
            p.close()
            p.join()
            # time.sleep(2)

    # def collect_par_info():

    def __getitem__(self, i):
        example = dict()
        # example[self.first_stage_key] = np.random.randn(self.size, self.size, 3)
        # example[self.cond_stage_key] = "diffusion model"
        # return example
        # example = self.base[i]
        # # open image file
        # image = Image.open(example["file_path_"])
        index_ = self.indices[i]
        imgfile_name, textfile_name, file_idx = self.data_info[index_]
        imgfile_name = imgfile_name.replace("/scratch", self.root)
        textfile_name = textfile_name.replace("/scratch", self.root)
        pre_t = time.time()
        if imgfile_name.endswith(".parquet"):
            df = pd.read_parquet(imgfile_name)
            # print("get image byte", time.time() - pre_t)
            img = df.jpg.iloc[file_idx]
            # print("get image byte", time.time() - pre_t)
        elif imgfile_name.endswith(".tsv"):
            with open(imgfile_name, "r") as f:
                line_ = f.readlines()[file_idx]
                file_idx, img = line_.split("\t")
                img = base64.b64decode(img)
                file_idx = int(file_idx)
        try:
            image = Image.open(BytesIO(img)).convert("RGB")
            image = np.array(image).astype(np.uint8)
            # print("image load", time.time() - pre_t)
        except:
            return self.__getitem__(np.random.randint(0, len(self.indices)))      
        # if isinstance()
        # if self.images:
        #     img = self.images[index_]
        #     try:
        #         image = Image.open(BytesIO(img)).convert("RGB")
        #         image = np.array(image).astype(np.uint8)
        #     except:
        #         return self.__getitem__(np.random.randint(0, len(self.indices)))
        #     example[self.cond_stage_key] = self.texts[index_]
        # elif self.paired_data:
        #     sec_ = bisect_right(self.ranges, index_)
        #     imgs, texts = self.paired_data[sec_]
        #     index_sec = index_ - self.ranges[sec_-1] if sec_ != 0 else index_
        #     line_ = imgs[index_sec].strip()
        #     list_ = line_.split("\t")
        #     img = list_[1]
        #     idx_ = int(list_[0])
        #     try:
        #         code_ = base64.b64decode(img) #.decode()
        #         image = Image.open(BytesIO(code_)).convert("RGB")
        #         image = np.array(image).astype(np.uint8)
        #     except:
        #         return self.__getitem__(np.random.randint(0, len(self.indices)))                
        #     example[self.cond_stage_key] = texts[idx_]
        # if len(image.shape) == 2:
        #     image = Image.fromarray(image)
        #     # if not image.mode == "RGB":
        #     image = image.convert("RGB")
        #     image = np.array(image).astype(np.uint8)
        if image.shape[0] < self.size or image.shape[1] < self.size:
            return self.__getitem__(np.random.randint(0, len(self.indices)))
        # random crop
        min_side_len = min(image.shape[:2])
        # if min_side_len == 0:
        #     return self.__getitem__(np.random.randint(0, len(self.indices)))
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)
        if self.center_crop:
            self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)
        else:
            self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)
        image = self.cropper(image=image)["image"] # ?
        # if min(image.shape[:2]) == 0:
        #     aa = 1
        # rescale
        image = self.image_rescaler(image=image)["image"]
        # flip
        if self.do_flip:
            image = self.flip(Image.fromarray(image))
            image = np.array(image).astype(np.uint8)
        # # degradation to get the low resolution images
        # if self.pil_interpolation:
        #     image_pil = PIL.Image.fromarray(image)
        #     LR_image = self.degradation_process(image_pil)
        #     LR_image = np.array(LR_image).astype(np.uint8)
        # else:
        #     LR_image = self.degradation_process(image=image)["image"]
        # # store to example
        # example["image"] = (image/127.5 - 1.0).astype(np.float32) #[-1, 1]
        # example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float32) #[-1, 1]
        example[self.first_stage_key] = (image/127.5 - 1.0).astype(np.float32)
        # print("image", time.time() - pre_t)
        pre_t = time.time()
        if imgfile_name != textfile_name:
            if textfile_name.endswith(".parquet"):
                df = pd.read_parquet(textfile_name)
            else:
                print(
                    "the format {} of the text file is not supported".format(
                        os.path.splitext(imgfile_name)[1]
                        )
                    )
                raise ValueError
        try:
            text = df.TEXT.iloc[file_idx]
        except:
            try:
                text = df.caption.iloc[file_idx]
            except:
                raise ValueError
        example[self.cond_stage_key] = text 
        # Sprint("text (text load)", time.time() - pre_t)   
        return example


class LAIONTrain(LAIONBase):
    def __init__(self, store_folder, *args,  ratio=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        # store_folder = os.path.join("./data", os.path.basename(folder))
        if not os.path.exists(os.path.join(store_folder, "train.txt")):
            rand_inds = np.random.permutation(self.__len__())
            self.indices = rand_inds[:int(len(rand_inds) * ratio)]
            rand_inds = [str(i) + "\n" for i in rand_inds]
            with open(os.path.join(store_folder, "train.txt"), "w") as f:
                f.writelines(rand_inds[:int(len(rand_inds) * ratio)])
            with open(os.path.join(store_folder, "val.txt"), "w") as f:
                f.writelines(rand_inds[int(len(rand_inds) * ratio):])
        else:
            with open(os.path.join(store_folder, "train.txt"), "r") as f:
                self.indices = [int(s.strip()) for s in f.readlines()]

    # def get_base(self):
    #     with open("data/imagenet_train_hr_indices.p", "rb") as f:
    #         indices = pickle.load(f)
    #     dset = ImageNetTrain(process_images=False,)
    #     return Subset(dset, indices)                
            
class LAIONValidation(LAIONBase):
    def __init__(self, store_folder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # store_folder = os.path.join("./data", os.path.basename(folder))
        if not os.path.exists(os.path.join(store_folder, "val.txt")):
            raise ValueError
        else:
            with open(os.path.join(store_folder, "val.txt"), "r") as f:
                self.indices = [int(s.strip()) for s in f.readlines()]

    # def get_base(self):
    #     with open("data/imagenet_val_hr_indices.p", "rb") as f:
    #         indices = pickle.load(f)
    #     dset = ImageNetValidation(process_images=False,)
    #     return Subset(dset, indices)
class LAIONIterableBaseDataset(Txt2ImgIterableBaseDataset):
    '''
    Define an interface to make the IterableDatasets for text2img data chainable
    '''
    def __init__(self, img_folder, caption_folder=None, size=256,
                 first_stage_key = "jpg", cond_stage_key = "txt", do_flip = False,
                 min_crop_f=0.5, max_crop_f=1., flip_p=0.5,
                 random_crop=True):
        assert size
        super().__init__(size=size)
        self.caption_folder = caption_folder
        if self.caption_folder:
            # self.origin_folders = glob(img_folder + "/*/") # "output_part_000000"
            self.valid_ids = glob(img_folder + "/*/")
            self.origin_tsv_paths = {
                subfolder: glob(subfolder + "/*.tsv") for subfolder in self.valid_ids #self.origin_folders
            }
            num = 0
            self.tsv_folder_idx = {}
            # self.tsv_nums = []
            for key, value in self.origin_tsv_paths.items():
                num += len(value)
                self.tsv_folder_idx[num] = key
                # self.tsv_nums.append(num)
            # self.num_records = len(self.origin_folders)
            # self.folders = self.origin_folders
            self.num_records = len(self.valid_ids)
            self.sample_ids = self.valid_ids
            self.tsv_paths = self.origin_tsv_paths # to be deprecated
            self.max_num = self.num_records * 100000
        else:
            parquet_paths = []
            for root, _, files in os.walk(os.path.abspath(img_folder)):
                for file in files:
                    if file.endswith(".parquet"):
                        parquet_paths.append(os.path.join(root, file))
            parquet_paths = parquet_paths[:170]
            # self.origin_parquet_paths = parquet_paths
            # self.parquet_paths = self.origin_parquet_paths
            # self.num_records = len(parquet_paths) 
            self.valid_ids = parquet_paths
            self.sample_ids = self.valid_ids
            self.num_records = len(self.valid_ids) 
            self.max_num = self.num_records * 1000
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key

        # self.num_records = len(self.folders)
        # self.num_records = np.sum([
        #     len(value_) for value_ in self.tsv_paths.values()
        # ])

        self.do_flip = do_flip
        if self.do_flip:
            self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        # self.base = self.get_base()
    
        # self.size = size
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
        
        # self.num_records = num_records
        # self.valid_ids = valid_ids
        # print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    # def __len__(self):
    #     # return self.num_records 
    #     return self.max_num
    
    def __iter__(self):
        if self.caption_folder:
            return self.parquet_tsv_iter()
        else:
            return self.parquet_iter()

    def parquet_iter(self):
        print("this shard on GPU {}: {}".format(_get_distributed_settings()[1], len(self.sample_ids)))
        idx = 0
        while idx >= 0:
            for parqut_path in self.sample_ids: #parquet_paths:
                df = pd.read_parquet(parqut_path)
                for file_idx in range(len(df)):
                    img_code = df.jpg.iloc[file_idx]
                    if img_code:
                        try:
                            image = self.generate_img(img_code)
                        except:
                            # print("can' t open")
                            continue
                        if image is None:
                            continue
                        # except:
                        #     continue
                        try:
                            text = df.caption.iloc[file_idx]
                        except:
                            try:
                                text = df.TEXT.iloc[file_idx]
                            except:
                                continue
                        if text is None:
                            continue
                        example = {}
                        example[self.first_stage_key] = image
                        example[self.cond_stage_key] = text
                        yield example
                del df
            print("has gone over the whole dataset, need to start next round")
            idx += 1

    def parquet_tsv_iter(self):
        for subfolder in self.sample_ids: #folders:
            caption_path = os.path.join(
                self.caption_folder,
                os.path.basename(subfolder).lstrip("output_") + ".parquet"
                )
            par_data = pd.read_parquet(caption_path) # faster
            for image_path in self.tsv_paths[subfolder]:
                with open(image_path, "r") as f:
                    for line_ in tqdm(f.readlines()): # shuffle could be done
                        # line_ = f.readline()
                        idx, img_code = line_.split("\t")
                        # if not list_[1].startswith("/"):
                        #     continue
                        try:
                            img_code = base64.b64decode(img_code) #.decode()
                            image = self.generate_img(img_code)
                            if not image:
                                continue
                        except:
                            continue
                        example = dict()
                        example[self.first_stage_key] = image
                        idx = int(idx)
                        text = par_data.iloc[idx].TEXT
                        example[self.cond_stage_key] = text
                        yield example
            del par_data
    
    def generate_img(self, img_code):
        image = Image.open(BytesIO(img_code)).convert("RGB")
        image = np.array(image).astype(np.uint8)
        if image.shape[0] < self.size or image.shape[1] < self.size:
            return None
        # crop
        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)
        if self.center_crop:
            self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)
        else:
            self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)
        image = self.cropper(image=image)["image"] # ?
        # rescale
        image = self.image_rescaler(image=image)["image"]
        # flip
        if self.do_flip:
            image = self.flip(Image.fromarray(image))
            image = np.array(image).astype(np.uint8)
        return (image/127.5 - 1.0).astype(np.float32)
        # pass
