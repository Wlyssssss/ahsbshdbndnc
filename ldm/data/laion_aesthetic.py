import os 
from glob import glob
import cv2
import albumentations
import numpy as np
from PIL import Image
import pandas as pd
from torchvision import transforms
# from skimage import io
from tqdm import tqdm
import base64
from io import BytesIO
# from ldm.data.base import Txt2ImgIterableBaseDataset
from torch.utils.data.dataloader import _get_distributed_settings
# from abc import abstractmethod
# from torch.utils.data import IterableDataset
import clip 
import subprocess
from ldm.data.base import Txt2ImgIterableBaseDataset
import tempfile
class LAIONIterableBaseDataset(Txt2ImgIterableBaseDataset):
    '''
    Load laion dataset into the IterableDatasets class
    '''
    def __init__(self, img_folder, caption_folder=None, img_txt_same_file = False, 
                 blob_folder=None, sas_token =None, 
                 max_num_records = 128, max_num_tsv_per_record = 182, tsv_patch_size = 10, start_tsv_idx=None,
                 do_azcopy=False,
                 remove_data_from_cluster=False,
                 size=256,
                 first_stage_key = "jpg", cond_stage_key = "txt", 
                 clip_model = None, preprocess = None,
                 do_flip = False, min_crop_f=0.5, max_crop_f=1., flip_p=0.5, random_crop=True):
        assert size
        super().__init__(size=size)

        self.img_folder = img_folder
        self.caption_folder = caption_folder
        self.img_txt_same_file = img_txt_same_file
        if not self.img_txt_same_file:
            # blob info
            self.blob_folder = blob_folder 
            self.sas_token = sas_token
            self.image_blob_name = os.path.basename(img_folder)
            self.caption_blob_name = os.path.basename(caption_folder)
            self.remove_data_from_cluster =  remove_data_from_cluster if do_azcopy else False
            self.do_azcopy = do_azcopy
            self.max_num_tsv_per_record = max_num_tsv_per_record
            self.tsv_patch_size = tsv_patch_size
            self.start_tsv_idx = int(self.tsv_patch_size / 2) if start_tsv_idx is None else start_tsv_idx
            if self.start_tsv_idx >= self.tsv_patch_size:# or self.start_tsv_idx < 1:
                print("wrongly set the data download time")
                raise ValueError
            if self.caption_folder:
                # try:
                if self.do_azcopy:
                # except:
                    self.valid_ids = [
                        os.path.join(img_folder, "output_part-" + "{:0>5d}".format(i)) for i in range(max_num_records)
                        ]
                    # self.valid_ids = [
                    #     os.path.join(img_folder, "output_part-" + "{:0>5d}".format(i)) for i in [4,5] #[4,5]
                    #     ]
                else:
                    self.valid_ids = [folder.rstrip("/") for folder in glob(img_folder + "/*/")]
                self.num_records = len(self.valid_ids)
                if not self.num_records:
                    print("zero data records, please check the data path")
                    raise ValueError
                self.sample_ids = self.valid_ids
                self.max_num = self.num_records * 100000 * self.max_num_tsv_per_record
            else:
                print("should provide caption folder")
                raise ValueError
        else:
            parquet_paths = []
            for root, _, files in os.walk(os.path.abspath(img_folder)):
                for file in files:
                    if file.endswith(".parquet"):
                        parquet_paths.append(os.path.join(root, file))
            # parquet_paths = parquet_paths[:32]
            # self.origin_parquet_paths = parquet_paths
            # self.parquet_paths = self.origin_parquet_paths
            # self.num_records = len(parquet_paths) 
            self.valid_ids = parquet_paths
            self.sample_ids = self.valid_ids
            self.num_records = len(self.valid_ids) 
            self.max_num = self.num_records * 1000
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key

        self.preprocess = None
        if preprocess is not None:
            self.preprocess = preprocess
        else:
            if clip_model is not None: # "ViT-L/14"
                _,  self.preprocess = clip.load(clip_model) #, device=self.device)  # RN50x64 
        
        self.do_flip = do_flip
        if self.do_flip:
            self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

    # def __len__(self):
    #     # return self.num_records 
    #     return self.max_num
    
    def __iter__(self):
        # if self.caption_folder:
        if self.img_txt_same_file:
            return self.parquet_iter()
        else:
            return self.parquet_tsv_iter()
        # else:
        #     return self.parquet_iter()

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
        print("this shard on GPU {}: {}".format(_get_distributed_settings()[1], len(self.sample_ids)))
        idx = 0
        # first_part = True
        while idx >= 0:
            for subfolder in self.sample_ids: #folders:
                parquet_name = os.path.basename(subfolder).split("output_")[1]
                caption_path = os.path.join(
                    self.caption_folder,
                    parquet_name + ".parquet"
                    )
                if self.do_azcopy:
                    tsv_paths = [
                        os.path.join(subfolder, "{:0>6d}.tsv".format(i)) for i in range(self.max_num_tsv_per_record)
                    ]
                    # tsv_paths = self.check_and_download(caption_path, tsv_paths, subfolder, parquet_name, first_part = first_part)
                    self.download_data(caption_path, tsv_paths[:self.tsv_patch_size], subfolder, parquet_name, first_part=True)
                    download_time = 1
                else:
                    tsv_paths = glob(subfolder + "/*.tsv")
                par_data = pd.read_parquet(caption_path) # faster
                # for image_path in self.tsv_paths[subfolder]:
                for rank, image_path in enumerate(tsv_paths):
                    print("start opening {}".format(image_path))
                    with open(image_path, "r") as f:
                        # for line_ in tqdm(f.readlines()): 
                        lines = f.readlines()
                    print("successfully open and read {}".format(image_path))
                    if self.remove_data_from_cluster:
                        self.remove_data(image_path)
                    if self.do_azcopy and rank == self.start_tsv_idx + (download_time-1) * self.tsv_patch_size:
                        self.download_data(
                            caption_path, 
                            tsv_paths[self.tsv_patch_size * download_time: self.tsv_patch_size * (download_time + 1)], 
                            subfolder, parquet_name, 
                            first_part=False
                            )
                        download_time += 1
                        print("download time: {}".format(download_time))
                    # for line_ in f.readlines(): 
                    for i, line_ in enumerate(lines): 
                        # print("the {}th line".format(i))
                        # line_ = f.readline()
                        idx, img_code = [str_.strip() for str_ in line_.split("\t")]
                        # if not list_[1].startswith("/"):
                        #     continue
                        try:
                            # img_code = base64.b64decode(img_code) #.decode()
                            # image = self.generate_img(img_code)
                            image = self.generate_img(base64.b64decode(img_code))
                            if image is None:
                                continue
                        except:
                            continue
                        example = dict()
                        example[self.first_stage_key] = image
                        # idx = int(idx)
                        text = par_data.iloc[int(idx)].TEXT
                        example[self.cond_stage_key] = text
                        example["data"] = "\t".join([
                            parquet_name,
                            idx, 
                            img_code, 
                            text
                        ])
                        yield example
                        # if i == 70000:
                        #     break
                del par_data
                if self.remove_data_from_cluster:
                    self.remove_data(caption_path)
                # if self.remove_data_from_cluster:
                #     self.remove_data(caption_path)
                #     self.remove_data(subfolder)
            print("has gone over the whole dataset, need to start next round")
            idx += 1
    
    def generate_img(self, img_code):
        image = Image.open(BytesIO(img_code))
        if self.preprocess:
            # pil_image = Image.open(img_path)
            image = self.preprocess(image)#.unsqueeze(0)#.to(device)
            return image
        else:
            image = image.convert("RGB")
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
    
    def check_and_download(self, caption_path, tsv_paths, subfolder, parquet_name):
        if not os.path.exists(caption_path):
            try:
                os.makedirs(self.caption_folder, exist_ok=True)
                self.azcopy_from_blob(
                    self.caption_blob_name,
                    parquet_name + ".parquet",
                    self.caption_folder,
                )
            except:
                print("fail to download caption file from blob!")
                raise ValueError                
        if not len(tsv_paths):
            try:
                os.makedirs(self.img_folder, exist_ok=True)
                self.azcopy_from_blob(
                    self.image_blob_name,
                    os.path.basename(subfolder),
                    self.img_folder,
                ) 
                return glob(subfolder + "/*.tsv")
            except:
                print("fail to download image tsv file from blob!")
                raise ValueError
        return tsv_paths

    def download_data(self, caption_path, tsv_paths, subfolder, parquet_name, first_part=True):
        if not os.path.exists(caption_path) and first_part:
            try:
                os.makedirs(self.caption_folder, exist_ok=True)
                self.azcopy_from_blob(
                    self.caption_blob_name,
                    parquet_name + ".parquet",
                    self.caption_folder,
                    first_part=first_part,
                )
            except:
                print("fail to download caption file from blob!")
                raise ValueError 
        os.makedirs(subfolder, exist_ok=True)               
        for tsv_path in tsv_paths:
            if not os.path.exists(tsv_path):
                try:    
                    self.azcopy_from_blob(
                        self.image_blob_name,
                        os.path.join(os.path.basename(subfolder), os.path.basename(tsv_path)),
                        subfolder,
                        first_part=first_part,
                    ) 
                    # return glob(subfolder + "/*.tsv")
                except:
                    print("fail to download image tsv file from blob to {}!".format(tsv_path))
                    raise ValueError
        # return tsv_paths

    def azcopy_from_blob(self, subfolder = "laion-5b", name = "output_part-00005", destination = "/scratch", first_part=True):
        command = 'sudo azcopy cp '
        if self.blob_folder is None:
            print("The blob storage for laion data is not provided!")
            raise ValueError
        if self.sas_token is None:
            print("The sas token for laion data is not provided!")
            raise ValueError
        file = self.blob_folder + "/" + subfolder + "/" + name
        # file = "https://itpsea4data.blob.core.windows.net/v-yukangyang/data/data/laion-5b/output_part-00005"
        # sas_token = "?sv=2021-08-06&st=2023-01-05T06%3A47%3A56Z&se=2023-01-11T06%3A47%3A00Z&sr=c&sp=racwl&sig=aAHHp4NhaVWuR7lnhT8GJqZicWvbQia%2FflKmoly4x0A%3D"
        # sas_token = "?sv=2021-08-06&st=2023-01-05T06%3A17%3A31Z&se=2023-01-06T06%3A17%3A31Z&sr=c&sp=raccl&sig=0gRoqwgEqeDzZHchhduf9N9jVHLzAnX5iPC%2FOb%2F%2Bk9Q%3D"
        # destination = "/scratch" 
        # sas_token = "?sv=2021-08-06&st=2023-01-05T06%3A17%3A31Z&se=2023-01-06T06%3A17%3A31Z&sr=c&sp=raccl&sig=0gRoqwgEqeDzZHchhduf9N9jVHLzAnX5iPC%2FOb%2F%2Bk9Q%3D"
        # file_str = '"' + file + self.sas_token + '"' 
        file_str = file + self.sas_token
        command_line = command + file_str + ' ' + destination + ' --recursive'
        command_list = command_line.split(" ")
        if first_part:
            subprocess.call(
                command_list
            )
            print("azcopy {} successfully!".format(file))
        else:
            # os.popen(command_line)

            # out_temp = tempfile.SpooledTemporaryFile(bufsize=10*1000)
            # with tempfile.SpooledTemporaryFile() as out_temp:
            #     fileno = out_temp.fileno()
            #     p = subprocess.Popen(command_list, stdout=fileno, stderr=fileno, close_fds=True, shell=True)
            #     p.communicate()
            # p = subprocess.Popen(command_list, close_fds=True, shell=True)
            # p.communicate()
            # p = subprocess.Popen(command_list, close_fds=True)
            # p.communicate()
            subprocess.Popen(command_list)
            # p = subprocess.Popen(command_list, close_fds=True, stdout=subprocess.PIPE)
            print("start downloading {}".format(file))
            # for line in iter(p.stdout.readline, b''):
            #     print(line)     
            # # print to stdout immediately
            # p.stdout.close()
        
    def remove_data(self, file =  "/scratch/output_part-00005"):
        command = "sudo rm -rf "
        command_list = (command + file).split(" ")
        subprocess.call(
            command_list
        )
        print("remove {} from the cluster successfully!".format(file))

