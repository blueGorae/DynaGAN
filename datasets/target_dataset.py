from torch.utils.data import Dataset
from PIL import Image
import PIL
from utils import data_utils
import torchvision.transforms as transforms
import os
from utils.shape_predictor import align_face
import sys
import torch


class TargetDataset(Dataset):

    def __init__(self, ZP_target_latent, ZP_img_tensor, ZP_img_tensor_256):

        self.ZP_target_latent = ZP_target_latent
        self.ZP_img_tensor = ZP_img_tensor
        self.ZP_img_tensor_256 = ZP_img_tensor_256

    def __len__(self):
        return len(self.ZP_target_latent)

    def __getitem__(self, index):
        
        ZP_target_latent = self.ZP_target_latent[index]
        ZP_img_tensor = self.ZP_img_tensor[index]
        ZP_img_tensor_256 = self.ZP_img_tensor_256[index]

        return ZP_target_latent, ZP_img_tensor, ZP_img_tensor_256



