#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset

try:
    from torchvision.transforms import v2 as transforms
except ImportError:
    from torchvision import transforms

class MultimodalDataset(Dataset):
    """
    This class is used to train models that deal with multimodal data (e.g., images, joints), such as CNNRNN/SARNN.

    Args:
        images (numpy array): Set of images in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        joints (numpy array): Set of joints in the dataset, expected to be a 3D array [data_num, seq_num, joint_dim].
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, images_l, images_r, joints, device="cpu", stdev=None):#training=True
        """
        The constructor of Multimodal Dataset class. Initializes the images, joints, and transformation.

        Args:
            images (numpy array): The images data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            joints (numpy array): The joints data, expected to be a 3D array [data_num, seq_num, joint_dim].
            stdev (float, optional): The standard deviation for the normal distribution to generate noise. Defaults to 0.02.
        """
        self.stdev = stdev
        self.device = device
        self.images_l = torch.Tensor(images_l).to(self.device)
        self.images_r = torch.Tensor(images_r).to(self.device)
        self.joints = torch.Tensor(joints).to(self.device)
        self.transform = nn.Sequential(
            transforms.RandomErasing(),
            transforms.ColorJitter(brightness=0.4),
            transforms.ColorJitter(contrast=[0.6, 1.4]),
            transforms.ColorJitter(hue=[0.0, 0.04]),
            transforms.ColorJitter(saturation=[0.6, 1.4]),
        ).to(self.device)

        

    def __len__(self):
        """
        Returns the number of the data.
        """
        return len(self.images_l)

    def __getitem__(self, idx):
        """
        Extraction and preprocessing of images and joints at the specified indexes.

        Args:
            idx (int): The index of the element.

        Returns:
            dataset (list): A list containing lists of transformed and noise added image and joint (x_img, x_joint) and the original image and joint (y_img, y_joint).
        """

        x_img_l = self.images_l[idx]
        x_img_r = self.images_r[idx]
        x_joint = self.joints[idx]
        y_img_l = self.images_l[idx]
        y_img_r = self.images_r[idx]
        y_joint = self.joints[idx]

        if self.stdev is not None:
            x_img_l = self.transform(y_img_l) + torch.normal(
                mean=0, std=0.02, size=x_img_l.shape, device=self.device
            )
            x_img_r = self.transform(y_img_r) + torch.normal(
                mean=0, std=0.02, size=x_img_r.shape, device=self.device
            ) 
            x_joint = y_joint + torch.normal(
                mean=0, std=self.stdev, size=y_joint.shape, device=self.device
            )

        return [[x_img_l, x_img_r, x_joint], [y_img_l, y_img_r,y_joint]]
