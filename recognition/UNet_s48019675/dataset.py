from collections import defaultdict

import numpy as np
import utils
import torch
import os
import re
import nibabel as nib


# resize and focus on the info, maybe use attention net

# return data for model
root2d = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
MRI3d = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
Segments3d = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

testNii = "C:/Users/hawki/OneDrive/Desktop/Uni/2026/Comp_Patterns/Report/seg_042_week_0_slice_9.nii"

def get_2d_data(path, truth = False):
    if truth == True:
        pattern = re.compile(r"seg_(\d+)_week_(\d+)_slice_(\d+).nii")
    else:
        pattern = re.compile(r"case_(\d+)_week_(\d+)_slice_(\d+).nii")
    
    volumes = defaultdict(list)

    for file in os.listdir(path):
        match = pattern.match(file)
        if match:
            patient_id = int(match.group(1))
            week = int(match.group(2))
            slice_num = int(match.group(3))

            volumes[(patient_id, week)].append((slice_num, os.path.join(path, file)))

    for key in volumes:
        volumes[key].sort(key=lambda x: x[0])

    images = {}

    for key, slices in volumes.items():
        image_name = [slice_path for _, slice_path in slices]

    if truth:
        images[key] = utils.load_data_2D(image_name, normImage=False, categorical=True, dtype=np.uint8)
    else:
        images[key] = utils.load_data_2D(image_name, normImage=True)

    return images

def get_training_2d():
    loc = root2d + "/keras_slices_train"
    return get_2d_data(loc)
    
def get_validation_2d():
    loc = root2d + "/keras_slices_validate"
    return get_2d_data(loc)

def get_testing_2d():
    loc = root2d + "/keras_slices_test"
    return get_2d_data(loc)


def ground_train_2d():
    loc = root2d + "/keras_slices_seg_train"
    return get_2d_data(loc, truth=True)

def ground_validation_2d():
    loc = root2d + "/keras_slices_seg_validate"
    return get_2d_data(loc, truth=True)

def ground_testing_2d():
    loc = root2d + "/keras_slices_seg_test"
    return get_2d_data(loc, truth=True)

class UNet2d_Data(torch.utils.data.Dataset):
    def __init__(self):
        self.images = get_training_2d()
        self.ground_truth = ground_train_2d()

        self.keys = sorted(self.images.keys())

        self.samples = []

        for key in self.keys:
            image_slices = self.images[key]
            ground_slices = self.ground_truth[key]

            for i in range(image_slices.shape[0]):
                self.samples.append((image_slices[i], ground_slices[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, ground_truth = self.samples[idx]

        image = image.astype(np.float32)
        ground_truth = ground_truth.astype(np.float32)

        image = torch.tensor(image).unsqueeze(0)
        ground_truth = torch.tensor(ground_truth).unsqueeze(0)
        return image, ground_truth



def get_tranining_3d():
    pass
def get_validation_3d():
    pass
def get_testing_3d():
    pass
