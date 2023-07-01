import torch
import numpy as np


def preproc(im_array):
    gray_array = np.mean(im_array, axis=2)
    tensor_im = torch.tensor(gray_array, dtype=torch.float32)
    tensor_im = tensor_im.unsqueeze(0).unsqueeze(0)
    cell_size = 20
    stride = cell_size
    padding = 0
    dilation = True
    kernel_size = cell_size
    tensor = torch.nn.functional.avg_pool2d(tensor_im, kernel_size, stride, padding, dilation)
    tensor = torch.rot90(tensor, 1, [2, 3])
    tensor = torch.flip(tensor, [2])
    tensor = 254 - tensor
    return tensor
