import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score
from PIL import Image
import numpy as np
import os

# Load pretrained Inception-v3
# inception_model = torchvision.models.inception_v3(pretrained=True).to(torch.device('cuda:0'))


def calc_FID(gt_path, gen_path):
    fid_value = fid_score.calculate_fid_given_paths([gt_path, gen_path],
                                                    batch_size=1024,
                                                    device=torch.device('cuda:1'),
                                                    dims=2048)  # 2048,768,192,64
    print('FID value:', fid_value)


    ref_path = os.path.join(gt_path, 'ref_imgs.npz')  
    
    if not os.path.exists(ref_path):
        arr = []
        filenames = [l for l in os.listdir(gt_path) if l.endswith('.png') or l.endswith('.jpg')]
        print(f'Number of imgs: {len(filenames)}')
        for filename in filenames:
            filepath = os.path.join(gt_path, filename)
            with Image.open(filepath) as img:
                image_array = np.expand_dims(np.array(img), axis=0)
                arr.append(image_array)
        
        arr = np.concatenate(arr, axis=0)
        np.savez(ref_path, arr)


    np_imgs_path = os.path.join(gen_path, 'gen_imgs.npz')  
    
    if not os.path.exists(np_imgs_path):
        arr = []
        filenames = [l for l in os.listdir(gen_path) if l.endswith('.png') or l.endswith('.jpg')]
        print(f'Number of imgs: {len(filenames)}')
        for filename in filenames:
            filepath = os.path.join(gen_path, filename)
            with Image.open(filepath) as img:
                image_array = np.expand_dims(np.array(img), axis=0)
                arr.append(image_array)
        
        arr = np.concatenate(arr, axis=0)
        np.savez(np_imgs_path, arr)
    
    return fid_value


calc_FID(gt_path='',
         gen_path='')
