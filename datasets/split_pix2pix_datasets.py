from glob import glob
import os
import numpy as np
import natsort
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm



class Pix2Pix_Dataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.transform_1 = transforms.Compose([transforms.ToTensor()])

        self.transform_2 = transforms.Compose([transforms.Resize(64),
                                            transforms.RandomHorizontalFlip(p=0.0),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ])
        all_imgs = os.listdir(main_dir)
        self.all_img_paths = natsort.natsorted(all_imgs)
        self.y0 = torch.nn.functional.one_hot(torch.zeros(len(self.all_img_paths)).long(),num_classes=2)
        self.y1 = torch.nn.functional.one_hot(torch.ones(len(self.all_img_paths)).long(),num_classes=2)
        

    def __len__(self):
        return len(self.all_img_paths)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_img_paths[idx])

        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform_1(image)
        imageA = tensor_image[:,:,:256]
        imageB = tensor_image[:,:,256:]
        
        
        
        return imageA, imageB, self.all_img_paths[idx]



# save_train_dir = ''
# train_dataset = Pix2Pix_Dataset('')
# train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4)

# for data in tqdm(train_dataloader):
#     imageA, imageB, img_name = data
#     save_image(imageA[0], os.path.join(save_train_dir, 'A', img_name[0]))
    
#     save_image(imageB[0], os.path.join(save_train_dir, 'B', img_name[0]))
    

save_val_dir = ''
val_dataset = Pix2Pix_Dataset('')
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4)

for data in tqdm(val_dataloader):
    imageA, imageB, img_name = data
    save_image(imageA[0], os.path.join(save_val_dir, 'A', img_name[0]))
    
    save_image(imageB[0], os.path.join(save_val_dir, 'B', img_name[0]))