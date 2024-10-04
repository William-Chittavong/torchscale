

# import os
# import torch
# import torch.utils.data as data
# import numpy as np
# from pycocotools.coco import COCO
# from PIL import Image
# import torchvision.transforms as transforms

# class COCOSegmentation(data.Dataset):
#     def __init__(self, root, annFile, transform=None, target_transform=None):
#         self.root = root
#         self.coco = COCO(annFile)
#         self.transform = transform
#         self.target_transform = target_transform
#         self.ids = list(self.coco.imgs.keys())

#     def __getitem__(self, index):
#         coco = self.coco
#         img_id = self.ids[index]
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         anns = coco.loadAnns(ann_ids)
        
#         # Load image
#         img_info = coco.loadImgs(img_id)[0]
#         path = os.path.join(self.root, img_info['file_name'])
#         img = Image.open(path).convert('RGB')

#         # Create empty mask
#         mask = np.zeros((img_info['height'], img_info['width']))

#         # Fill mask with instance segmentations
#         for ann in anns:
#             pixel_value = ann['category_id']
#             mask = np.maximum(coco.annToMask(ann) * pixel_value, mask)

#         # Convert mask to PIL Image
#         mask = Image.fromarray(mask.astype(np.uint8))

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             mask = self.target_transform(mask)
#         else:
#             mask = transforms.ToTensor()(mask)
#             mask = mask.long()

#         return img, mask

#     def __len__(self):
#         return len(self.ids)

#     @property
#     def classes(self):
#         return len(self.coco.cats)

import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image
import torch

class COCOSegmentation:
    def __init__(self, root, split='val2017', transform=None, target_transform=None):
        ann_file = f'{root}/annotations/instances_{split}.json'
        self.root = f'{root}/{split}'
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Load image
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(f'{self.root}/{path}').convert('RGB')
        
        # Create binary mask
    
        
        mask = 0
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])>0
        # Convert mask to PIL Image
        mask = Image.fromarray(mask)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask

    def __len__(self):
        return len(self.ids)
