
import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

class COCOSegmentation(data.Dataset):
    CLASSES = 91  # COCO has 80 classes + background

    def __init__(self, path, split='train', transform=None, target_transform=None):
        self.root = path
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        ann_file = os.path.join(self.root, f'annotations/instances_{self.split}2017.json')
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Load image
        img_metadata = coco.loadImgs(img_id)[0]
        path = os.path.join(self.root, 'images', self.split + '2017', img_metadata['file_name'])
        img = Image.open(path).convert('RGB')
        
        # Create binary mask
        mask = np.zeros((img_metadata['height'], img_metadata['width']))
        for ann in anns:
            mask = np.maximum(mask, coco.annToMask(ann))
        
        mask = Image.fromarray(mask.astype(np.uint8) * 255)
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            mask = self.target_transform(mask)
            mask = np.array(mask).astype('int32')
            mask = torch.from_numpy(mask).long()
        
        return img, mask

    def __len__(self):
        return len(self.ids)



# from pycocotools.coco import COCO
# import torch
# import os
# import torch.utils.data as data
# import torch
# from torchvision import transforms
# from PIL import Image
# import numpy as np





# class COCOSegmentation:
#     def __init__(self, path, split='val', transform=None, target_transform=None):
#         self.root = path
#         self.split = split
#         self.transform = transform
#         self.target_transform = target_transform
        
#         self.image_dir = os.path.join(self.root, self.split, 'images')
#         self.mask_dir = os.path.join(self.root, self.split, 'masks')
        
#         self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg') or f.endswith('.jpeg')])
#         self.masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.png')])
        
#         assert len(self.images) == len(self.masks), "Number of images and masks should be the same"

#     def __getitem__(self, index):
#         img_name = self.images[index]
#         mask_name = self.masks[index]

#         # Load image
#         img_path = os.path.join(self.image_dir, img_name)
#         img = Image.open(img_path).convert('RGB')

#         # Load mask
#         mask_path = os.path.join(self.mask_dir, mask_name)
#         mask = Image.open(mask_path).convert('L')  # Convert to grayscale

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             mask = self.target_transform(mask)

#         # Convert mask to tensor and then to long type
#         mask = torch.from_numpy(np.array(mask)).long()

#         return img, mask

#     def __len__(self):
#         return len(self.images)

