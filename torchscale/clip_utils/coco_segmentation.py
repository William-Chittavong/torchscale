import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import pycocotools.mask as mask_util  # Import mask utilities
import torchvision.transforms as T


class COCOSegmentation(data.Dataset):
    CLASSES = 2  # Example, change according to the number of classes in the dataset

    def __init__(self, root, split='val2017', transform=None, target_transform=None):
        """
        Args:
            root (str): Root directory where the COCO dataset is stored.
            split (str): The dataset split to use (e.g., 'train2017', 'val2017').
            transform (callable, optional): A function/transform to apply to the images.
            target_transform (callable, optional): A function/transform to apply to the segmentation masks.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Construct the path to the annotation file based on the root and split
        ann_file = os.path.join(root, 'annotations', f'instances_{split}.json')
        self.coco = COCO(ann_file)  # Initialize COCO object with the annotation file
        self.ids = list(self.coco.imgs.keys())  # List of image IDs in the COCO dataset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the item.
        Returns:
            tuple: (image, target), where target is the segmentation mask.
        """
        # Get image info and load the image
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img_path = os.path.join(self.root, self.split, path)
        img = Image.open(img_path).convert('RGB')

        # Load and create the segmentation mask
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        for ann in anns:
            mask = self.coco.annToMask(ann)  # Convert segmentation to RLE format
               

        mask = Image.fromarray(mask)

        # Apply transformations
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)
            mask = torch.from_numpy(np.array(mask)).long()  # Convert to tensor

        return img, mask

    def __len__(self):
        return len(self.ids)



# import os
# import torch
# import numpy as np
# from PIL import Image
# from pycocotools.coco import COCO
# from pycocotools import mask as coco_mask

# class COCOSegmentation(torch.utils.data.Dataset):
#     def __init__(self, root, split='val2017', transform=None, target_transform=None):
#         self.root = root
#         self.split = split
#         self.transform = transform
#         self.target_transform = target_transform
        
#         ann_file = os.path.join(root, 'annotations', f'instances_{split}.json')
#         self.coco = COCO(ann_file)
#         self.ids = list(self.coco.imgs.keys())

#     def __getitem__(self, index):
#         coco = self.coco
#         img_id = self.ids[index]
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         anns = coco.loadAnns(ann_ids)

#         img_info = coco.loadImgs(img_id)[0]
        
#         img_path = os.path.join(self.root, self.split, img_info['file_name'])
        
#         img = Image.open(img_path).convert('RGB')

#         # Create a binary mask
#         mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
#         for ann in anns:
#             mask = np.maximum(mask, coco_mask.decode(ann['segmentation']))

#         # Convert mask to PIL Image
#         mask = Image.fromarray(mask)

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             mask = self.target_transform(mask)

#         # Convert mask to tensor
#         mask = torch.from_numpy(np.array(mask)).long()

#         return img, mask

#     def __len__(self):
#         return len(self.ids)
# import os
# import torch
# import torch.utils.data as data
# import numpy as np
# from PIL import Image
# from pycocotools.coco import COCO



# class COCOSegmentation(torch.utils.data.Dataset):
#     def __init__(self, root, split='train2017', transform=None):
#         self.root = root
#         self.split = split
#         self.transform = transform
        
#         ann_file = os.path.join(root, 'annotations', f'instances_{split}.json')
#         self.coco = COCO(ann_file)
#         self.ids = list(self.coco.imgs.keys())

#     def __getitem__(self, index):
#         coco = self.coco
#         img_id = self.ids[index]
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         target = coco.loadAnns(ann_ids)

#         img_info = coco.loadImgs(img_id)[0]
        
#         # Construct the correct path to the image
#         img_path = os.path.join(self.root, self.split, img_info['file_name'])
        
#         img = Image.open(img_path).convert('RGB')

#         if self.transform is not None:
#             img = self.transform(img)

#         return img, target

#     def __len__(self):
#         return len(self.ids)



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

