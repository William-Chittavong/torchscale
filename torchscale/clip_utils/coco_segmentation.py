from pycocotools.coco import COCO
import torch
import os
import torch.utils.data as data
import torch
from torchvision import transforms
from PIL import Image
import numpy as np




class COCOSegmentation(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = os.path.join(root, split)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self.image_dir = os.path.join(self.root, 'images')
        self.mask_dir = os.path.join(self.root, 'masks')
        
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        
        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        if self.target_transform is None:
            self.target_transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_name = self.images[index]
        mask_name = img_name.replace('.jpg', '.png')
        
        # Load image
        image_path = os.path.join(self.image_dir, img_name)
        img = Image.open(image_path).convert('RGB')
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, mask_name)
        target = Image.open(mask_path).convert('L')  # Convert to grayscale
        
        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # Ensure target is a long tensor
        target = target.squeeze().long()
        
        return img, target


# class COCOSegmentation(Dataset):
#     def __init__(self, root, annFile, transform=None, target_transform=None):
#         self.root = root
#         self.coco = COCO(annFile)
#         self.transform = transform
#         self.target_transform = target_transform
        
#         # Filter out images without annotations
#         self.ids = []
#         for img_id in self.coco.imgs.keys():
#             ann_ids = self.coco.getAnnIds(imgIds=img_id)
#             if len(ann_ids) > 0:
#                 self.ids.append(img_id)

#     def __getitem__(self, index):
#         coco = self.coco
#         img_id = self.ids[index]
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         anns = coco.loadAnns(ann_ids)
        
#         # Load image
#         img = coco.loadImgs(img_id)[0]
#         path = img['file_name']
#         image = Image.open(os.path.join(self.root, path)).convert('RGB')

#         # Create mask from all annotations
#         mask = np.zeros((image.height, image.width), dtype=np.uint8)
#         for ann in anns:
#             mask = np.maximum(mask, coco.annToMask(ann))
#         target = Image.fromarray(mask)

#         if self.transform is not None:
#             image = self.transform(image)

#         if self.target_transform is not None:
#             target = self.target_transform(target)
#             target = torch.from_numpy(np.array(target)).long()

#         return image, target

#     def __len__(self):
#         return len(self.ids)
    
# class COCOSegmentation(data.Dataset):
#     CLASSES = 2  # COCO has 80 classes + 1 background + 10 extra unlabeled classes

#     def __init__(self, root, annFile, transform=None, target_transform=None):
#         self.root = root
#         self.coco = COCO(annFile)
#         self.ids = list(self.coco.imgs.keys())
#         self.transform = transform
#         self.target_transform = target_transform
#     def __getitem__(self, index):
#         coco = self.coco
#         img_id = self.ids[index]
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         anns = coco.loadAnns(ann_ids)
        
#         # Load image
#         img = coco.loadImgs(img_id)[0]
#         path = img['file_name']
#         image = Image.open(os.path.join(self.root, path)).convert('RGB')

#         # Check if there are annotations
#         if len(anns) == 0:
#             # If no annotations, create a blank mask 
#             target = Image.new('L', (image.width, image.height))
#         else:
#             # Otherwise, create the mask from the first annotation
#             target = Image.fromarray(coco.annToMask(anns[0]))

#         if self.transform is not None:
#             image = self.transform(image)

#         if self.target_transform is not None:
#             target = self.target_transform(target)
#             target = torch.from_numpy(np.array(target)).long()

#         return image, target

    # def __getitem__(self, index):
    #     coco = self.coco
    #     img_id = self.ids[index]
    #     ann_ids = coco.getAnnIds(imgIds=img_id)
    #     anns = coco.loadAnns(ann_ids)
    #     img = coco.loadImgs(img_id)[0]
    #     path = img['file_name']

    #     image = Image.open(os.path.join(self.root, path)).convert('RGB')
    #     target = Image.fromarray(coco.annToMask(anns[0]))  # Assuming one mask per image, adjust as needed

    #     if self.transform is not None:
    #         image = self.transform(image)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #         target = torch.from_numpy(np.array(target)).long()

    #     return image, target

    # def __len__(self):
    #     return len(self.ids)
