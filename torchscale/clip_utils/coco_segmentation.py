from pycocotools.coco import COCO
import torch
import os
import torch.utils.data as data
from PIL import Image
import numpy as np

class COCOSegmentation(data.Dataset):
    CLASSES = 2  # COCO has 80 classes + 1 background + 10 extra unlabeled classes

    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Load image
        img = coco.loadImgs(img_id)[0]
        path = img['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        # Check if there are annotations
        if len(anns) == 0:
            # If no annotations, create a blank mask (adjust size if needed)
            target = Image.new('L', (image.width, image.height))
        else:
            # Otherwise, create the mask from the first annotation
            target = Image.fromarray(coco.annToMask(anns[0]))

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)
            target = torch.from_numpy(np.array(target)).long()

        return image, target

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

    def __len__(self):
        return len(self.ids)
