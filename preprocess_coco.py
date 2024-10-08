import os
import cv2
from pycocotools.coco import COCO
import ujson as json
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
# Filter the UserWarning related to low contrast images
warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image.*")

# Specify the paths to the COCO dataset files
data_dir = "/home/william/project/coco_seg"

train_dir = os.path.join(data_dir, 'train2017')
val_dir = os.path.join(data_dir, 'val2017')
annotations_dir = os.path.join(data_dir, 'annotations')
train_annotations_file = os.path.join(annotations_dir, 'instances_train2017.json')
val_annotations_file = os.path.join(annotations_dir, 'instances_val2017.json')

# Create directories for preprocessed images and masks
preprocessed_dir = './preprocessed'
os.makedirs(os.path.join(preprocessed_dir, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(preprocessed_dir, 'train', 'masks'), exist_ok=True)
os.makedirs(os.path.join(preprocessed_dir, 'val', 'images'), exist_ok=True)
os.makedirs(os.path.join(preprocessed_dir, 'val', 'masks'), exist_ok=True)

batch_size = 10  # Number of images to process before updating the progress bar



# def preprocess_image(img_info, coco, data_dir, output_dir):
#     image_path = os.path.join(data_dir, img_info['file_name'])
#     ann_ids = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
#     if len(ann_ids) == 0:
#         return

#     anns = coco.loadAnns(ann_ids)
#     mask = coco.annToMask(anns[0])  # Get the first annotation's mask

#     # Save the preprocessed image
#     image = cv2.imread(image_path)
#     cv2.imwrite(os.path.join(output_dir, 'images', img_info['file_name']), image)

#     # Convert the mask to white (255) on black (0) background
#     mask = (mask * 255).astype(np.uint8)

#     # Save the corresponding mask
#     cv2.imwrite(os.path.join(output_dir, 'masks', img_info['file_name'].replace('.jpg', '.png')), mask)
def preprocess_image(img_info, coco, data_dir, output_dir):
    image_path = os.path.join(data_dir, img_info['file_name'])
    ann_ids = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
    if len(ann_ids) == 0:
        return

    mask = coco.annToMask(coco.loadAnns(ann_ids)[0])

    # Save the preprocessed image
    image = cv2.imread(image_path)
    cv2.imwrite(os.path.join(output_dir, 'images', img_info['file_name']), image)

    # Save the corresponding mask
    cv2.imwrite(os.path.join(output_dir, 'masks', img_info['file_name'].replace('.jpg', '.png')), mask)

def preprocess_dataset(data_dir, annotations_file, output_dir):
    coco = COCO(annotations_file)
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    image_infos = coco_data['images']

    total_images = len(image_infos)
    num_batches = total_images // batch_size

    # Use tqdm to create a progress bar
    progress_bar = tqdm(total=num_batches, desc='Preprocessing', unit='batch(es)', ncols=80)

    with ThreadPoolExecutor() as executor:
        for i in range(0, total_images, batch_size):
            batch_image_infos = image_infos[i:i+batch_size]
            futures = []

            for img_info in batch_image_infos:
                future = executor.submit(preprocess_image, img_info, coco, data_dir, output_dir)
                futures.append(future)

            # Wait for the processing of all images in the batch to complete
            for future in futures:
                future.result()

            progress_bar.update(1)  # Update the progress bar for each batch

    progress_bar.close()  # Close the progress bar once finished

# Preprocess the training set
preprocess_dataset(train_dir, train_annotations_file, os.path.join(preprocessed_dir, 'train'))

# Preprocess the validation set (if required)
preprocess_dataset(val_dir, val_annotations_file, os.path.join(preprocessed_dir, 'val'))
