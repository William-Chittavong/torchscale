import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import os.path
import argparse
import datetime
import json
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm

from torchscale.clip_utils.openai_templates  import OPENAI_IMAGENET_TEMPLATES
from torchscale.clip_utils.imagenet_classes import imagenet_classes
from torchscale.clip_utils.cub_classes import cub_classes, waterbird_classes

from torchscale.model.BEiT3 import create_beit3_retrieval_model
from torchscale.component.beit3_utils import load_model_and_may_interpolate


from transformers import XLMRobertaTokenizer

def get_args_parser():
    parser = argparse.ArgumentParser('Get classifier weights', add_help=False)
    # Model parameters
    
    parser.add_argument('--dataset', default='imagenet', help='waterbirds or imagenet')
    
    
    parser.add_argument("--checkpoint_path",default = "https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_224.pth" , help = "pretrained checkpoints")
    
    parser.add_argument("tokenizer",default = "/home/william/Documents/GitHub/torchscale/beit3.spm" , help = "sentence piece tokenizer" )
    
    parser.add_argument("--img_size",default = 224 , help="image size", type = int )
    
    parser.add_argument("model_size",default = "base",help = "model size",type = str)
    
    parser.add_argument()
    
    
    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for testing')
    return parser



def zero_shot_classifier(model, tokenizer, classnames, templates, 
                         device, amp=True, use_format=False):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    

    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    classnames: list of str
        name of classes
    
    templates: list of str
        templates to use.
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    autocast = torch.cuda.amp.autocast
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for classname in tqdm.tqdm(classnames):
            texts = [template.format(c=classname) if use_format else template(classname) for template in templates]
            texts = tokenizer(texts, return_tensors='pt').to(device)  # tokenize
            class_embeddings = model(text_description = texts , only_infer = True)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0) # [768] , d 
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def main(args):
    """Calculates the classifier projection weights."""
   
    
    # can modify later to add parse argument and create model function with choice of beit3 models. 
    model = create_beit3_retrieval_model(model_size='base', img_size= args.img_size)
    
    
    load_model_and_may_interpolate(args.checkpoint_path, model, model_key='model', model_prefix='')

    tokenizer = XLMRobertaTokenizer(args.tokenizer)
    model.to(args.device)
    model.eval()
    
    
    classes = {
        'imagenet': imagenet_classes, 
        'waterbirds': cub_classes, 
        'binary_waterbirds': waterbird_classes, 
        'cub': cub_classes}[args.dataset]
    classifier = zero_shot_classifier(model, tokenizer, classes, OPENAI_IMAGENET_TEMPLATES, args.device)
    with open(os.path.join(args.output_dir, f'{args.dataset}_classifier_{args.model}.npy'), 'wb') as f:
        np.save(f, classifier.detach().cpu().numpy())
    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)