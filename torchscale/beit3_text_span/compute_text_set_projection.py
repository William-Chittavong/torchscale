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



from torchscale.model.BEiT3 import create_beit3_retrieval_model
from torchscale.component.beit3_utils import load_model_and_may_interpolate


from transformers import XLMRobertaTokenizer



def get_args_parser():
    parser = argparse.ArgumentParser('Get text list weights', add_help=False)
    # Model parameters
    parser.add_argument('--batch_size', default=2048, type=int,
                        help='Batch size')
    parser.add_argument('--model', default= "BEiT3ForRetrieval", type=str, metavar='MODEL',
                        help='Name of model to use')
    parser.add_argument("--checkpoint_path",default = "https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_224.pth" , help = "pretrained checkpoints")
    
    parser.add_argument("--tokenizer",default = "/home/william/Documents/GitHub/torchscale/beit3.spm" ,  help = "sentence piece tokenizer" , type =str )
    
    parser.add_argument("--img_size",default = 224 , help="image size", type = int )
    
    parser.add_argument("--model_size",default = "base",help = "model size",type = str)
    

    # Dataset parameters
    parser.add_argument('--data_path', default='text_descriptions/image_descriptions_general.txt', 
                        type=str, help='dataset path')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for testing')
    return parser



def get_text_features(model, tokenizer, lines, 
                      device, batch_size, amp=True, use_format=False):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    

    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    lines: list of str
        name of classes
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    autocast = torch.cuda.amp.autocast
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for i in tqdm.trange(0, len(lines), batch_size):
            texts = lines[i:i+batch_size] 
            texts = tokenizer(texts, return_tensors='pt',padding=True, truncation=True).to(device)  # tokenize
            _ ,class_embeddings = model(text_description = texts["input_ids"] , only_infer = True) # shape 1, 768
            class_embeddings = F.normalize(class_embeddings, dim=-1) # (shape 1, 768 before mean)
            print("class embeddings shape #\n",class_embeddings.shape)
            zeroshot_weights.append(class_embeddings.detach().cpu())
        zeroshot_weights = torch.concatenate(zeroshot_weights, dim=0)
    print("zeroshot_weights shape \n",zeroshot_weights.shape)
    return zeroshot_weights


def main(args):
    """Calculates the classifier projection weights."""
    model = create_beit3_retrieval_model(model_size='base', img_size= args.img_size)
    
    
    load_model_and_may_interpolate(args.checkpoint_path, model, model_key='model', model_prefix='')

    tokenizer = XLMRobertaTokenizer(args.tokenizer)
    model.to(args.device)
    model.eval()
    
    with open(args.data_path, 'r') as f:
        lines = f.readlines()
    base, name = os.path.split(args.data_path)
    name = name.replace('.txt', '')
    features = get_text_features(model, tokenizer, lines, args.device, args.batch_size)
    # with open(os.path.join(args.output_dir, f'{name}.npy'), 'wb') as f:
    #     np.save(f, features.numpy())
    
    ####
    #current
    # with open(os.path.join(args.output_dir, f'{name}_{args.model}.npy'), 'wb') as f:
    #     np.save(f, features.numpy())
    
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)