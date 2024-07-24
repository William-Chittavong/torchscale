

import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
import tqdm

from torchscale.clip_utils.binary_waterbirds import BinaryWaterbirds

from torchscale.component.prs_hook import hook_prs_logger

from torchvision.datasets import CIFAR100, CIFAR10, ImageNet, ImageFolder

from torchscale.component.transform import image_transform

from torchscale.model.BEiT3 import create_beit3_retrieval_model
from torchscale.component.beit3_utils import load_model_and_may_interpolate

from einops import rearrange



def get_args_parser():
    parser = argparse.ArgumentParser("Project Residual Stream", add_help=False)
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size")
    # Model parameters
    
    parser.add_argument("--model", default= "BEiT3ForRetrieval" ,type = str)
    
    parser.add_argument("--checkpoint_path",default = "https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_224.pth" , help = "pretrained checkpoints")
    
    
    parser.add_argument("--img_size",default = 224 , help="image size", type = int )
    
    parser.add_argument("--model_size",default = "base",help = "model size",type = str)
    
    parser.add_argument("--num_heads" ,default = 1 , help = "attention heads", type = int)
    
    
    # Dataset parameters
    parser.add_argument(
        "--data_path", default="/home/william", type=str, help="dataset path"
    )
    parser.add_argument(
        "--dataset", type=str, default="imagenet", help="imagenet, cub or waterbirds"
    )
    parser.add_argument("--num_workers", default=1, type=int)
    
    
    
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where to save"
    )
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    

    
    return parser


def main(args):
    """Calculates the projected residual stream for a dataset."""
    model = create_beit3_retrieval_model(model_size='base', img_size= args.img_size)
    
    
    load_model_and_may_interpolate(args.checkpoint_path, model, model_key='model', model_prefix='')

    model.to(args.device)
    model.eval()
    
   
    prs = hook_prs_logger(model, args.device , spatial = False) # spatial by default is true
    
    preprocess = image_transform(
        args.img_size,
        is_train=False,
    )

    # Data:
    if args.dataset == "imagenet":
        ds = ImageNet(root=args.data_path, split="val", transform=preprocess)
    elif args.dataset == "binary_waterbirds":
        ds = BinaryWaterbirds(root=args.data_path, split="test", transform=preprocess)
    elif args.dataset == "CIFAR100":
        ds = CIFAR100(
            root=args.data_path, download=True, train=False, transform=preprocess
        )
    elif args.dataset == "CIFAR10":
        ds = CIFAR10(
            root=args.data_path, download=True, train=False, transform=preprocess
        )
    else:
        ds = ImageFolder(root=args.data_path, transform=preprocess)
    dataloader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    attention_results = []
    ffn_results = []
    #cls_to_cls_results = []
    
    for i, (image, _) in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            prs.reinit()
            representation , _ = model(
                image.to(args.device), normalize=False , only_infer = True
            )
           
            attentions, ffns = prs.finalize(representation)
            #attentions, _ = prs.finalize(representation)
            attentions = attentions.detach().cpu().numpy()  # torch.Size([2, 12, 197, 12, 768])

            ffns = ffns.detach().cpu().numpy()  # [b, l+1, d] , for now since no ln before, its actually [ b , l , (h d) ]
            # if args.spatial:
            #     attention_results.append(
            #         np.sum(attentions, axis=2)
            #     )  
            # else:
            #     attention_results.append(
            #             attentions
            # )
                
            attention_results.append(
                        attentions
            )
            ffn_results.append(ffns) 
            # cls_to_cls_results.append(
            #      np.sum(attentions[:, :, 0], axis=2)
            #  )  # Store the cls->cls attention, reduce the heads
            torch.cuda.empty_cache()
            

    with open(
        os.path.join(args.output_dir, f"{args.dataset}_attn_{args.model}.npy"), "wb"
    ) as f:
        np.save(f, np.concatenate(attention_results, axis=0))
    with open(
         os.path.join(args.output_dir, f"{args.dataset}_ffn_{args.model}.npy"), "wb"
     ) as f:
         np.save(f, np.concatenate(ffn_results, axis=0))
    # with open(
    #      os.path.join(args.output_dir, f"{args.dataset}_cls_attn_{args.model}.npy"), "wb"
    #  ) as f:
    #      np.save(f, np.concatenate(cls_to_cls_results, axis=0))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
