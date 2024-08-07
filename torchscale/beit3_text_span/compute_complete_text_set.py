import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import os
import einops
from torch.utils.data import DataLoader
import tqdm
import argparse
from torchvision.datasets import ImageNet
from pathlib import Path

from torchscale.clip_utils.misc import accuracy


@torch.no_grad()
def replace_with_iterative_removal(data, text_features, texts, iters, rank, device , num_heads):
    results = []
    
    print(f"Data shape: {data.shape}") # 50000 , 768
    print(f"Text features shape: {text_features.shape}") # (3498, 768)
    print(f"Rank: {rank}") #  80
    u, s, vh = np.linalg.svd(data, full_matrices=False)
    vh = vh[:rank]
    
    print(f"vh shape after truncation: {vh.shape}") # (80, 768)
    
    text_features = (
        vh.T.dot(np.linalg.inv(vh.dot(vh.T)).dot(vh)).dot(text_features.T).T
    )  # Project the text to the span of W_OV
    
    data = torch.from_numpy(data).float().to(device)
    mean_data = data.mean(dim=0, keepdim=True)
    data = data - mean_data
    reconstruct = einops.repeat(mean_data, "A B -> (C A) B", C=data.shape[0])
    reconstruct = reconstruct.detach().cpu().numpy()
    text_features = torch.from_numpy(text_features).float().to(device)
    for i in range(iters):
        projection = data @ text_features.T
        projection_std = projection.std(axis=0).detach().cpu().numpy()
        top_n = np.argmax(projection_std)
        results.append(texts[top_n])
        text_norm = text_features[top_n] @ text_features[top_n].T
        reconstruct += (
            (
                (data @ text_features[top_n] / text_norm)[:, np.newaxis]
                * text_features[top_n][np.newaxis, :]
            )
            .detach()
            .cpu()
            .numpy()
        )
        data = data - (
            (data @ text_features[top_n] / text_norm)[:, np.newaxis]
            * text_features[top_n][np.newaxis, :]
        )
        text_features = (
            text_features
            - (text_features @ text_features[top_n] / text_norm)[:, np.newaxis]
            * text_features[top_n][np.newaxis, :]
        )
    return reconstruct, results


def get_args_parser():
    parser = argparse.ArgumentParser("Completeness part", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-H-14",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    # Dataset parameters
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--input_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--text_descriptions",
        default="image_descriptions_per_class",
        type=str,
        help="name of the evalauted text set",
    )
    parser.add_argument(
        "--text_dir",
        default="./text_descriptions",
        type=str,
        help="The folder with the text files",
    )
    parser.add_argument(
        "--dataset", type=str, default="imagenet", help="imagenet or waterbirds"
    )
    parser.add_argument(
        "--num_of_last_layers",
        type=int,
        default=8,
        help="How many attention layers to replace.",
    )
    parser.add_argument(
        "--w_ov_rank", type=int, default=80, help="The rank of the OV matrix"
    )
    parser.add_argument(
        "--texts_per_head",
        type=int,
        default=10,
        help="The number of text examples per head.",
    )
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    
    parser.add_argument("--num_heads" , default = 12, type = int,help = "attn heads" )
    return parser

def check_summary_statistics(tensor, name):
    print(f"\n{name} statistics:")
    print(f"Mean: {tensor.mean().item()}")
    print(f"Std: {tensor.std().item()}")
    print(f"Min: {tensor.min().item()}")
    print(f"Max: {tensor.max().item()}")

def main(args):
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_attn_{args.model}.npy"), "rb"
    ) as f:
        attns = np.load(f)
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_ffn_{args.model}.npy"), "rb"
    ) as f:
        ffns = np.load(f)  # [b, l+1, d]
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_classifier_{args.model}.npy"),
        "rb",
    ) as f:
        classifier = np.load(f)
    
    # print(f"Number of layers: {attns.shape[1]}")
    # print(f"\n attn shape: {attns.shape}")

    all_images = set()
    for i in tqdm.trange(attns.shape[1] - args.num_of_last_layers):
        for head in range(attns.shape[2]):
            attns[:, i, head] = np.mean(attns[:, i, head], axis=0, keepdims=True)
    
    with open(
        os.path.join(args.input_dir, f"{args.text_descriptions}_{args.model}.npy"), "rb"
    ) as f:
        text_features = np.load(f)
    with open(os.path.join(args.text_dir, f"{args.text_descriptions}.txt"), "r") as f:
        lines = [i.replace("\n", "") for i in f.readlines()]
    with open(
        os.path.join(
            args.output_dir,
            f"{args.dataset}_completeness_{args.text_descriptions}_top_{args.texts_per_head}_heads_{args.model}.txt",
        ),
        "w",
    ) as w:
        for i in tqdm.trange(attns.shape[1] - args.num_of_last_layers, attns.shape[1]):
            for head in range(attns.shape[2]):
               # print("attns[:, i, head] shape \n", attns[:, i, head].shape)
                results, images = replace_with_iterative_removal(
                    attns[:, i, head],
                    text_features,
                    lines,
                    args.texts_per_head,
                    args.w_ov_rank,
                    args.device,
                    args.num_heads
                )
                attns[:, i, head] = results
                all_images |= set(images)
                w.write(f"------------------\n")
                w.write(f"Layer {i}, Head {head}\n")
                w.write(f"------------------\n")
                for text in images:
                    w.write(f"{text}\n")
        # print("attns before mean ablated and replaced \n", attns)
        # print(attns.shape)
        
        mean_ablated_and_replaced = ffns.sum(axis = 1) + attns.sum(axis=(1, 2))
        #print("\n mean ablated replaced numpy before projection", mean_ablated_and_replaced)
        
        mean_ablated_tensor = torch.from_numpy(mean_ablated_and_replaced).float()
        # print("\n mean replaced shape ", mean_ablated_tensor.shape)
        # check_summary_statistics(mean_ablated_tensor, "Mean Ablated Tensor")
        
        classifier_tensor = torch.from_numpy(classifier).float()
        # print("\n classifier before projection ", classifier)
        # print("\n classifier shape", classifier_tensor.shape)
        # check_summary_statistics(classifier_tensor, "Classifier Tensor")
        
        projections = mean_ablated_tensor.to(args.device) @ classifier_tensor.to(args.device)
        
        # print("\n projections shape \n", projections.shape)
        # print("projections \n ", projections)
        # check_summary_statistics(projections, "Projections")
        
        labels = np.array([i // 50 for i in range(attns.shape[0])])
        labels_tensor = torch.from_numpy(labels)
        # print("labels tensor \n", labels_tensor)
        # print(labels_tensor.shape)
        
        current_accuracy = (
            accuracy(projections.cpu(), labels_tensor)[0] * 100.0
        )
        print(
            f"Current accuracy:", current_accuracy, "\nNumber of texts:", len(all_images)
        )
        w.write(f"------------------\n")
        w.write(f"Current accuracy: {current_accuracy}\nNumber of texts: {len(all_images)}")

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)