import numpy as np
import torch
import os.path
import argparse
import einops
from pathlib import Path
import random
import tqdm
from torchscale.clip_utils.misc import accuracy


def full_accuracy(preds, labels, locs_attributes):
    locs_labels = labels.detach().cpu().numpy()
    accs = {}
    for i in [0, 1]:
        for j in [0, 1]:
            locs = np.logical_and(locs_labels == i, locs_attributes == j)
            accs[f"({i}, {j})"] = accuracy(preds[locs], labels[locs])[0] * 100
    accs[f"full"] = accuracy(preds, labels)[0] * 100
    return accs


def get_args_parser():
    parser = argparse.ArgumentParser("Ablations part", add_help=False)

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
        "--figures_dir", default="/home/william/project/results", help="path where data is saved"
    )
    parser.add_argument(
        "--input_dir", default="/home/william/project/results", help="path where data is saved"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="binary_waterbirds",
        help="imagenet, waterbirds, waterbirds_binary or cub",
    )
    return parser


def main(args):

    to_mean_ablate_setting = [(11, 3), (10, 11), (10, 10), (9, 8), (9, 6)]
    to_mean_ablate_geo = [(11, 6), (11, 0)]
    
    to_mean_ablate_output = to_mean_ablate_geo + to_mean_ablate_setting
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_attn_{args.model}.npy"), "rb"
    ) as f:
        attns = np.load(f)  # [b, l, h, d]
        print("attns shape \n ",attns.shape)
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_ffn_{args.model}.npy"), "rb"
    ) as f:
        ffns = np.load(f)  # [b, l+1, d]
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_classifier_{args.model}.npy"),
        "rb",
    ) as f:
        classifier = np.load(f)

    if args.dataset == "imagenet":
        labels = np.array([i // 50 for i in range(attns.shape[0])])
    else:
        with open(
            os.path.join(args.input_dir, f"{args.dataset}_labels.npy"), "rb"
        ) as f:
            labels = np.load(f)
            labels = labels[:, :, 0]
    baseline = attns.sum(axis=(1, 2)) + ffns.sum(axis=1)
    print("lables shape \n ",labels)
    print("labels[:, 0]shape \n ",labels[:, 0])
    baseline_acc = full_accuracy(
        torch.from_numpy(baseline @ classifier).float(),
        torch.from_numpy(labels[:, 0]),
        labels[:, 1],
    )
    print("Baseline:", baseline_acc)
    for layer, head in to_mean_ablate_output:
        attns[:, layer, head, :] = np.mean(
            attns[:, layer, head, :], axis=0, keepdims=True
        )
    for layer in range(attns.shape[1] - 4):
        for head in range(attns.shape[2]):
            attns[:, layer, head, :] = np.mean(
                attns[:, layer, head, :], axis=0, keepdims=True
            )
    for layer in range(ffns.shape[1]):
        ffns[:, layer] = np.mean(ffns[:, layer], axis=0, keepdims=True)
    ablated = attns.sum(axis=(1, 2)) + ffns.sum(axis=1)
    ablated_acc = full_accuracy(
        torch.from_numpy(ablated @ classifier).float(),
        torch.from_numpy(labels[:, 0]),
        labels[:, 1],
    )
    print("Replaced:", ablated_acc)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.figures_dir:
        Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    main(args)