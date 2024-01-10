import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def get_args():
    parser = argparse.ArgumentParser(description="Action Recognition")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("-n", "--name", type=str, required=True)

    parser.add_argument("-lrh", "--lr-head", type=float, default=0.001)
    parser.add_argument("-lrb", "--lr-base", type=float, default=0.0)
    parser.add_argument("-wd", "--weight-decay", type=float, default=0.0)

    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--wandb", action="store_true", default=False)

    return parser.parse_args()


def freeze_all_except_head(model):
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def plot_conf_mat(*, mat, name, labels):
    mat_long = torch.round(mat * 100).long()

    plt.figure(figsize=(14, 12))
    sns.heatmap(mat_long, square=True, annot=True, cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels, vmin=0, vmax=100)
    plt.yticks(rotation=0)
    plt.title(f"{name}")
    plt.xlabel('Predicted')
    plt.ylabel('Ground truth')
    plt.tight_layout()

    return plt