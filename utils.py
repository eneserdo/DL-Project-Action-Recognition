import argparse
from typing import Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as pth_transforms
from torchvision import datasets
from torchvision.utils import make_grid
from collections import Counter
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassF1Score


def get_training_args():
    parser = argparse.ArgumentParser(description="Action Recognition")

    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-ds", "--dataset", type=str, required=True, choices=["our", "Stanford40"])
    parser.add_argument("--wandb", action="store_true", default=False)
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)

    # Optimizer parameters
    parser.add_argument("-lrh", "--lr-head", type=float, default=0.0001)
    parser.add_argument("-lrb", "--lr-base", type=float, default=0.0)
    parser.add_argument("-wd", "--weight-decay", type=float, default=1e-5)
    parser.add_argument("--weighted-loss", action="store_true", default=False)

    return parser.parse_args()

def get_test_args():
    parser = argparse.ArgumentParser(description="Action Recognition")
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-ds", "--dataset", type=str, required=True, choices=["our", "Stanford40"])

    return parser.parse_args()


def freeze_all_except_head(model):
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False


def reshape_transform(tensor, height=14, width=14):
    # This function is used by Grad-CAM

    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def plot_conf_mat(*, mat, name, labels):
    mat_long = torch.round(mat * 100).long()

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        mat_long,
        square=True,
        annot=True,
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=100,
    )
    plt.yticks(rotation=0)
    plt.title(f"{name}")
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.tight_layout()

    return plt


def get_transforms(ds):

    # These values are from the ImageNet dataset
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if ds == "Stanford40":
        transform = pth_transforms.Compose(
            [
                pth_transforms.Resize(224),
                pth_transforms.CenterCrop(224),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(mean, std),
            ]
        )

    elif ds == "our":
        transform = pth_transforms.Compose(
            [
                pth_transforms.Resize(224),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(mean, std),
            ]
        )

    else:
        raise NotImplementedError(f"Dataset {ds} not implemented")

    denormalize = pth_transforms.Compose(
        [
            pth_transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/m for m in std]),
            pth_transforms.Normalize([-m for m in mean], std=[1.0, 1.0, 1.0]),
        ]
    )

    return transform, denormalize


def get_dataloaders(args):
    # Construct datasets, dataloaders
    SPLIT_RATIO = [0.7, 0.15, 0.15]  # train, val, test (3rd one is inferred)

    ds_path = "dl_action_dataset" if args.dataset == "our" else "Stanford40/processed"

    transform, _ = get_transforms(args.dataset)

    my_dataset = datasets.ImageFolder(root=ds_path, transform=transform)

    CLASS_NAMES = my_dataset.classes
    NUM_CLASSES = len(CLASS_NAMES)

    # split the dataset in train, validation and test set
    train_size = int(SPLIT_RATIO[0] * len(my_dataset))
    val_size = int(SPLIT_RATIO[1] * len(my_dataset))
    test_size = len(my_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        my_dataset, [train_size, val_size, test_size]
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # Compute class distribution (using only the training set)
    od=dict(Counter([my_dataset.targets[idx] for idx in train_dataset.indices]))
    distribution = [od[k] for k in range(len(od))]
    distribution = torch.tensor(distribution)
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader, distribution, NUM_CLASSES, CLASS_NAMES


def visualize_dataset(ds: torch.utils.data.Dataset, denormalize, N=4, fsize=10, use_n_classes=None):
    samples = {}

    

    # after transform
    h, w = 224, 224

    if use_n_classes is None:
        # use all classes
        selected_classes = np.arange(len(ds.classes))
    else:
        selected_classes = np.random.choice(len(ds.classes), use_n_classes, replace=False)

    for c in selected_classes:
        print(".", end="", flush=True) # a primitive progress bar

        shuffled_indices = np.random.permutation(len(ds))

        samples[c] = []
        for i in shuffled_indices:
            if ds[i][1] == c:
                samples[c].append(denormalize(ds[i][0]))
            if len(samples[c]) == N:
                break


    grid = make_grid(torch.cat([torch.stack(samples[c]) for c in selected_classes]), nrow=N)
    plt.figure(figsize=(fsize, fsize))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    print(" ".join([ds.classes[c] for c in selected_classes]))


# To encapsulate the metrics 
class Metrics:
    def __init__(self, num_classes) -> None:
        self.cf=MulticlassConfusionMatrix(num_classes=num_classes, normalize="true")
        self.acc=MulticlassAccuracy(num_classes=num_classes)
        self.f1=MulticlassF1Score(num_classes=num_classes)

    def update(self, preds, targets) -> Any:
        self.cf.update(preds, targets)
        self.acc.update(preds, targets)
        self.f1.update(preds, targets)
        
    def compute(self) -> dict:
        return {
            "cm": self.cf.compute(),
            "acc": self.acc.compute(),
            "f1": self.f1.compute(),
        }

    def reset(self) -> None:
        self.cf.reset()
        self.acc.reset()
        self.f1.reset()


def logdict2str(d):
    logs=""
    for k, v in d.items():
        if k=="cm":
            continue
        logs+=f"{k}: {v:.4f}, "
    
    return logs[:-2]

def get_optimizer(model, args):

    # Construct optimizer
    optimizer = torch.optim.Adam(
        [
            # All the other parameters
            {"params": model.patch_embed.parameters()},
            {"params": model.pos_drop.parameters()},
            {"params": model.patch_drop.parameters()},
            {"params": model.norm_pre.parameters()},
            {"params": model.blocks.parameters()},
            {"params": model.norm.parameters()},
            {"params": model.fc_norm.parameters()},
            {"params": model.head_drop.parameters()},

            # The last layer
            {"params": model.head.parameters(), "lr": args.lr_head},
        ],
        lr=args.lr_base,
        weight_decay=args.weight_decay
    )

    return optimizer