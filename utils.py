import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as pth_transforms
from torchvision import datasets
from torchvision.utils import make_grid
from collections import Counter

def get_args():
    parser = argparse.ArgumentParser(description="Action Recognition")

    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-ds", "--dataset", type=str, required=True, choices=["our", "Stanford40"])
    parser.add_argument("--wandb", action="store_true", default=False)
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)

    # Optimizer parameters
    parser.add_argument("-lrh", "--lr-head", type=float, default=0.0001)
    parser.add_argument("-lrb", "--lr-base", type=float, default=0.0)
    parser.add_argument("-wd", "--weight-decay", type=float, default=1e-5)
    parser.add_argument("--weighted-loss", action="store_true", default=False)

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

    od=dict(Counter(train_dataset.targets))
    distribution = [od[k] for k in range(len(od))]
    distribution = torch.tensor(distribution)
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader, distribution, NUM_CLASSES, CLASS_NAMES


def visualize_dataset(ds: torch.utils.data.Dataset, denormalize, N=6, fsize=15):
    N = 6
    samples = {}

    # after transform
    h, w = 224, 224

    NUM_CLASSES = len(ds.classes)
    for c in range(NUM_CLASSES):
        print(".", end="", flush=True) # a primitive progress bar

        samples[c] = []
        for i in range(len(ds)):
            if ds[i][1] == c:
                samples[c].append(denormalize(ds[i][0]))
            if len(samples[c]) == N:
                break


    grid = make_grid(torch.cat([torch.stack(samples[c]) for c in range(NUM_CLASSES)]), nrow=N)
    plt.figure(figsize=(fsize, fsize))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
