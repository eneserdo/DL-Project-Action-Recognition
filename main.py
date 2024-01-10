# Enes Erdogan
# 504211555

import wandb
from utils import freeze_all_except_head, get_args, plot_conf_mat
import timm
from torchvision import transforms as pth_transforms
from torchvision import datasets
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy



# Setting seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Some constants
NUM_CLASSES = None # Will be inferred from the dataset
CLASS_NAMES = None # Will be inferred from the dataset

# Some hyperparameters
SPLIT_RATIO = [0.7, 0.15, 0.15] # train, val, test (3rd one is inferred)

assert torch.cuda.is_available(), "No GPU, no training!"
device = "cuda:0"

def run_one_epoch(model, loader, mode, criterion, optimizer, epoch):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    acc = MulticlassAccuracy(num_classes=NUM_CLASSES)
    conf_mat = MulticlassConfusionMatrix(num_classes=NUM_CLASSES, normalize="true")

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(mode == 'train'):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        acc.update(outputs.cpu(), labels.cpu())
        conf_mat.update(outputs.cpu(), labels.cpu())


    metrics = {
        'loss': loss.item(),
        'accuracy': acc.compute(),
        'confusion_matrix': conf_mat.compute()
    }


    wandb.log({f"{mode}/": {'loss': metrics['loss'], 'accuracy': metrics['accuracy']}, 'epoch': epoch})


    conf_plt=plot_conf_mat(mat=metrics['confusion_matrix'], name=f'{mode}_confusion_matrix', labels=CLASS_NAMES)   
    wandb.log({f"{mode}/": {"confusion_matrix": wandb.Image(conf_plt)}, "epoch": epoch})
    conf_plt.close()

    return metrics


def main(args):
    global NUM_CLASSES, CLASS_NAMES

    wandb.init(
        project="dl-action-recog",
        name=args.name,
        mode="online" if args.wandb else "disabled",
        config=args.__dict__,
    )

    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("test/*", step_metric="epoch")

    # Construct datasets, dataloaders
    transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # denormalize = pth_transforms.Compose(
    #     [
    #         pth_transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    #         pth_transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    #     ]
    # )

    my_dataset = datasets.ImageFolder(root="dl_action_dataset", transform=transform)
    CLASS_NAMES = my_dataset.classes
    NUM_CLASSES = len(CLASS_NAMES)

    # split the dataset in train, validation and test set
    train_size = int(SPLIT_RATIO[0] * len(my_dataset))
    val_size = int(SPLIT_RATIO[1] * len(my_dataset))
    test_size = len(my_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(my_dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=True)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    # Construct model, loss function, optimizer, scheduler
    critereon = torch.nn.CrossEntropyLoss()


    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = torch.nn.Linear(model.head.in_features, NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    
    if args.lr_base == 0.0:
        freeze_all_except_head(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_head)
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": model.head.parameters(), "lr": args.lr_head},
            ],
            lr=args.lr_base,
        )


    head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    print(f"Total params: {total_params:_}")
    print(f"Trainable head params: {head_params:_}")
    wandb.watch(model, log='all')

    model.to(device)

    # Experiment loop
    best_model_accuracy = 0.0
    best_model_epoch = 0

    for e in range(args.epochs):
        train_metrics = run_one_epoch(model, train_dataloader, 'train', critereon, optimizer, e)
        val_metrics = run_one_epoch(model, val_dataloader, 'val', critereon, optimizer, e)

        print(f'E {e+1}/{args.epochs}: '
              f'training loss: {train_metrics["loss"]:.4f} acc: {train_metrics["accuracy"]:.4f}, '
              f'val loss: {val_metrics["loss"]:.4f} acc: {val_metrics["accuracy"]:.4f}')

        if val_metrics['accuracy'] > best_model_accuracy:
            best_model_epoch = e
            best_model_accuracy = val_metrics['accuracy']
            torch.save(model.state_dict(), f'best_model.pth')


    # Load best model and evaluate on test set
    print(f'Loading {best_model_epoch}th model')
    model.load_state_dict(torch.load('best_model.pth'))

    test_metrics=run_one_epoch(model, test_dataloader, 'test', critereon, optimizer, 0)

    print(f'Test loss: {test_metrics["loss"]:.4f} acc: {test_metrics["accuracy"]:.4f}')

    wandb.finish()


if __name__ == "__main__":
    main(get_args())
