# Enes Erdogan
# 504211555

import os
from time import perf_counter
import wandb
import timm
import torch

from utils import freeze_all_except_head, get_training_args, get_dataloaders, logdict2str, plot_conf_mat, Metrics

# Setting seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Some constants
NUM_CLASSES = None # Will be inferred from the dataset
CLASS_NAMES = None # Will be inferred from the dataset

assert torch.cuda.is_available(), "No GPU, no training!"
device = "cuda:0"

def run_one_epoch(model, loader, mode, criterion, optimizer, epoch):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    # acc = MulticlassAccuracy(num_classes=NUM_CLASSES)
    # conf_mat = MulticlassConfusionMatrix(num_classes=NUM_CLASSES, normalize="true")
    metrics_calculator = Metrics(num_classes=NUM_CLASSES)


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

        metrics_calculator.update(outputs.cpu(), labels.cpu())


    logs = {
        'loss': loss.item(),
        **metrics_calculator.compute(),
    }


    wandb.log({f"{mode}/": logs, 'epoch': epoch})


    conf_plt=plot_conf_mat(mat=logs['cm'], name=f'{mode}_confusion_matrix', labels=CLASS_NAMES)   
    wandb.log({f"{mode}/": {"confusion_matrix": wandb.Image(conf_plt)}, "epoch": epoch})
    conf_plt.close()

    return logs


def main(args):
    global NUM_CLASSES, CLASS_NAMES

    # Initialize wandb
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

    print(args)

    # Construct datasets, dataloaders
    train_dataloader, val_dataloader, test_dataloader, distribution, NUM_CLASSES, CLASS_NAMES = get_dataloaders(args)

    # Construct model, loss function, optimizer
    loss_weights = None if not args.weighted_loss else (1/torch.sqrt(distribution)) # there could be other normalization policies
    critereon = torch.nn.CrossEntropyLoss(weight=loss_weights)

    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = torch.nn.Linear(model.head.in_features, NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    
    if args.lr_base == 0.0:
        freeze_all_except_head(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_head)
    else:
        # two learning rates for head and the rest of the model (I call it base)
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},  
                {"params": model.head.parameters(), "lr": args.lr_head},  
            ],
            lr=args.lr_base,
        )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


    head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    print(f"Total params: {total_params:_}")
    print(f"Trainable head params: {head_params:_}")
    wandb.watch(model, log='all')

    model.to(device)

    # Experiment loop
    print("Starting training...")
    best_model_score = 0.0
    best_model_epoch = 0
    best_model_name = os.path.join("saved", f"{args.name}_best_model.pth")
    comparison_metric = "f1"

    for e in range(args.epochs):
        start = perf_counter()
        train_logs = run_one_epoch(model, train_dataloader, 'train', critereon, optimizer, e)
        val_logs = run_one_epoch(model, val_dataloader, 'val', critereon, optimizer, e)
        end = perf_counter()

        print(f'E {e+1: >2}/{args.epochs}: '
              f'[Training] {logdict2str(train_logs)} '
              f'[Val] {logdict2str(val_logs)} '
              f'({end-start:.2f}s)')

        if val_logs[comparison_metric] > best_model_score:
            best_model_epoch = e
            best_model_score = val_logs[comparison_metric]
            torch.save(model.state_dict(), best_model_name)
        
        scheduler.step()


    # Load best model and evaluate on test set
    print(f'Loading {best_model_epoch}th model')
    model.load_state_dict(torch.load(best_model_name))

    test_logs=run_one_epoch(model, test_dataloader, 'test', critereon, optimizer, 0)

    print(f'Test {logdict2str(val_logs)}')


    wandb.finish()


if __name__ == "__main__":
    main(get_training_args())
