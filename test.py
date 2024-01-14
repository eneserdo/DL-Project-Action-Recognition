# Enes Erdogan
# 504211555

import os
from time import perf_counter
import wandb
import timm
import torch

from utils import freeze_all_except_head, get_test_args, get_training_args, get_dataloaders, logdict2str, plot_conf_mat, Metrics

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

def main(args):
    global NUM_CLASSES, CLASS_NAMES

    # Initialize wandb
    wandb.init(project="dl-action-recog", name=args.name, mode="disabled")
    print(args)

    # Construct datasets, dataloaders
    _, _, test_dataloader, _, NUM_CLASSES, CLASS_NAMES = get_dataloaders(args)


    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = torch.nn.Linear(model.head.in_features, NUM_CLASSES)
    model.to(device)


    # Load best model and evaluate on test set
    print(f'Loading the model...')
    best_model_name = os.path.join("saved", f"{args.name}_best_model.pth")
    model.load_state_dict(torch.load(best_model_name))

    test_logs=run_one_epoch(model, test_dataloader, 'test', None, None, 0)

    print(f'Test {logdict2str(test_logs)}')

    wandb.finish()


if __name__ == "__main__":
    raise NotImplementedError("Not ready yet")
    main(get_test_args())
