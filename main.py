# Enes Erdogan
# 504211555

import wandb
import torch

from utils import get_args


def main(args):
    wandb.init(
        project="dl-action-recognition",
        name=args.name,
        mode="online" if args.wandb else "disabled",
    )

    # optimizer = torch.optim.Adam(
    #     [
    #         {"params": model.head.parameters(), "lr": 1e-3},            
    #     ],
    #     lr=1e-2,
    # )

    for epoch in range(1, args.epochs + 1):
        metrics_train = run_one_epoch(
            args, model, device, train_loader, optimizer, epoch
        )
        metrics_train_val = run_one_epoch(args, model, device, test_loader)
        # pick best model and save it

    metrics_test = run_one_epoch(args, model, device, train_loader, optimizer, epoch)

    wandb.finish()


if __name__ == "__main__":
    main(get_args())
