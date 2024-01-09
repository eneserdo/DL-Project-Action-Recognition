import argparse


def get_args():

    parser = argparse.ArgumentParser(description="Action Recognition")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="input batch size for training (default: 64)")
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--lr-head", type=float, default=0.001)
    parser.add_argument("--lr-base", type=float, default=0.0)

    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--wandb", action="store_true", default=False)

    

    return parser.parse_args()
