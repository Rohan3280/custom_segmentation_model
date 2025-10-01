import argparse
from src.utils.config import load_config
from src.train.train_loop import train_model
import torch

def main(config_path, save_path):
    config = load_config(config_path)
    model = train_model(config)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--save", type=str, default="experiments/weights/vit_new.pt")
    args = parser.parse_args()
    main(args.config, args.save)
    