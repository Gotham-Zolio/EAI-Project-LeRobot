import tyro
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig):
    print(f"Training with config:\n{cfg}")

if __name__ == "__main__":
    train()
