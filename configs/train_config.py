from dataclasses import dataclass


@dataclass
class TrainConfig:
    modality: str = "image"
    image_source: str = "folder"  # synthetic | folder | cifar10
    image_folder: str = "/Users/lintianjian/Desktop"
    cifar10_root: str = "data/cifar10"
    cifar10_train: bool = True
    cifar10_download: bool = True
    batch_size: int = 16
    num_workers: int = 0
    image_size: int = 32
    channels: int = 3
    train_steps: int = 200
    lr: float = 1e-4
    timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    device: str = "cpu"
    log_every: int = 20
    synthetic_dataset_size: int = 1024
    seed: int = 42
    sample_batch_size: int = 4
    checkpoint_path: str = "checkpoints/ddpm.pt"
    sample_output_dir: str = "outputs/samples"
