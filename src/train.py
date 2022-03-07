import datetime

import pytorch_lightning as pl

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from src.dataset import get_dataloader
from src.model.ligtning_model import Classifier


def main(config_path: str = "config.yaml"):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    train_config = config["train"]
    wandb_config = config["wandb"]
    seed_everything(train_config["seed"])

    group_name = f"cross_val_{datetime.datetime.now()}"
    for test_fold in range(1, 10):

        wandb_logger = WandbLogger(project=wandb_config["project"], config=train_config,
                                   name=f"test_folder_{test_fold}", group=group_name)
        train_loader, val_loader = get_dataloader(test_fold)
        model = Classifier(train_config)
        trainer = pl.Trainer(max_epochs=train_config["epochs"], logger=wandb_logger, check_val_every_n_epoch=1,
                             log_every_n_steps=10, gpus=train_config.get("gpu", None))
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
