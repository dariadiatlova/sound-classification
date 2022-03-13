import datetime

import pytorch_lightning as pl
import wandb
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.dataset import get_dataloader
from src.model.ligtning_model import Classifier


def main(config_path: str = "src/config.yaml"):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    train_config = config["train"]
    wandb_config = config["wandb"]
    seed_everything(train_config["seed"])


    group_name = f"cross_val_{datetime.datetime.now()}"
    for test_fold in range(1, 11):
        wandb.init()
        wandb_logger = WandbLogger(project=wandb_config["project"], config=train_config,
                                   name=f"test_folder_{test_fold}", group=group_name)

        train_loader, val_loader = get_dataloader(test_fold)
        model = Classifier(train_config)
        callbacks = ModelCheckpoint(dirpath=wandb_logger.experiment.dir, monitor="val/accuracy", save_top_k=-1,
                                    every_n_epochs=1)

        trainer = pl.Trainer(max_epochs=train_config["epochs"], logger=wandb_logger,
                             check_val_every_n_epoch=wandb_config["log_val_each_n_epoch"],
                             log_every_n_steps=wandb_config["log_train_loss_each_n_step"],
                             gpus=train_config.get("gpu", None),
                             callbacks=[callbacks])
        trainer.fit(model, train_loader, val_loader)
        wandb.finish()


if __name__ == "__main__":
    main()
