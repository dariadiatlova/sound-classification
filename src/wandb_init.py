import wandb
from datetime import datetime


def _init_wandb(wandb_config: dict, train_config: dict):
    if wandb_config["use_wandb"]:
        wandb.login()
        now = datetime.now()
        date_time = f"{now.hour}:{now.minute}:{now.second}-{now.day}.{now.month}.{now.year}"

        wandb.init(project=wandb_config["project"],
                   name=f'{wandb_config["name"]}:{date_time}',
                   notes=wandb_config["notes"],
                   entity=wandb_config["entity"],
                   config=train_config)
