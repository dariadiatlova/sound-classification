#### This repository consists of the source code for the sound classification problem.

### Task artifacts:

[UrbanSound dataset](https://urbansounddataset.weebly.com/urbansound8k.html).

[Model weights](src/artifacts/epoch23.ckpt).

[Weights & biases report](https://wandb.ai/daryoou_sh/sound-classification/reports/UrbanSound-cross-validation--VmlldzoxNjgyMTkz).

### Training:
To replicate training process clone this repository, install the [requirenments](requirenments.txt), update [config](src/config.yaml) and run command from the command line: 

    python3 -m src.train
    
Note: following script implies the logging using Weights&Biases, please run: `wandb loging` from cli if you are not logged in wandb on your machine already.
