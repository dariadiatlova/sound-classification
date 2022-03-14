## Urban-sound classification with LSTM

### Dataset:
[UrbanSound dataset](https://urbansounddataset.weebly.com/urbansound8k.html).

### Task artifacts:
[Presentation with results](test-task-presentation-diatlova.pdf).

[Model weights](src/artifacts/epoch23.ckpt).

[Weights & biases report](https://wandb.ai/daryoou_sh/sound-classification/reports/UrbanSound-cross-validation--VmlldzoxNjgyMTkz).

### Training:
To replicate training process clone this repository, install the [requirements.txt](requirements.txt), update [config.yaml](src/config.yaml) and run command from the command line: 

    python3 -m src.train
    
Note: following script implies logging via Weights&Biases, please run: `wandb loging` to log into your account.

### Evaluation:
As we train model using cross-validation this repository does not consist a separate script to compute accuracy on validation folder, insted we can run [evaluate.py](src/evaluation/evaluate.py) script to predict class for the audio file by running: 
    
    python3 -m src.evaluaton.evaluate -p <path to the .wav file> -w <path to the model weights>
    
An example of running [evaluate.py](src/evaluation/evaluate.py) script can be found in the [notebook](src/evaluation/eval_example.ipynb), feel free to open it in Google Colab, to listen and observe model predictions for 4 random samples from urbansound dataset.
