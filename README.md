# Put AI CV project 3

## Problem of inpainting using deep neural networks

## How to run this repo:

- `python train.py` - script used to train models that will be saved in the `model_snapshots` folder and its training logs
  for tensorboard will be stored in `logs/fit` folder.
To view the analysis in tensorboard use the following command
```sh
tensorboard --logdir logs/fit
``` 
- `python evaluate.py` - script for model evaluation on the whole dataset
- `python image_generator` - generates examples of model performance from the dataset
