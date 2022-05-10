# Image SuperResolution

Bachelor thesis on Image SuperResolution.

&nbsp;


## Requirements

Python version 3.9.1 was used for the development, but it is likely that any Python version above 3.7 will work as well.

Two packages from nvidia: cudatoolkit 11.2 and cuDNN 8.2.1. The versions of these two packages depend on each other and on other Python libraries.

Python libraries:
- keras 2.6.0
- numpy 1.19.5
- opencv-python-headless 4.5.4.58
- tensorflow 2.6.0
- pillow 9.1.0

If WandB logging is to be used, the wandb package must be installed.

If Azure ML is to be used, the azureml packages must be installed.

&nbsp;


## Running

- Run [predict.py](predict.py)
- Select image to upscale
- Choose upscaling factor


&nbsp;

#

Samuel Buban