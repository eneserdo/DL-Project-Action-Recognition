# DL-Project-Action-Recognition

~~We will compare two separate methods for action recognition on still images. The first method will be using pre-trained vision models and the second method will be using graph neural networks.~~

Unfortunately, my teammate left the project. So, I will just apply various fine-tuning strategies on ViT models for action recognition using the dataset we collected.


## Directories

```
DL-Project-Action-Recognition/
├── README.md
├── dl_action_dataset   # Dataset for action recognition
├── prepare_dataset.py  # Script for preparing the dataset
├── inspect.ipynb       # Notebook for inspecting the model
├── utils.py            # Utility functions
├── main.py             # main code for graph-based action recognition
└── scripts             # Tools for data collection
    └── ...
```


## Environment

You can use following commands to create the required environments:

```
conda create -n myenv python=3.9

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install timm # huggingface's model hub

pip install gdown # for downloading the dataset

pip install grad-cam # to inspect the model

pip install ipykernel seaborn torchmetrics wandb
```

## Dataset

We curated a video dataset of 4 actions with 5 repetitions each. Since our model is image-based, we extracted the frames, so in total we have 3364 images. 

The dataset can be found [here](https://drive.google.com/file/d/16pMYcodc5oshaTIcQYcHrK2VDG7vUuLa/view?usp=drive_link).

However, you can simply run the following command to download the dataset:

```gdown --no-cookie 16pMYcodc5oshaTIcQYcHrK2VDG7vUuLa```

Afterwards, run `python prepare_dataset.py` script to prepare the dataset for training.


## Fine-tuning ViT Models 
...
...

## Results
...
...

## Resources I used:

- https://huggingface.co/docs/timm/index
- https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md
- https://jacobgil.github.io/pytorch-gradcam-book/CAM%20Metrics%20And%20Tuning%20Tutorial.html


