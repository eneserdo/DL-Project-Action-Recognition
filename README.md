# DL-Project-Action-Recognition

~~We will compare two separate methods for action recognition on still images. The first method will be using pre-trained vision models and the second method will be using graph neural networks.~~

Unfortunately, my teammate left the project. So, I will just apply various fine-tuning strategies on ViT models for action recognition using the dataset we collected.


## Directories

```
DL-Project-Action-Recognition/
├── README.md
├── prepare_dataset.py  # Script for preparing the dataset
├── visualization.ipynb       # Notebook to inspect the model and dataset
├── utils.py            # Utility functions
├── main.py             # main training and validation code for action recognition
└── scripts             # Tools for data collection
    └── ...

```


## Environment

You can use following commands to create the required environments:

```
conda create -n myenv python=3.9

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install timm gdown grad-cam ipykernel seaborn torchmetrics wandb
```

Some package details:

- timm: huggingface's model hub
- gdown: for downloading the dataset
- grad-cam: to inspect the model


## Dataset

### Our Dataset
We curated a video dataset of 4 actions with 5 repetitions each. Since our model is image-based, we extracted the frames, so in total we have 3364 images. 

The dataset can be found [here](https://drive.google.com/file/d/16pMYcodc5oshaTIcQYcHrK2VDG7vUuLa/view?usp=drive_link).

However, you can simply run the following script to download and prepare the dataset:

```python prepare_dataset.py --dataset our```


### Stanford40 Dataset

> The Stanford 40 Action Dataset contains images of humans performing 40 actions. In each image, we provide a bounding box of the person who is performing the action indicated by the filename of the image. There are 9532 images in total with 180-300 images per action class. [1]

To download and prepare the dataset for training:

```python prepare_dataset.py --dataset Stanford40```


## Fine-tuning ViT Models 
...
...

## Results
...
...

## REFERENCES:

[1] http://vision.stanford.edu/Datasets/40actions.html 

## Other Resources I used:

- https://huggingface.co/docs/timm/index
- https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md
- https://jacobgil.github.io/pytorch-gradcam-book/CAM%20Metrics%20And%20Tuning%20Tutorial.html
