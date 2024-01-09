# DL-Project-Action-Recognition

~~We will compare two separate methods for action recognition on still images. The first method will be using pre-trained vision models and the second method will be using graph neural networks.~~

Unfortunately, my teammate left the project. So, I will just apply various fine-tuning strategies on ViT models for action recognition using the dataset we collected.


## Directories

```
DL-Project-Action-Recognition/
├── README.md
├── dl_action_dataset   # Dataset for action recognition
├── prepare_dataset.py  # Script for preparing the dataset
├── ...
├── main.py        # main code for graph-based action recognition
└── scripts             # Tools for data preparation
    └── ...
```


## Environment

You can use following codes to create the required environments:

```
conda create -n myenv python=3.8


conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge 

pip install transformers datasets # huggingface

pip install natsort torchmetrics matplotlib gdown

```

## Dataset

We curated a video dataset of 4 actions with 5 repetitions each. Since our model is image-based, we extracted the frames, so in total we have X images. The dataset can be downloaded from [here](https://drive.google.com/file/d/16pMYcodc5oshaTIcQYcHrK2VDG7vUuLa/view?usp=drive_link).

## Fine-tuning ViT Models 

...
...

## Results
...
...