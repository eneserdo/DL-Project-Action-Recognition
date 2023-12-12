# DL-Project-Action-Recognition

We will compare two separate methods for action recognition on still images. The first method will be using pre-trained vision models and the second method will be using graph neural networks. 

## Directory Structure

```
DL-Project-Action-Recognition/
├── README.md
├── graph_module        # Module for graph-based action recognition  
│   └── ...
├── image_module        # Module for image-based action recognition  
│   └── ...
├── run_graph.py        # main code for graph-based action recognition
├── run_image.py        # main code for graph-based action recognition
└── scripts             # Tools for data preparation
    └── ...
```


## Environment

You can use following codes to create the required environments:

```
conda create -n myenv python=3.8

conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge 

conda install pyg==2.2 -c pyg

pip install natsort torchmetrics matplotlib gdown
```

## Dataset

We curated a dataset of 4 actions with 5 repetitions each. The dataset can be downloaded from [here](). 

## Fine-tuning ViT Models (Not sure)
...
This part was completed by Ömür Can Ünlü



## Graph Neural Networks 
...
This part was completed by Enes Erdoğan


## Results
...


