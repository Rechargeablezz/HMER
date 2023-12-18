# CAN



## Description



## How to run

```
# install project
conda create -y -n can python=3.7
conda activate can
conda install --yes -c pytorch=1.7.0 torchvision cudatoolkit=<11.0.221>  # <ur-cuda-version>
pip install -e .
```

## Pipeline



## Datasets



## Training

Check the config file ```config.py``` and train with the CROHME dataset:

```
python train.py --dataset CROHME
```

By default the ```batch size``` is set to 8 and you may need to use a GPU with 32GB RAM to train your model. 

## Testing

Fill in the ```checkpoint``` (pretrained model path) in the config file ```config.py``` and test with the CROHME dataset:

```
python inference.py --dataset CROHME
python inference.py --dataset CROHME --draw_map True
```

Note that the testing dataset path is set in the ```inference.py```.
