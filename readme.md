## This is the official repository for CoRL 2022 paper "See, Hear, and Feel: Smart Sensory Fusion for Robotic Manipulation"
[Project](https://ai.stanford.edu/~rhgao/see_hear_feel/)|[Paper](https://arxiv.org/abs/2212.03858)

## Getting Started
To clone this repo, run:
```
git clone https://github.com/JunzheJosephZhu/see_hear_touch.git
cd see_hear_touch
```

## Setup
To set up the required libraries to train/test a model, run:
```
conda create -n "multimodal" python=3.7 -y && conda activate multimodal
pip install -r requirements.txt
```

## Train/test your own model
For the ResNet Encoder + MSA model described in the original paper, run
```python train_imitation.py --ablation vg_t_ah```
Alternatively, we also provide a modified implementation of TimeSformer that takes multimodel tokens as inputs. To train this, run
```python train_transformer.py --ablation vg_t_ah```

## Run ablation studies
To run ablation studies, change the "--ablation" argument. For example, to train a model with only vision+tactile inputs, run:
```python train_imitation.py --ablation vg_t```


### TODO(for Hao)
- [ ] Add some demo videos
- [ ] Add dataset structure/action space description/example dataset
- [ ] test the setup commands locally
- [ ] provide a pretrained model

