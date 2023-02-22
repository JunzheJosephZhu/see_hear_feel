## This is the official repository for CoRL 2022 paper "See, Hear, and Feel: Smart Sensory Fusion for Robotic Manipulation"
[Project](https://ai.stanford.edu/~rhgao/see_hear_feel/)|[Paper](https://arxiv.org/abs/2212.03858)|[Bibtex](https://ai.stanford.edu/~rhgao/see_hear_feel/bibtex_seehearfeel.txt)

## Getting Started
To clone this repo, run:
```
git clone https://github.com/JunzheJosephZhu/see_hear_touch.git
cd see_hear_touch
```
## Install Dependencies
To set up the required libraries to train/test a model, run:
```
conda create -n "multimodal" python=3.7 -y && conda activate multimodal
pip install -r requirements.txt
```
## Prepare dataset
You can download an example dataset [here](https://drive.google.com/drive/folders/13S6YcKJIIRKctB0SjdiaKEv_mvJEM_pk)

After downloading, unzip and rename the folder to ```data```, and place it under the project folder.

To preprocess the data, run <br>
```python utils/h5py_convert.py```

To split the training/testing dataset, run <br>
```python split_train_val.py```

Brief explanation for the example dataset: Under ```data/test_recordings```, each folder is an episode. ```timestamps.json``` contains the human demo actions and the pose history of the robot, while each subfolder contains a stream of sensory inputs.

## Train/test your own model
For the ResNet Encoder + MSA model described in the original paper, run <br>
```python train_imitation.py --ablation vg_t_ah```

Alternatively, we also provide a modified implementation of TimeSformer that takes multimodel tokens as inputs. To train this, run <br>
```python train_transformer.py --ablation vg_t_ah```

## Run ablation studies
To run ablation studies, change the ```--ablation``` argument. For example, to train a model with only vision+tactile inputs, run <br>
```python train_imitation.py --ablation vg_t```
Here are what each symbol means:

| Symbol      | Description |
| ----------- | ----------- |
| vg      | camera input from a gripper-mounted(first person) camera       |
| vf   | camera input from a fixed perspective        |
| ah   | microphone input from piezo-electric stuck to the platform(i.e. peg insertion base/tube for pouring) |
| ag   | microphone input from piezo-electric mounted on gripper |
| t    | Gelsight sensor input |

## Evaluate your results
To view your model's results, run <br>
```tensorboard --logdir exp{data}{task}```

## Citation
If you find our work relevant, please cite us using the following bibtex:
```
@inproceedings{li2022seehearfeel,
    title={See, Hear, and Feel: Smart Sensory Fusion for Robotic Manipulation},
    author={Hao Li and Yizhi Zhang and Junzhe Zhu and Shaoxiong Wang and Michelle A. Lee and Huazhe Xu and Edward Adelson and Li Fei-Fei and Ruohan Gao and Jiajun Wu},
    booktitle={CoRL},
    year={2022}
}
```

### TODO(for Hao)
- [ ] Add some demo videos
- [ ] test the setup commands locally
- [ ] provide a pretrained vg_t_ah model

