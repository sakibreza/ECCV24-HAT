
# HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization (ECCV 2024) 
### Sakib Reza, Yuexi Zhang, Mohsen Moghaddam, Octavia Camps
#### Northeastern University, Boston, United States
{reza.s,zhang.yuex,mohsen,o.camps}@northeastern.edu

## [Arxiv Preprint](https://arxiv.org/abs/2408.06437) 


## Updates
- Aug 22, 2024 - EGTEA pre-extracted features and config files for other datasets added
- Aug 14, 2024 - Arxiv preprint added
- July 7, 2024 - initial code release 

## Installation

### Prerequisites
- Ubuntu 20.04 
- Python 3.10.9 
- CUDA 12.0  

### Requirements
- pytorch==2.0.0
- numpy==1.23.5
- h5py==3.9.0
- ...

To install all required libraries, execute the pip command below.
```
pip install -r requirement.txt
```

## Training

### Input Features
The Kinetics I3D pre-trained feature of EGTEA dataset can be downloaded from [GDrive link](https://drive.google.com/drive/folders/1Zj1B2UZnjPgLrylhKOfu7m_9rkQFa14T?usp=sharing).   
Files should be located in 'data/'.  
You can get other features from the following links -  
- [EPIC-Kitchen 100](https://github.com/happyharrycn/actionformer_release)
- [THUMOS'14](https://github.com/YHKimGithub/OAT-OSN/)
- [MUSES](https://songbai.site/muses/)

### Config Files
The configuration files for EGTEA are already provided in the repository. For other datasets, they can be downloaded from [GDrive link](https://drive.google.com/drive/folders/19__GnM2HZCCDshED9kadsLNAI9XBvrFd?usp=sharing).

### Training Model 
To train the main HAT model, execute the command below.
```
python main.py --mode=train --split=[split #]*
```
*If the dataset has any splits (e.g., EGTEA has 4 splits)

To train the post-processing network (OSN), execute the commands below.
```
python supnet.py --mode=make --inference_subset=train --split=[split #]
python supnet.py --mode=make --inference_subset=test --split=[split #]
python supnet.py --mode=train --split=[split #]
```


## Testing
To test HAT, execute the command below.
```
python main.py --mode=test --split=[split #]
```

## Citing HAT
Please cite our paper in your publications if it helps your research:

```BibTeX
@inproceedings{reza2022history,
  title={HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization},
  author={Reza, Sakib and Zhang, Yuexi and Moghaddam, Mohsen and Camps, Octavia},
  booktitle={European Conference on Computer Vision},
  pages={XXX--XXX},
  year={2024},
  organization={Springer}
}
```

## Acknowledgment
This repository is created based on the repository of the baseline work [OAT-OSN](https://github.com/YHKimGithub/OAT-OSN/).
