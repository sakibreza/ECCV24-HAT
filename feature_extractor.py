from models.i3d.extract_i3d import ExtractI3D
from utils.utils import build_cfg_path
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import os
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.get_device_name(0))
# Select the feature type
feature_type = 'i3d'

# Load and patch the config
args = OmegaConf.load(build_cfg_path(feature_type))
args.step_size = 12
args.flow_type = 'raft' # 'pwc'

# Load the model
extractor = ExtractI3D(args)

args.video_paths = os.listdir('./Videos')

# Extract features
for video_path in tqdm(args.video_paths):
    print(f'Extracting for {video_path}')
    feature_dict = extractor.extract('./Videos/'+video_path)
    np.savez('./I3D/'+video_path[:-4]+'.npz', **feature_dict)
    [(print(k), print(v.shape)) for k, v in feature_dict.items()]