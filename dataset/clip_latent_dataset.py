import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import json


def get_dataloader(phase, cfg, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle
    
    dataset = CLIPLatentDataset(
        latent_path=cfg.latent_path,  # path to all_zs_ckpt1000.h5
        clip_path=cfg.clip_path,      # path to CLIP_feats.json
        phase=phase
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=is_shuffle,
        num_workers=cfg.num_workers, 
        worker_init_fn=np.random.seed()
    )
    return dataloader


class CLIPLatentDataset(Dataset):
    def __init__(self, latent_path, clip_path, phase='train'):
        super(CLIPLatentDataset, self).__init__()
        self.phase = phase
        
        # Load CLIP features
        print(f"Loading CLIP features from {clip_path}")
        with open(clip_path, 'r') as f:
            clip_data = json.load(f)
            self.clip_data = {item['id']: torch.tensor(item['clip_feats']).float() 
                            for item in clip_data[phase]}
            
        # Load latent codes
        print(f"Loading latent codes from {latent_path}")
        with h5py.File(latent_path, 'r') as f:
            self.latents = torch.tensor(f[f'{phase}_zs'][:]).float()
            
        # Get ordered list of IDs from CLIP data to maintain consistent ordering
        self.ids = list(self.clip_data.keys())
        
        print(f"Loaded {len(self.ids)} samples for {phase}")
        print(f"CLIP feature shape: {next(iter(self.clip_data.values())).shape}")  # (24, 512) expected
        print(f"Latent shape: {self.latents.shape}")  # (N, 256) expected

    def __getitem__(self, index):
        model_id = self.ids[index]
        clip_feat = self.clip_data[model_id]  # Shape: (24, 512)
        latent = self.latents[index]          # Shape: (256,)
        
        # Average the CLIP features across views
        clip_feat = clip_feat.mean(dim=0)     # Shape: (512,)
        
        return {
            'clip_feature': clip_feat,        # Shape: (512,)
            'latent': latent,                 # Shape: (256,)
            'id': model_id
        }

    def __len__(self):
        return len(self.ids) 