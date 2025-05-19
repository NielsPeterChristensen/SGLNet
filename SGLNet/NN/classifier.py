#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================


"""
Custom classifier network to be used on downstream DINO [1] latent features.

:Authors
    NPKC / 18-02-2025 / creation / s203980@dtu.dk

:Todo
    Add more comments

:References
    [1] Mathilde Caron et. al. Emerging Properties in Self-Supervised Vision 
    Transformers https://arxiv.org/abs/2104.14294

:Note:
    Requires DINO [1]
"""


# =============================================================================
# Modules
# =============================================================================

import sys
import numpy as np
from typing import Union
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# =============================================================================
# Classes
# =============================================================================


class LakeDataset(Dataset):
    
    def __init__(self, features_path: Union[str, Path], labels_path: Union[str, Path], config_path: Union[str, Path], use_mmap: bool):
        
        #-- Verify input type
        if not isinstance(features_path, (str, Path)):
            print(f"Unknown data type {type(features_path)}.", file=sys.stderr)
            sys.exit(1)
            
        if not isinstance(labels_path, (str, Path)):
            print(f"Unknown data type {type(labels_path)}.", file=sys.stderr)
            sys.exit(1)
            
        if not isinstance(config_path, (str, Path)):
            print(f"Unknown data type {type(config_path)}.", file=sys.stderr)
            sys.exit(1)
            
        #-- Verify data file
        self.features_path = Path(str(features_path)).resolve()
        if not self.features_path.exists():
            print(f"No file found at {str(self.features_path)}.", file=sys.stderr)
            sys.exit(1)
        if not self.features_path.suffix == '.dat':
            print(f"Unknown file extension {self.features_path.suffix}.", file=sys.stderr)
            sys.exit(1)
            
        self.labels_path = Path((labels_path)).resolve()
        if not self.labels_path.exists():
            print(f"No file found at {str(self.labels_path)}.", file=sys.stderr)
            sys.exit(1)
        if not self.labels_path.suffix == '.dat':
            print(f"Unknown file extension {self.labels_path.suffix}.", file=sys.stderr)
            sys.exit(1)
            
        self.config_path = Path((config_path)).resolve()
        if not self.config_path.exists():
            print(f"No file found at {str(self.config_path)}.", file=sys.stderr)
            sys.exit(1)
        if not self.config_path.suffix == '.txt':
            print(f"Unknown file extension {self.config_path.suffix}.", file=sys.stderr)
            sys.exit(1)
        
        #-- Get data
        with open(self.config_path, 'r') as file:
            config = json.load(file)
        self.feature_shape = tuple(config["feature_shape"]) # new
        self.label_shape = tuple(config["label_shape"]) # new
        # self.features = np.memmap(features_path, dtype=np.float32, mode='r', shape=tuple(config['feature_shape']))
        # self.labels = np.memmap(labels_path, dtype=np.float32, mode='r', shape=tuple(config['label_shape']))
        self.n_lake = config['n_lake']
        self.n_nolake = config['n_nolake']
        self.embed_dim = config['embed_dim']
        self.use_mmap = use_mmap
        
        if self.use_mmap:
            self.features = None
            self.labels = None
        else:
            self.features = np.array(np.memmap(self.features_path, dtype=np.float32, mode="r", shape=self.feature_shape))
            self.labels = np.array(np.memmap(self.labels_path, dtype=np.float32, mode="r", shape=self.label_shape))
            
    def __len__(self):
        return self.feature_shape[0]
    
    def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
        if self.use_mmap:
            features = np.memmap(self.features_path, dtype=np.float32, mode="r", shape=self.feature_shape)[idx]
            labels = np.memmap(self.labels_path, dtype=np.float32, mode="r", shape=self.label_shape)[idx]
            return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
        else:
            return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
    

class LakeDataset_v2(Dataset):
    
    def __init__(self, feature_paths: list[Path], label_paths: list[Path], meta_paths: list[Path]):
                
        self.feature_paths = feature_paths
        self.label_paths = label_paths
        self.meta_paths = meta_paths
        
        #-- Get data
        self.feature_shapes = []
        self.label_shapes = []
        self.lengths = [0]
        for meta_path in self.meta_paths:
            with open(meta_path, 'rb') as file:
                metadata = pickle.load(file)
                self.embed_dim = metadata['Setup']['embed_dim']
                self.lengths.append(metadata['Setup']['total_length'])
                self.feature_shapes.append((self.lengths[-1], self.embed_dim))
                self.label_shapes.append((self.lengths[-1], 1))
        self.ind = np.cumsum(self.lengths).astype(int)
        
        self.features = torch.zeros((self.ind[-1], self.embed_dim), dtype=torch.float32)
        self.labels = torch.zeros((self.ind[-1], 1), dtype=torch.float32)
            
        for i in range(len(self.ind)-1):
            self.features[self.ind[i]:self.ind[i+1]] = torch.from_numpy(np.array(np.memmap(self.feature_paths[i], dtype=np.float32, mode="r", shape=self.feature_shapes[i])))
            self.labels[self.ind[i]:self.ind[i+1]] = torch.from_numpy(np.array(np.memmap(self.label_paths[i], dtype=np.float32, mode="r", shape=self.label_shapes[i])))
        
        self.n_lake = int(sum(self.labels))
        self.n_nolake = int(self.ind[-1] - self.n_lake)
            
    def __len__(self):
        return self.ind[-1]
    
    def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
        return self.features[idx], self.labels[idx]
    

    
class BinaryClassifier(nn.Module):
    
    def __init__(self, embed_dim: int, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 1)
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        if self.normalize:
            x = self.layer_norm(x)
        x = self.fc(x)
        return x
    
    def activation(self, x):
        return self.act(x)


if __name__ == "__main__":
    import time
    lakefile = r"D:\dtu\speciale\Data\feature_dataset\vit_base16_recta224\lake.npy"
    nolakefile = r"D:\dtu\speciale\Data\feature_dataset\vit_base16_recta224\nolake.npy"
    start_time = time.time()
    dataset = LakeDataset(lakefile, nolakefile)
    print("Execution took: %d seconds" %(time.time() - start_time))