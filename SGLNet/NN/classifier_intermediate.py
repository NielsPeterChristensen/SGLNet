# =============================================================================
# Modules
# =============================================================================

import sys
import numpy as np
from typing import Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# =============================================================================
# Classes
# =============================================================================


class LakeDataset(Dataset):
    
    def __init__(self, lake_path: Union[str, Path], nolake_path: Union[str, Path]):
        
        #-- Verify input type
        if not isinstance(lake_path, (str, Path)):
            print(f"Unknown data type {type(lake_path)}.", file=sys.stderr)
            sys.exit(1)
        if not isinstance(nolake_path, (str, Path)):
            print(f"Unknown data type {type(nolake_path)}.", file=sys.stderr)
            sys.exit(1)
        #-- Verify data file
        if isinstance(lake_path, (str, Path)):
            lake_path = Path(str(lake_path)).resolve()
            if not lake_path.exists():
                print(f"No file found at {str(lake_path)}.", file=sys.stderr)
                sys.exit(1)
            if not lake_path.suffix == '.npy':
                print(f"Unknown file extension {lake_path.suffix}.", file=sys.stderr)
                sys.exit(1)
        if isinstance(nolake_path, (str, Path)):
            nolake_path = Path((nolake_path)).resolve()
            if not nolake_path.exists():
                print(f"No file found at {str(nolake_path)}.", file=sys.stderr)
                sys.exit(1)
            if not nolake_path.suffix == '.npy':
                print(f"Unknown file extension {nolake_path.suffix}.", file=sys.stderr)
                sys.exit(1)
        
        #-- Get data
        self.features = MemmapData(lake_path, nolake_path)
        self.n_lake = self.features.n_lake
        self.n_nolake = self.features.n_nolake
        self.embed_dim = self.features.embed_dim
        #-- Assign labels
        self.labels = torch.zeros((self.features.total_rows, 1), dtype=torch.float32)
        self.labels[:self.n_lake] = 1
        self.labels[self.n_lake:] = 0
            
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
        return torch.tensor(self.features[idx], dtype=torch.float32), self.labels[idx]
    
    
class BinaryClassifier(nn.Module):
    
    def __init__(self, embed_dim: int, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 1)
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()
    
    def forward(self, x):
        if self.normalize:
            x = self.layer_norm(x)
        x = self.fc(x)
        return x
        
    
class MemmapData:
    def __init__(self, lake_path: Union[str, Path], nolake_path: Union[str, Path]):
        self.lake = np.load(lake_path, mmap_mode='r')
        self.nolake = np.load(nolake_path, mmap_mode='r')
        self.n_lake = self.lake.shape[0]
        self.n_nolake = self.nolake.shape[0]
        self.embed_dim = self.nolake.shape[1]
        self.total_rows = self.n_lake + self.n_nolake
        
    def __len__(self):
        return self.total_rows
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.total_rows)
            return np.vstack([self[i] for i in range(start, stop, step)])
        if idx < 0:
            idx += self.total_rows
        if idx >= self.total_rows:
            print("Index out of range", file=sys.stderr)
            sys.exit(1)
        if idx < self.n_lake:
            return self.lake[idx]
        if idx >= self.n_lake and idx < self.total_rows:
            return self.nolake[idx - self.n_lake]
        print(f'Unable to retrieve {idx}. Please check if input is correct.', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import time
    lakefile = r"D:\dtu\speciale\Data\feature_dataset\vit_base16_recta224\lake.npy"
    nolakefile = r"D:\dtu\speciale\Data\feature_dataset\vit_base16_recta224\nolake.npy"
    start_time = time.time()
    dataset = LakeDataset(lakefile, nolakefile)
    print("Execution took: %d seconds" %(time.time() - start_time))