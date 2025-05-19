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
# Macros
# =============================================================================

DataType = Union[str, Path, np.ndarray, torch.tensor]

# =============================================================================
# Classes
# =============================================================================

class LakeDataset(Dataset):
    
    def __init__(self, lake_data: DataType, nolake_data: DataType):
        
        #-- Get lake_data data
        if isinstance(lake_data, str):
            lake_data = Path(lake_data).resolve()
        if isinstance(lake_data, Path):
            if not lake_data.exists():
                print(f"No file found at {str(lake_data)}.", file=sys.stderr)
                sys.exit(1)
            if lake_data.suffix == '.npy':#'.npz':
                lake_data = np.load(lake_data)#['data']
            elif lake_data.suffix == '.pt':
                lake_data = torch.load(lake_data)
            else:
                print(f"Unknown file extension {lake_data.suffix}.", file=sys.stderr)
                sys.exit(1)
        if isinstance(lake_data, np.ndarray):
            lake_data = torch.tensor(lake_data)
        if not isinstance(lake_data, torch.Tensor):
            print(f"Unexpected type {type(lake_data)} for lake_data.", file=sys.stderr)
            sys.exit(1)
        if not lake_data.dim() == 2:
            print(f"Expected lake_data data to be 2D but got {lake_data.dim()}D array.", file=sys.stderr)
            sys.exit(1)
        lake_data.to(torch.float32)
        
        #-- Get nolake_data data
        if isinstance(nolake_data, str):
            nolake_data = Path(nolake_data).resolve()
        if isinstance(nolake_data, Path):
            if not nolake_data.exists():
                print(f"No file found at {str(nolake_data)}.", file=sys.stderr)
                sys.exit(1)
            if nolake_data.suffix == '.npy':#'.npz':
                nolake_data = np.load(nolake_data)#['data']
            elif lake_data.suffix == '.pt':
                lake_data = torch.load(lake_data)
            else:
                print(f"Unknown file extension {nolake_data.suffix}.", file=sys.stderr)
                sys.exit(1)
        if isinstance(nolake_data, np.ndarray):
            nolake_data = torch.tensor(nolake_data)
        if not isinstance(nolake_data, torch.Tensor):
            print(f"Unexpected type {type(nolake_data)} for nolake_data.", file=sys.stderr)
            sys.exit(1)
        if not nolake_data.dim() == 2:
            print(f"Expected nolake_data data to be 2D but got {nolake_data.dim()}D array.", file=sys.stderr)
            sys.exit(1)
        nolake_data.to(torch.float32)
            
        #-- Assign labels
        lake_labels = torch.ones((len(lake_data), 1), dtype=torch.float32)
        nolake_labels = torch.zeros((len(nolake_data), 1), dtype=torch.float32)
        self.n_lake = len(lake_labels)
        self.n_nolake = len(nolake_labels)
        self.embed_dim = nolake_data.size(1)
        
        #-- Concatenate to single dataset
        self.features = torch.cat((lake_data, nolake_data), dim=0)
        self.labels = torch.cat((lake_labels, nolake_labels), dim=0)
        
        #-- Normalize
        # if normalize:
        #     self.features = self.features / self.features.norm(p=2, dim=1, keepdim=True)
            
    def __len__(self):
        return len(self.features)
    
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
        # self.apply(self._init_weights)
    
    def forward(self, x):
        if self.normalize:
            x = self.layer_norm(x)
        x = self.fc(x)
        return x
    
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
        
    

if __name__ == "__main__":
    import time
    lakefile = r"D:\dtu\speciale\Data\feature_dataset\vit_base16_recta224\lake.npz"
    nolakefile = r"D:\dtu\speciale\Data\feature_dataset\vit_base16_recta224\nolake.npz"
    start_time = time.time()
    dataset = LakeDataset(lakefile, nolakefile)
    print("Execution took: %d seconds" %(time.time() - start_time))