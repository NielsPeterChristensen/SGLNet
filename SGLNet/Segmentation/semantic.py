# -*- coding: utf-8 -*-

# =============================================================================
# Docstring
# =============================================================================


"""
[Description]

:Authors
    NPKC / 01-05-2025 / creation / s203980@dtu.dk

:Todo
    Create binary masks from dino_segmentation

:References
    [1] Mathilde Caron et. al. Emerging Properties in Self-Supervised Vision 
    Transformers https://arxiv.org/abs/2104.14294
    [2] Oriane Siméoni et al. Localizing Objects with Self-Supervised
    Transformers and no Labels https://arxiv.org/abs/2109.14279

:Note:
    Requires DINO [1] and LOST [2]
"""

# =============================================================================
# Packages
# =============================================================================

#-- Utilities
import sys
import numpy as np
import numpy.typing as npt
import torch
import skimage as ski
import matplotlib.pyplot as plt
import scipy
import cv2
import scipy.interpolate as si
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from PIL import Image
#-- My own
import SGLNet.Corefunc.utils as utils
#-- From LOST [2]
import SGLNet.lost.object_discovery as object_discovery


# =============================================================================
# Functions
# =============================================================================

def upscale_mask(arr: npt.NDArray, scale: int) -> npt.NDArray:
    return zoom(arr, zoom=(scale,scale), order=0)


def DINO_segmentation(attn: torch.Tensor, num_heads: int, dims: tuple[int, int], mass_threshold: float = 0.6) -> npt.NDArray:
    """
    Dino attention segmentation from [1]. Modified from:
    https://github.com/facebookresearch/dino/blob/main/visualize_attention.py
    """
    
    w_featmap, h_featmap = dims
    
    #-- Check dimension of attn tensor
    if attn.ndim == 3:
        attn = attn[None]

    #-- Get attentions corresponding to [CLS] token
    attentions = attn[0, :, 0, 1:].reshape(num_heads, -1)

    #-- Threshold attention mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    thresholded_attn = cumval > (1 - mass_threshold)
    idx2 = torch.argsort(idx)
    for h in range(num_heads):
        thresholded_attn[h] = thresholded_attn[h][idx2[h]]
    thresholded_attn = thresholded_attn.reshape(num_heads, w_featmap, h_featmap).float()
    mask_per_head = np.asarray(thresholded_attn)
    sum_attn = np.sum(mask_per_head, axis=0) / num_heads
    # gauss_attn = ski.filters.gaussian(sum_attn, sigma=2)
    # mask = gauss_attn > head_threshold
    return sum_attn
    

def PCA_segmentation(embeddings: torch.Tensor, dims: tuple[int, int], pc_th: float = 0, sign_method: int = 0) -> torch.Tensor:
    
    if sign_method not in [0, 1, 2, 3]:
        print(f"Invalid input {sign_method} for sign_method. Must be 0, 1, 2, or 3", file=sys.stderr)
        sys.exit(1)
    
    pca_model = PCA(n_components=1)
    first_pc = pca_model.fit_transform(embeddings).flatten().reshape(dims)
    
    #-- Simple threshold
    if sign_method == 0:
        mask = first_pc > pc_th
        
    #-- Choose sign that maximizes inter-class variance
    if sign_method == 1:
        positive_mean = np.mean(first_pc[first_pc > 0])
        negative_mean = np.mean(first_pc[first_pc < 0])
        if abs(positive_mean - negative_mean) > abs(negative_mean - positive_mean):
            mask = first_pc 
        else:
            mask = first_pc*(-1)

    #-- If foreground is usually smaller than background
    if sign_method == 2:
        if np.sum(first_pc > 0) < np.sum(first_pc < 0):
            mask = first_pc
        else:
            mask = first_pc*(-1)
    
    #-- Determine the right sign based on distribution properties
    if sign_method == 3:
        if np.mean(first_pc[first_pc > 0]) > abs(np.mean(first_pc[first_pc < 0])):
            mask = first_pc
        else:
            mask = first_pc*(-1)  

    return mask


def consistent_PCA_segmentation(patch_embeddings_list: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Generate consistent principal components from PCA of patch embeddings across 
    multiple crops of the same image.
    
    Args:
        patch_embeddings_list: List of patch embeddings from different crops
    
    Returns:
        List of first principal components with consistent direction
    """
    if not patch_embeddings_list:
        return []
    
    #-- Run PCA on the list of embeddings to get a single shared direction
    all_embeddings = np.concatenate([emb.reshape(-1, emb.shape[-1]) for emb in patch_embeddings_list])
    pca = PCA(n_components=1)
    pca.fit(all_embeddings)
    
    #-- Project embeddings from each chunk onto shared direction
    first_pcs = []
    for embeddings in patch_embeddings_list:
        h, w, emb_dim = embeddings.shape
        reshaped = embeddings.reshape(-1, emb_dim)
        first_pc = pca.transform(reshaped)[:,0].reshape(h, w)
        first_pcs.append(first_pc*(-1))
    
    return first_pcs
    

def LOST_segmentation(feats, dims, scales, init_image_size, k_patches: int = 100) -> torch.Tensor:
     
    if feats.ndim == 2:
        feats = feats[None]
    
    #-- Run LOST algorithm [2]
    pred, A, scores, seed = object_discovery.lost(
        feats,
        dims,
        scales,
        init_image_size,
        k_patches=k_patches,
    )
    #-- Compute patch correlation with seed patch
    A = A.clone().cpu().numpy().copy()
    corr = A[seed, :].copy().reshape(dims)
    #-- Compute correlation degree
    A[A < 0] = 0
    A[A > 0] = 1
    deg = 1 / A.sum(-1) * dims[0] * dims[1]
    deg = deg.reshape(dims)
    
    return corr, deg


def smooth_outlines(mask: npt.NDArray, sigma: float = 0, smoothness: float = 5.0, n_points: int = 200) -> list:
    
    def smooth_outline(cnt, smoothness, n_points):
        cnt = cnt[:, 0, :]  # shape (N, 2)
        if len(cnt) < 3:
            return cnt
        try:
            tck, u = si.splprep([cnt[:, 0], cnt[:, 1]], s=smoothness, per=True)
            u_fine = np.linspace(0, 1, n_points)
            x_fine, y_fine = si.splev(u_fine, tck)
            return np.stack((x_fine, y_fine), axis=-1)
        except Exception:
            return cnt  # fallback if spline fails
    
    mask = (mask > 0).astype(np.uint8) * 255
    if sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    outlines, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    
    smoothed_outlines = []
    for o in outlines:
        s = smooth_outline(o, smoothness, n_points)
        if s.shape == (n_points, 2):
            smoothed_outlines.append(s)
    smoothed_outlines = np.array(smoothed_outlines)
    # smoothed_outlines = np.array([smooth_outline(o, smoothness, n_points) for o in outlines])
    # if len(smoothed_outlines) == 0:
    #     return np.zeros((0, 200, 2), dtype=int)
    return smoothed_outlines
    

# =============================================================================
# Classes
# =============================================================================

class BinaryMask:
    
    def __init__(self, shape: npt.ArrayLike, patch_size: int, rgb: npt.ArrayLike = (0, 255, 0)):
        self.shape = shape
        self.patch_size = patch_size
        self.rgb = rgb
        self.mask = self.create_mask()

    def __setitem__(self, bbox, binary_mask):
        (sa0, li0, sa1, li1) = bbox
        sa1 = np.minimum(sa1, self.shape[0])
        li1 = np.minimum(li1, self.shape[1])
        alpha_mask = (binary_mask).astype(np.uint8)*133
        upscaled_mask = upscale_mask(alpha_mask, self.patch_size)
        self.mask[li0:li1, sa0:sa1, 3] = np.maximum(self.mask[li0:li1, sa0:sa1, 3], 
                                                       upscaled_mask[:(li1-li0), :(sa1-sa0)])
    
    def __getitem__(self, bbox):
        (sa0, li0, sa1, li1) = bbox
        return self.mask[li0:li1, sa0:sa1, :]
    
    def __call__(self):
        return Image.fromarray(self.mask, mode="RGBA")
        
    def create_mask(self, shape: npt.ArrayLike = None, rgb: npt.ArrayLike = None) -> npt.NDArray[np.uint8]:
        #-- Get input
        shape = shape or self.shape
        rgb = rgb or self.rgb
        #-- Create RGBA array of shape=(width,height) !NOT (row,col)!
        mask = np.zeros((shape[1], shape[0], 4), dtype=np.uint8)
        mask[..., 0] = rgb[0]
        mask[..., 1] = rgb[1]
        mask[..., 2] = rgb[2]
        return mask
    
    def apply_to(self, img: Image) -> Image:
        img = img.convert('RGBA')
        mask = self()
        img.paste(mask, (0, 0), mask)
        return img
    

class SalientMask:
    
    def __init__(self, shape: npt.ArrayLike, patch_size: int, chunk_size: int, rgb: npt.ArrayLike = (0, 255, 0), accum: str = 'max') -> None:
        self.patch_size = patch_size
        self.chunk_size = chunk_size
        self.shape = tuple([utils.first_multiple(s, self.chunk_size)//self.patch_size for s in shape])
        self.rgb = rgb
        self.mask = self.create_array()
        self.count = self.create_array(dtype=np.int8)
        assert accum == 'max' or accum == 'mean', f"Unknown accumulation method {accum}."
        self.accum = accum
        
    def __setitem__(self, bbox, arr) -> None:
        (sa0, li0, sa1, li1) = (bbox / self.patch_size).astype(int)
        sa1 = np.minimum(sa1, self.shape[0])
        li1 = np.minimum(li1, self.shape[1])
        if self.accum == 'max':
            self.mask[li0:li1, sa0:sa1] = np.maximum(arr[:(li1-li0), :(sa1-sa0)], self.mask[li0:li1, sa0:sa1])
            self.count[li0:li1, sa0:sa1] = 1
        if self.accum == 'mean':
            self.mask[li0:li1, sa0:sa1] += arr[:(li1-li0), :(sa1-sa0)]
            self.count[li0:li1, sa0:sa1] += 1
    
    # def __call__(self, threshold: float = 0, sigma: float = 0, min_patch: int = 10, connectivity: int = 1) -> Image:
    #     """Returns the RGBA mask when function is called."""
    #     self.make(threshold, sigma, min_patch, connectivity)
    #     rgba_mask = self.create_rgba()
    #     rgba_mask[self.upscaled_mask, 3] = 133
    #     return Image.fromarray(rgba_mask, mode="RGBA")
    
    def __call__(self, threshold: float = 0, sigma: float = 0, min_patch: int = 10, connectivity: int = 1) -> Image:
        """Returns the RGBA mask when function is called."""
        if not hasattr(self, 'upscaled_mask'):
            self.make(threshold, sigma, min_patch, connectivity)
        mask = (self.upscaled_mask*255).astype(np.uint8)
        return Image.fromarray(mask, mode="L")
    
    def get_mask(self, bbox, upscale: bool = True) -> npt.NDArray:
        (sa0, li0, sa1, li1) = (bbox / self.patch_size).astype(int)
        mask = self.mask[li0:li1, sa0:sa1]
        if upscale is True:
            mask = upscale_mask(mask, self.patch_size)
        return mask
    
    def get_norm_mask(self, bbox, upscale: bool = True) -> npt.NDArray:
        (sa0, li0, sa1, li1) = (bbox / self.patch_size).astype(int)
        mask = self.normalized_mask[li0:li1, sa0:sa1]
        if upscale is True:
            mask = upscale_mask(mask, self.patch_size)
        return mask
    
    def get_smooth_mask(self, bbox, upscale: bool = True) -> npt.NDArray:
        (sa0, li0, sa1, li1) = (bbox / self.patch_size).astype(int)
        mask = self.smoothed_mask[li0:li1, sa0:sa1]
        if upscale is True:
            mask = upscale_mask(mask, self.patch_size)
        return mask
    
    def get_binary_mask(self, bbox, upscale: bool = True) -> npt.NDArray:
        (sa0, li0, sa1, li1) = (bbox / self.patch_size).astype(int)
        mask = self.binary_mask[li0:li1, sa0:sa1]
        if upscale is True:
            mask = upscale_mask(mask, self.patch_size)
        return mask
    
    def get_filter_mask(self, bbox, upscale: bool = True) -> npt.NDArray:
        (sa0, li0, sa1, li1) = (bbox / self.patch_size).astype(int)
        mask = self.filtered_mask[li0:li1, sa0:sa1]
        if upscale is True:
            mask = upscale_mask(mask, self.patch_size)
        return mask
    
    def make(self, threshold: float = 0, sigma: float = 0, min_patches: int = 10, connectivity: int = 1) -> None:
        self.normalized_mask = np.zeros_like(self.mask, dtype=np.float32)
        self.normalized_mask[self.count>0] = self.mask[self.count>0] / self.count[self.count>0]
        self.smoothed_mask = ski.filters.gaussian(self.normalized_mask, sigma)
        self.binary_mask = self.smoothed_mask > threshold
        self.filtered_mask = self.filter(min_patches, connectivity)
        self.upscaled_mask = upscale_mask(self.filtered_mask, self.patch_size)
    
    def create_array(self, shape: npt.ArrayLike = None, dtype: type = np.float32) -> npt.NDArray:
        """Create base array"""
        #-- Get input
        shape = shape or self.shape
        #-- Create RGBA array of shape=(width,height) !NOT (row,col)!
        arr = np.zeros((shape[1], shape[0]), dtype=dtype)
        return arr
    
    def create_rgba(self, shape: npt.ArrayLike = None, rgb: npt.ArrayLike = None) -> npt.NDArray[np.uint8]:
        """Create RGBA base mask as numpy array"""
        #-- Get input
        shape = shape or tuple([int(s*self.patch_size) for s in self.shape])
        rgb = rgb or self.rgb
        #-- Create RGBA array of shape=(width,height) !NOT (row,col)!
        mask = np.zeros((shape[1], shape[0], 4), dtype=np.uint8)
        mask[..., 0] = rgb[0]
        mask[..., 1] = rgb[1]
        mask[..., 2] = rgb[2]
        return mask
    
    def filter(self, min_patches: int = 10, connectivity : int = 1) -> npt.NDArray:
        """Filter binary mask to exclude connected components with fewer than min_patches."""
        structure = scipy.ndimage.generate_binary_structure(2, connectivity)
        labeled_array, num_features = scipy.ndimage.label(self.binary_mask, structure=structure)
        sizes = np.bincount(labeled_array.ravel())
        mask_sizes = sizes >= min_patches
        mask_sizes[0] = 0  # Always exclude background (label 0)
        return mask_sizes[labeled_array]
        
    def apply_to(self, img: Image, threshold: float = 0, sigma: float = 0, min_patch: int = 10, connectivity: int = 1) -> Image:
        """Apply RGBA mask to PIL.Image"""
        img = img.convert('RGBA')
        if not hasattr(self, 'upscaled_mask'):
            self.make(threshold, sigma, min_patch, connectivity)
        rgba_mask = self.create_rgba()
        rgba_mask[self.upscaled_mask, 3] = 133
        mask = Image.fromarray(rgba_mask, mode="RGBA")
        img.paste(mask, (0, 0), mask)
        return img
    
    def get_outlines(self, sigma: float = 0, smoothness: float = 5.0, n_points: int = 200) -> list:
        if not hasattr(self, 'upscaled_mask'):
            print(("Warning. Method .make(...) has not be executed yet,"
                   " so it is now performed with default settings."), file=sys.stderr)
            self.make()
        return smooth_outlines(self.upscaled_mask, sigma, smoothness, n_points)
    
    def get_salient(self):
        if not hasattr(self, 'normalized_mask'):
            self.normalized_mask = np.zeros_like(self.mask, dtype=np.float32)
            self.normalized_mask[self.count>0] = self.mask[self.count>0] / self.count[self.count>0]
        upscaled = upscale_mask(self.normalized_mask, self.patch_size)
        idx = upscaled[:, :] == 0
        norm = (upscaled - upscaled.min()) / (upscaled.max() - upscaled.min() + 1e-8)
        cmap = plt.get_cmap('inferno')
        rgba = (cmap(norm) * 255).astype(np.uint8)
        rgba[idx] = 0
        return Image.fromarray(rgba, mode='RGBA')
