# -*- coding: utf-8 -*-
"""
Library for [TEXT].

@author: niels, 2025
"""

# =============================================================================
# Packages
# =============================================================================

import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import sys
from typing import Union

SCALE = 1

def rescale_image(image, new_size=224, pad_value=(0, 0, 0)):
    if not isinstance(new_size, tuple):
        new_size = (int(new_size), int(new_size))
    if len(new_size) == 1:
        new_size = new_size * 2
    image.thumbnail(new_size)
    image = ImageOps.pad(image, new_size, color=pad_value)
    return image

def pad_image(image, chunk_size, pad_value=(0, 0, 0)):
    width, height = image.size
    new_width = (width + chunk_size - 1) // chunk_size * chunk_size
    new_height = (height + chunk_size - 1) // chunk_size * chunk_size
    padded_image = Image.new('RGB', (new_width, new_height), pad_value)
    padded_image.paste(image, (0, 0))
    return padded_image

def subdivide_image(image, chunk_size, pad_value=(0, 0, 0)):
    # dim = max(width, height)
    # size = (int(dim*SCALE), int(dim*SCALE))
    size = (int(chunk_size*14), int(chunk_size*14))
    scaled_image = ImageOps.fit(image, size)
    width, height = scaled_image.size
    chunks = []
    for i in range(0, height, chunk_size):
        for j in range(0, width, chunk_size):
            box = (j, i, j + chunk_size, i + chunk_size)
            chunk = scaled_image.crop(box)
            chunks.append(chunk)
    return chunks

def save_chunks(chunks, file_path):
    for idx, chunk in enumerate(chunks):
        chunk.save(f'{file_path}_{idx}.png')

def subdivide_iff_dd(
        image: Union[np.ndarray, Image.Image], 
        outdir_path: Union[str, Path],
        basename: str,
        chunk_size: int = 224,
        pad_value: tuple = (0,0,0),
        ):
    if not isinstance(outdir_path, Path):
        outdir_path = Path(outdir_path)
    if not outdir_path.is_dir():
        outdir_path.mkdir()
    outdir_path = outdir_path / basename
    if not outdir_path.is_dir():
        outdir_path.mkdir()
    if isinstance(image, np.ndarray):
        """Currently uses only the phase."""
        exp = np.exp(1j*image, dtype=np.complex64)
        image = np.zeros(image.shape + (3,), dtype=np.float32)
        image[:,:,0] = np.real(exp).astype(np.float32)
        image[:,:,1] = np.image(exp).astype(np.float32)
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
        chunks = subdivide_image(image, chunk_size, pad_value)
        save_chunks(chunks, outdir_path / basename)
    else:
        print(f'Unsupported image type: {type(image)}')
        sys.exis()
        
if __name__ == "__main__":
    track = "track_002_ascending"
    basename = "20171112_20171124_20171124_20171206"
    outname = "iff_dd_14x14chunks"
    image = Image.open(r"D:\dtu\speciale\ipp\processed\%s\iff_dd\%s.pha.png" % (track, basename))
    outdir_path = Path(r"D:\dtu\speciale\ipp\processed\%s" % track) / outname
    chunk_size = 224
    pad_value = (0,0,0)
    subdivide_iff_dd(image, outdir_path, basename, chunk_size, pad_value)
    # image = rescale_image(image, new_size=2000)
    image.save(outdir_path + '.png')