#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subdivide phase images (pngs) into chunks and store

@author: niels, 2025
"""

# =============================================================================
# Packages
# =============================================================================

from pathlib import Path
import sys
from typing import Union
from argparse import ArgumentParser
from glob import glob
import SGLNet.Corefunc.chunk_loader as chunk_loader
import SGLNet.Corefunc.utils as utils
from tqdm import tqdm
import numpy as np
from PIL import Image


CLASS_LAKE = 'lake'
CLASS_NO_LAKE = 'nolake'


def parse_arguments():
    """
    Import arguments parsed through the command line call.

    Returns
    -------
    ArgumentParser.parse_args()
        Class containing the parsed arguments.

    """
    parser = ArgumentParser(description="Subdivide iff_dd image into small chunks.")
    
    parser.add_argument(
        "iff_dd_paths",
        nargs = '*',
        metavar = "./track_<xxx>_<pass>/iff_dd/<iff_dd_interval>.pha.png",
        help = "Path/to/iff_dd_images. Wildcards are possible.")
    parser.add_argument(
        "-o",
        "--out_dir",
        type = str,
        default = None,
        metavar = "Path/to/out_dir",
        help = "Path to output directory. It is created if it does not exist.\n" \
               + "Default location is ../../subdivided relative to iff_dd_paths.")
    parser.add_argument(
        "--chunk_size",
        type = int,
        default = 224,
        metavar = '<class int>',
        help = "Size of subdivided image chunks. Default is 224.")
    parser.add_argument(
        "--overlap",
        type = utils.bool_flag,
        default = False,
        metavar = '<class bool>',
        help = "Whether to use 50%% overlap on images. Default is False.")
    parser.add_argument(
        "--downscale_factor",
        type = list,
        default = 1, # [1, 2, 4]
        metavar = '<class int> or <class list>',
        help = "Downscaling factor(s) as integer or list of integers. Default is 1.")
    parser.add_argument(
        "--form",
        type = str,
        default = 'phase',
        metavar = '<class str>',
        help = "Data form. Options are ['image', 'phase', 'recta', 'polar']. Default is 'phase'.")
    parser.add_argument(
        "--skip_class",
        type = str,
        default = None,
        metavar = '<class str>',
        help = "Skip class with given label (either lake or nolake). Default is None.")
    parser.add_argument(
        "--save_bbox",
        type = utils.bool_flag,
        default = False,
        metavar = '<class bool>',
        help = "Save all bboxes in txt file. Default is False.")
    parser.add_argument(
        "--pxcount",
        type = int,
        default = 2500,
        metavar = '<class int>',
        help = "Number of lake pixels (from mask) required within a chunk before" \
               + "it is classified as a lake. Default is 2500.")
    
    return parser.parse_args()


def make_dir(path: Path):
    if not path.is_dir():
        path.mkdir()
        
        
def make_out_dirs(out_dir: Path):
    #-- Check out directory        
    out_dir = Path(str(out_dir))
    make_dir(out_dir)
    #-- Create lake subdirectory
    lake_dir = out_dir / CLASS_LAKE
    make_dir(lake_dir)
    #-- Create no-lake subdirectory
    nolake_dir = out_dir / CLASS_NO_LAKE
    make_dir(nolake_dir)
    return out_dir, lake_dir, nolake_dir
    
        
def does_chunk_overlap(chunk_box, EventCoords, margin: int = 0):
    """
    Checks if the given chunk (x_start, y_start, x_end, y_end) overlaps any bounding box.
    """
    x_start, y_start, x_end, y_end = chunk_box

    for by_min, by_max, bx_min, bx_max in zip(EventCoords.li0, 
                                               EventCoords.li1, 
                                               EventCoords.sa0, 
                                               EventCoords.sa1):
        overlap_x = not (x_end <= (bx_min+margin) or x_start >= (bx_max-margin))
        overlap_y = not (y_end <= (by_min+margin) or y_start >= (by_max-margin))
        
        if overlap_x and overlap_y:
            return True

    return False

def does_chunk_overlap_lake(bbox, label: np.ndarray, pxcount: int = 2500):
    crop = label[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    count = np.count_nonzero(crop)
    if count >= pxcount:
        return True
    return False
    
def load_mask(mask_path, loader_shape):
    mask = ~np.array(Image.open(mask_path)).astype(bool)
    y_pad = loader_shape[0]-mask.shape[0]
    x_pad = loader_shape[1]-mask.shape[1]
    return np.pad(mask, ((0, y_pad), (0, x_pad)), mode='constant', constant_values=False)


def main(files: Union[str, list], out_dir: str, chunk_size: int, overlap: bool, downscale_factor: Union[int, list], form: str, skip_class: str, save_bbox: bool, pxcount: int):
    
    #-- Check files input
    if isinstance(files, str):
        files = glob(files)
    if not isinstance(files, list):
        print(f'Expected positional arguments to be <class str> or <class list> but got {type(files)}.')
        sys.exit()
    files = [Path(f) for f in files]
    #-- Check out_dir input
    if not isinstance(out_dir, str) and out_dir is not None:
        print(f'Expected --out_dir to be <class str> but got {type(out_dir)}.')
        sys.exit()
    #-- Check chunk_size
    if not isinstance(chunk_size, int):
        print(f'Expected --chunk_size to be <class int> but got {type(chunk_size)}.')
        sys.exit()
    #-- Check overlap
    if not isinstance(overlap, bool):
        print(f'Expected --overlap to be <class bool> but got {type(overlap)}.')
        sys.exit()
    #-- Check downscale_factor
    if isinstance(downscale_factor, int):
        downscale_factor = [downscale_factor]
    if not isinstance(downscale_factor, list):
        print(f'Expected --downscale_factor to be <class int> or <class list> but got {type(downscale_factor)}.')
        sys.exit()
    #-- Check skip_class
    if not isinstance(skip_class, str) and skip_class is not None:
        print(f'Expected --skip_class to be <class str> but got {type(skip_class)}.')
        sys.exit()
    if skip_class is not None and skip_class not in [CLASS_LAKE, CLASS_NO_LAKE]:
        print(f'--skip_class of {skip_class} not found in {[CLASS_LAKE, CLASS_NO_LAKE]}.')
    #-- Check save_bbox
    if not isinstance(save_bbox, bool):
        print(f'Expected --save_bbox to be <class bool> but got {type(save_bbox)}.')
        sys.exit()
    #-- Check margin
    if not isinstance(pxcount, int):
        print(f'Expected --pxcount to be <class int> but got {type(pxcount)}.')
        sys.exit()
    #-- Check form
    if not isinstance(form, str):
        print(f'Expected --form to be <class str> but got {type(form)}.')
        sys.exit()
    
    #-- Get phase images and events
    for file in tqdm(files):
        # ChunkLoader = ImageChunkLoader(file, chunk_size, overlap)
        file_stem = str(file.stem).split('.')[0]
        
        temp_out_dir = out_dir or file.parent.parent / 'subdivided'
        temp_out_dir, temp_lake_dir, temp_nolake_dir = make_out_dirs(temp_out_dir)
        
        if save_bbox:
            txt_name = (temp_out_dir / file_stem).with_suffix('.txt')
            with open(txt_name, "w") as f:
                f.write('i scale y0 y1 x0 x1\n')
        
        for scale in downscale_factor:
            print(file)
            ChunkLoader = chunk_loader.ImageChunkLoader(file, chunk_size, overlap, scale, form)
            mask = load_mask((file.parent / file_stem).with_suffix('.png'), ChunkLoader.shape)
            
            for (Img, bbox) in zip(ChunkLoader, ChunkLoader.bboxes):
                
                #-- Save chunks
                if does_chunk_overlap_lake(bbox, mask, pxcount):
                    if not skip_class == CLASS_LAKE:
                        class_dir = temp_lake_dir
                        img_name = (class_dir / (file_stem + f'_s{scale}' + f'_{ChunkLoader._iter_idx}')).with_suffix('.png')
                        Img.save(img_name)
                else:
                    if not skip_class == CLASS_NO_LAKE:
                        class_dir = temp_nolake_dir
                        img_name = (class_dir / (file_stem + f'_s{scale}' + f'_{ChunkLoader._iter_idx}')).with_suffix('.png')
                        Img.save(img_name)
                
                #-- Save chunk overview
                if save_bbox:
                    with open(txt_name, "a") as f:
                        f.write(f"{ChunkLoader._iter_idx} {scale} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


if __name__ == "__main__":
    #-- FOR TESTING ONLY
    # if True:
    #     main(r"D:\dtu\speciale\ipp\processed\track_002_ascending\iff_dd\20171112_20171124_20171124_20171206.pha.png",#r"D:\dtu\speciale\ipp\processed\track_002_ascending\iff_dd\*.png", 
    #          None,
    #          None,
    #          224,
    #          False,
    #          [1],
    #          'recta',
    #          None,
    #          False,
    #          50,)
    
    args = parse_arguments()
    
    
    main(args.iff_dd_paths, args.out_dir, args.chunk_size, args.overlap, args.downscale_factor, args.form, args.skip_class, args.save_bbox, args.pxcount)