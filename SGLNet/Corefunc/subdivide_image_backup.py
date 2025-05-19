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
import argparse
import SGLNet.PyIPP.iff_dd_event as ipp_iff_dd_event


CLASS_LAKE = 'lake'
CLASS_NO_LAKE = 'nolake'


def is_Path(path : Union[Path, str]):
    if isinstance(path, str):
        path = Path(path)
    if not isinstance(path, Path):
        print(f'Unknown path type: {type(path)}')
        sys.exit()
    return path


def exist_dir(path : Path):
    if not path.is_dir():
        print(f'Invalid path: {str(path)}\nDirectory does not exist.')
        sys.exit()


def make_dir(path : Path):
    if not path.is_dir():
        path.mkdir()
        
        
def does_chunk_overlap(chunk_box, EventCoords):
    """
    Checks if the given chunk (x_start, y_start, x_end, y_end) overlaps any bounding box.
    """
    x_start, y_start, x_end, y_end = chunk_box

    for by_min, by_max, bx_min, bx_max in zip(EventCoords.li0, 
                                               EventCoords.li1, 
                                               EventCoords.sa0, 
                                               EventCoords.sa1):
        overlap_x = not (x_end <= bx_min or x_start >= bx_max)
        overlap_y = not (y_end <= by_min or y_start >= by_max)
        
        if overlap_x and overlap_y:
            return True

    return False
        

def split_image_strict(image, N):
    """Splits a PIL image into (N, N) chunks, discarding edge parts if they don't fit."""
    X, Y = image.size
    chunks = []
    boxes = []
    for i in range(0, X - X % N, N):
        for j in range(0, Y - Y % N, N):
            box = (i, j, i + N, j + N)
            chunks.append(image.crop(box))
            boxes.append(box)

    return chunks, boxes


def pad_and_split(image, N):
    """Pads the image to ensure it divides evenly, then splits into (N, N) chunks."""
    X, Y = image.size
    pad_x = (N - X % N) % N
    pad_y = (N - Y % N) % N

    # Pad image with black (or use ImageOps.expand to specify color)
    padded_image = ImageOps.expand(image, (0, 0, pad_x, pad_y), fill=0)
    return split_image_strict(padded_image, N)


def parse_arguments():
    """
    Import arguments parsed through the command line call.

    Returns
    -------
    ArgumentParser.parse_args()
        Class containing the parsed arguments.

    """
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument(
        "event_dirs",
        nargs='*',
        metavar="./track_<xxx>_<pass>/gim_events/<iff_dd_interval>_<#event>",
        help="Path/to/gim_event_dirs. Wildcards are possible")
    parser.add_argument(
        "-l", 
        "--lowpass", 
        type=int,
        default=100,
        metavar="<int>",
        help="Exclude segmentations with pixel areas less than filter value.")
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=None,
        metavar="Path/to/outdir",
        help="Path to output directory. It is created if it does not exist.")
    parser.add_argument(
        "-s",
        "--show_unwrap",
        type=bool,
        default = True,
        metavar='<bool>',
        help="Use unwrapped phase image for visualization during" \
            +"segmentation. If False, the wrapped phase is shown instead.")
    return parser.parse_args()


def main(iff_dir: str, out_dir: str, chunk_size: int, plot: bool):
    #-- Check iff directory
    iff_dir = is_Path(iff_dir)
    exist_dir(iff_dir)
    #-- Check out directory
    out_dir = is_Path(out_dir)
    make_dir(out_dir)
    exist_dir(out_dir)
    #-- Create lake subdirectory
    lake_dir = out_dir / CLASS_LAKE
    make_dir(lake_dir)
    exist_dir(lake_dir)
    #-- Create no-lake subdirectory
    nolake_dir = out_dir / CLASS_NO_LAKE
    make_dir(nolake_dir)
    exist_dir(nolake_dir)
    
    #-- Get phase images and events
    paths = [p for p in iff_dir.iterdir() if p.suffix=='.png']
    Obj_list = [ipp_iff_dd_event.IffDDEvent(path) for path in paths]
    
    #-- Loop through images
    for Obj in Obj_list:
        #-- Plot if specified
        if plot is True:
            Obj.plot()
        #-- Divide into chunks
        chunks, boxes = pad_and_split(Obj.png, chunk_size)
        #-- Save chunks
        for i, (chunk, box) in enumerate(zip(chunks, boxes)):
            if does_chunk_overlap(box, Obj.EventCoords):
                subdir = lake_dir
            else:
                subdir = nolake_dir
            name = (subdir / (Obj.basename + f'_{i}')).with_suffix('.png')
            chunk.save(name)
        #-- Save chunk overview
        with open((out_dir / Obj.basename).with_suffix('.txt'), "w") as f:
            for i, (y0, y1, x0, x1) in enumerate(boxes):
                f.write(f"{i} {y0} {y1} {x0} {x1}\n")

if __name__ == "__main__":
    IFF_DIR = r"D:\dtu\speciale\ipp\processed\track_002_ascending\iff_dd"
    OUT_DIR = r"D:\dtu\speciale\ipp\processed\track_002_ascending\subdivided"
    CHUNK_SIZE = 224
    PLOT = False
    out = main(IFF_DIR, OUT_DIR, CHUNK_SIZE, PLOT)