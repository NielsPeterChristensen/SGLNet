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
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import SGLNet.Corefunc.utils as utils
import SGLNet.Corefunc.chunk_loader as chunk_loader
import SGLNet.plotting.plot_utils as plot_utils


CLASS_LAKE = 'lake'
CLASS_NO_LAKE = 'nolake'


def get_args_parser():
    """
    Import arguments parsed through the command line call.

    Returns
    -------
    ArgumentParser.parse_args()
        Class containing the parsed arguments.

    """
    parser = argparse.ArgumentParser('bbox2img', add_help=False)
    
    parser.add_argument(
        "iff_dd_path",
        metavar = "./track_<xxx>_<pass>/iff_dd/<iff_dd_interval>.pha",
        help = "Path/to/iff_dd_image")
    parser.add_argument(
        "-o",
        "--output_dir",
        type = str,
        required = True,
        metavar = "Path/to/output_dir",
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
        type = int,
        default = 1,
        metavar = '<class int>',
        help = "Downscaling factor as integer. Default is 1.")
    parser.add_argument(
        "--form",
        type = str,
        default = 'phase',
        metavar = '<class str>',
        help = "Data form. Options are ['image', 'phase', 'recta', 'polar']. Default is 'phase'.")
    parser.add_argument(
        '--save_3x3',
        type = utils.bool_flag,
        default = False,
        metavar = '<class bool>',
        help = "Whether to also divide the chunk into smaller 3x3 'pseudo-patches' for visualisation purposes. Default is False.")
    
    return parser


def get_clicked_bbox(bboxes, x, y):
    conditions = (bboxes[:, 0] <= x) & (x <= bboxes[:, 2]) & (bboxes[:, 1] <= y) & (y <= bboxes[:, 3])
    bbox_indices = np.where(conditions)[0]
    
    return bboxes[bbox_indices]
    

def divide_and_save_image(Img: Image, output_dir: Path):
    width, height = Img.size
    square_width = width // 3
    square_height = height // 3
    for i in range(3):
        for j in range(3):
            left = j * square_width
            upper = i * square_height
            right = (j + 1) * square_width
            lower = (i + 1) * square_height
            cropped_img = Img.crop((left, upper, right, lower))
            cropped_img.save(output_dir / f"patch_{i}_{j}.png")


def main(args: argparse.Namespace):
    
    #-- Check files input
    file = Path(args.iff_dd_path).resolve()
    if not file.exists():
        print(f'Found no file at {file}.', file=sys.stderr)
        print(1)
    #-- Check output_dir input
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    #-- Check chunk_size
    if not isinstance(args.chunk_size, int):
        print(f'Expected --chunk_size to be <class int> but got {type(args.chunk_size)}.', file=sys.stderr)
        sys.exit(1)
    #-- Check overlap
    if not isinstance(args.overlap, bool):
        print(f'Expected --overlap to be <class bool> but got {type(args.overlap)}.', file=sys.stderr)
        sys.exit(1)
    #-- Check downscale_factor
    if not isinstance(args.downscale_factor, int):
        print(f'Expected --downscale_factor to be <class int> but got {type(args.downscale_factor)}.', file=sys.stderr)
        sys.exit(1)
    #-- Check form
    if not isinstance(args.form, str):
        print(f'Expected --form to be <class str> but got {type(args.form)}.', file=sys.stderr)
        sys.exit(1)
    
    ChunkLoader = chunk_loader.ImageChunkLoader(file, args.chunk_size, args.overlap, args.downscale_factor, args.form)
    Img = ChunkLoader().copy()
    plot_utils.draw_bbox(Img, ChunkLoader.bboxes)
    img = np.array(Img)
    
    # Create the figure and axes
    fig, ax = plt.subplots()
    ax.imshow(img)  # Display the image
    ax.set_title("Click on the Image")
    
    # Function to handle clicks
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:  # Ensure it's within the image bounds
            print(f"Clicked at: ({int(event.xdata)}, {int(event.ydata)})")
            bboxes = get_clicked_bbox(ChunkLoader.bboxes, event.xdata, event.ydata)
            for bbox in bboxes:
                Crop = Img.crop(bbox)
                Crop.save(output_dir / f"bbox{bbox}.png")
                if args.save_3x3:
                    save_dir = output_dir / 'patches' / f"bbox{bbox}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    divide_and_save_image(Crop, save_dir)
    
    # Connect the click event
    fig.canvas.mpl_connect("button_press_event", on_click)
    
    # Show the image
    plt.show()


if __name__ == "__main__":
    
    TESTING = True
    parser = argparse.ArgumentParser('bbox2img', description="Shows bboxes and allows user to select boxes to be cropped out and saved individually", parents=[get_args_parser()])
        
    #-- Get input
    if TESTING:
        print("TESTING!")
        args = parser.parse_args(r"D:\dtu\speciale\ipp\processed\track_002_ascending\iff_dd\20200101_20200113_20200113_20200125.pha -o D:\dtu\speciale\Vejledning\Vejledning_6\Chunks --chunk_size 448 --overlap False --save_3x3 True".split())
    else:
        args = parser.parse_args()
        
    main(args)