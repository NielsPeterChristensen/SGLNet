#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 06 2025

@author: niels
"""

# =============================================================================
# Imports
# =============================================================================


import os
import sys
import json
import pickle
import numpy as np
import numpy.typing as npt
import argparse
import skimage as ski
from PIL import Image
from pathlib import Path
#-- SGLNet functions
import SGLNet.Corefunc.utils as utils
import SGLNet.Corefunc.stats as stats


# =============================================================================
# Functions
# =============================================================================


def get_args_parser():
    """
    Import arguments parsed through the command line call.

    Returns
    -------
    ArgumentParser.parse_args()
        Class containing the parsed arguments.

    """
    parser = argparse.ArgumentParser('compute_segmentation_overlap', add_help=False)
    
    parser.add_argument(
        "-i",
        "--iff_eventTracking_dir",
        metavar = "/path/to/iff_eventTracking/ main directory",
        type = str,
        help = ("Path to the iff_eventTracking directory. Directory is located inside the"
                " track_[XYZ]_[DIRECTION] directory, and it contains subdirectories"
                " with names corresponding to iff_dd acquisitions. These subdirectories"
                " should each contain a files ending in _meta.json and (*) _binaryMask.png"
                " for this script to work. The _binaryMask.png images should follow"
                " the same pixel value convention as for ground truth images.\n"
                "(*) Note: see description for --segmentation_dir.")
        )
    parser.add_argument(
        "-gt",
        "--ground_truth_dir",
        metavar = "/path/to/ground_truth_dir",
        type = str,
        help = ("Path to the ground truth mask directory. Masks should have the same dimension"
                " as the original iff_dd .pha products (see the '_meta.json' file)"
                " to avoid potential errors. Masks are expected to be uint8 images"
                " where pixel value <127 means 'no lake' and value >=127 means 'lake'."
                " File names are expected to be identical to the iff_dd products,"
                " i.e. [DATE1]_[DATE2]_[DATE2]_[DATE3].png"
               )
        )
    parser.add_argument(
        "-o",
        "--output_dir",
        metavar = "/path/to/output_dir",
        default = None,
        type = str,
        help = ("Path to output directory to save segmentation metrics."
                " Default is None, meaning in the same subdir as _binartMask.png"
                " are stored.")
        )
    parser.add_argument(
        "-seg",
        "--segmentation_dir",
        metavar = "/path/to/segmentation_dir",
        type = str,
        default = None,
        help = ("Optional. This argument should only be used if _binaryMask.png"
                " files are not located inside subdirectories within the"
                " iff_eventTracking directory. Default is None, meaning "
                " _binaryMask.png files are expected within iff_eventTracking."
                " Note: This script still expected the _binaryMask.png files"
                " to be located within subdirectories inside segmentation_dir,"
                " following the same convention as subdirectories within"
                " iff_eventTracking.")
        )
    parser.add_argument(
        "-w",
        "--warnings",
        type = utils.bool_flag,
        default = True,
        metavar = "<bool>",
        help = ("Specify whether to show warnings. Default is True.")
        )
    
    return parser
    
def main(args: argparse.Namespace) -> None:
    
    print("\nRunning compute_segmentation_overlap.py", file=sys.stderr)
    
    #-- Set valid segmentation directory
    args.segmentation_dir = args.segmentation_dir or args.iff_eventTracking_dir
    
    #-- Make paths to Path objects
    args.iff_eventTracking_dir= Path(args.iff_eventTracking_dir).resolve()
    args.ground_truth_dir = Path(args.ground_truth_dir).resolve()
    args.segmentation_dir = Path(args.segmentation_dir).resolve()
    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir).resolve()
    
    #-- Check inputs
    if not args.iff_eventTracking_dir.is_dir():
        print(f"Invalid eventTracking directory {str(args.iff_eventTracking_dir)}", file=sys.stderr)
        sys.exit(1)
    if not args.ground_truth_dir.is_dir():
        print(f"Invalid ground truth directory {str(args.ground_truth_dir)}", file=sys.stderr)
        sys.exit(1)
    if not args.segmentation_dir.is_dir():
        print(f"Invalid segmentation directory {str(args.segmentation_dir)}", file=sys.stderr)
        sys.exit(1)
   
    #-- Check output
    if args.output_dir is not None and not args.output_dir.is_dir():
        print(f"Generating output directory {str(args.output_dir)}", file=sys.stderr)
        args.output_dir.mkdir(exist_ok=True)

    #-- Get derived files
    basenames = np.array([str(d.name) for d in args.iff_eventTracking_dir.iterdir() if d.is_dir()])
    iff_subdirs = np.array([args.iff_eventTracking_dir / name for name in basenames])
    seg_subdirs = np.array([args.segmentation_dir / name for name in basenames])
    gt_files = np.array([args.ground_truth_dir / f"{name}.png" for name in basenames])
    meta_files = np.array([subdir / f"{name}_meta.json" for (subdir, name) in zip(iff_subdirs, basenames)])
    mask_files = np.array([subdir / f"{name}_binaryMask.png" for (subdir, name) in zip(seg_subdirs, basenames)])
    
    #-- Check derived files
    gt_exists = np.array([f.is_file() for f in gt_files])
    if not all(gt_exists):
        print(f"Not all gt files were found. Missing are {gt_files[~gt_exists]}.", file=sys.stderr)
        sys.exit(1)
    meta_exists = np.array([f.is_file() for f in meta_files])
    if not all(meta_exists):
        print(f"Not all _meta.json files were found. Missing are {meta_files[~meta_exists]}.", file=sys.stderr)
        sys.exit(1)
    mask_exists = np.array([f.is_file() for f in mask_files])
    if not all(mask_exists):
        print(f"Not all _binaryMask.png files were found. Missing are {mask_files[~mask_exists]}.", file=sys.stderr)
        sys.exit(1)    
    
    #-- Loop over directories
    num_files = len(mask_files)
    for i, (mask_file, meta_file, gt_file) in enumerate(zip(mask_files, meta_files, gt_files)):
        evaluate(mask_file, meta_file, gt_file, args.output_dir, args.warnings)
        if (i+1)%5 == 0:
            print(f"{i+1} / {num_files} Completed")
    print("Done!", file=sys.stderr)
    
    
def evaluate(mask_file: os.PathLike, meta_file: os.PathLike, gt_file: os.PathLike, output_dir: os.PathLike, show_warnings: bool) -> None:
    
    #-- Load metadata
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    #-- Load segmentation mask
    seg_mask = Image.open(mask_file).convert('L')
    if list(seg_mask.size) != meta['imsize']:
        if show_warnings:
            print((f"Warning: size of ground truth mask ({seg_mask.size}) does not "
                   " fit size of iff_dd product ({meta['imsize']}) so the mask is "
                   " being resized. Be cautious as this might cause trouble."),
                  file = sys.stderr)
        seg_mask = seg_mask.resize(meta['imsize'], Image.LANCZOS)
    seg_mask = np.array(seg_mask) >= 127
    seg_mask = utils.pad_array(seg_mask, meta['imsize'][::-1], meta['chunk_size'], meta['overlap'])
    #-- Load ground truth mask
    gt_mask = Image.open(gt_file).convert('L')
    if list(gt_mask.size) != meta['imsize']:
        if show_warnings:
            print((f"Warning: size of ground truth mask ({gt_mask.size}) does not "
                   " fit size of iff_dd product ({meta['imsize']}) so the mask is "
                   " being resized. Be cautious as this might cause trouble."),
                  file = sys.stderr)
        gt_mask = gt_mask.resize(meta['imsize'], Image.LANCZOS)
    gt_mask = np.array(gt_mask) >= 127
    gt_mask = utils.pad_array(gt_mask, meta['imsize'][::-1], meta['chunk_size'], meta['overlap'])
    
    #-- Get metrics
    metrics = stats.SegmentationOverlap(seg_mask, gt_mask)
    
    #-- Store results
    output_dir = output_dir or mask_file.parent
    with open(output_dir / f"{mask_file.parent.name}_segStats.json", 'w') as f:
        json.dump(metrics(), f, indent=4)
            

if __name__ == '__main__':
    TESTING = False
    #-- Get input
    parser = argparse.ArgumentParser(
        'compute_segmentation_overlap', 
        description=("Compare segmentations produces by semantic_segmentation.py"
                     " and ground truth segmentations. Script calculates"
                     " Dice similarity coefficient and Jaccard index."),
        parents=[get_args_parser()])
    if TESTING:
        print("TESTING!", file=sys.stderr)
        args = parser.parse_args((
            "-i"
            r" D:\dtu\speciale\ipp\processed\track_010_ascending\iff_eventTracking"
            " -gt"
            r" D:\dtu\speciale\Data\testData\Cook\track_010_ascending\ground_truth"
            " -seg"
            r" D:\dtu\speciale\ipp\processed\track_010_ascending\segmentation\dino"
            " -w"
            " False"
        ).split())
    else: 
        args = parser.parse_args()
        
    main(args)
    
    