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
import warnings
import numpy as np
import numpy.typing as npt
import argparse
import skimage as ski
from PIL import Image
from pathlib import Path
#-- SGLNet functions
import SGLNet.Corefunc.utils as utils


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
    parser = argparse.ArgumentParser('compute_lake_count', add_help=False)
    
    parser.add_argument(
        "-i",
        "--iff_eventCoords_dir",
        metavar = "/path/to/iff_eventCoords/ directory",
        type = str,
        help = ("Path to the iff_eventCoords directory. Directory is located inside"
                " the track_[XYZ]_[DIRECTION] directory, and it contains txt-files"
                " with names corresponding to iff_dd acquisitions. These txt-files"
                " contain output boundary boxes for detected events."
               )
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
        type = str,
        help = ("Path to output directory to save lake count files. These files"
                " are stored in subdirectories named after the iff_dd product."
                " This output is complient with the structure of the"
                " iff_eventTracking directory produced by event_tracking_iff_DINO.py"
                " and it is recommended to give the path to the iff_eventTracking"
                " dir. Note: If the path given does not exist, it is created.")
        )
    parser.add_argument(
        "-s",
        "--save_lake_mask",
        type = utils.bool_flag,
        default = False,
        metavar = "<bool>",
        help = ("Specify whether to save additional _lakeMask.pkl file which"
                " contains labelled gt lakes and some other informations."
                " This is currently not used for anything. Default is False.")
        )
    
    return parser
    
def main(args: argparse.Namespace) -> None:
    
    print("\nRunning compute_lake_count.py", file=sys.stderr)
    
    #-- Make paths to Path objects
    args.iff_eventCoords_dir = Path(args.iff_eventCoords_dir).resolve()
    args.ground_truth_dir = Path(args.ground_truth_dir).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    
    #-- Check inputs
    if not args.iff_eventCoords_dir.is_dir():
        print(f"Invalid eventCoords directory {str(args.iff_eventCoords_dir)}", file=sys.stderr)
        sys.exit(1)
    if not args.ground_truth_dir.is_dir():
        print(f"Invalid ground truth directory {str(args.ground_truth_dir)}", file=sys.stderr)
        sys.exit(1)
   
    #-- Check output
    if not args.output_dir.is_dir():
        print(f"Generating output directory {str(args.output_dir)}", file=sys.stderr)
        args.output_dir.mkdir(exist_ok=True)
        
    #-- Get derived files
    ec_files = np.array(list(args.iff_eventCoords_dir.glob('*.txt')))
    basenames = np.array([str(f.stem) for f in ec_files if f.is_file()])
    gt_files = np.array([args.ground_truth_dir / f"{name}.png" for name in basenames])
    
    #-- Check derived files
    gt_exists = np.array([f.is_file() for f in gt_files])
    if not all(gt_exists):
        print(f"Not all gt files were found. Missing are {gt_files[~gt_exists]}.", file=sys.stderr)
        sys.exit(1)
    
    #-- Loop over directories
    num_files = len(ec_files)
    for i, (ec_file, gt_file, bname) in enumerate(zip(ec_files, gt_files, basenames)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            bboxes = np.loadtxt(ec_file, skiprows=1, dtype=int).reshape((-1, 4))
        bboxes[:, [0, 1]] = bboxes[:, [1, 0]]
        bboxes[:, [2, 3]] = bboxes[:, [3, 2]]
        count_dict, lake_dict = compare_prediction_and_groundtruth(bboxes, gt_file, threshold=1.0)
        #-- Store output
        odir = args.output_dir / bname
        odir.mkdir(exist_ok=True)
        with open(odir / f"{bname}_lakeCount.json", "w") as f:
            json.dump(count_dict, f, indent=4)
        if args.save_lake_mask:
            with open(odir / f"{bname}_lakeMask.pkl", "wb") as f:
                pickle.dump(lake_dict, f)
        if (i+1)%5 == 0:
            print(f"{i+1} / {num_files} Completed")
    print("Done!", file=sys.stderr)
    
        
def get_groundtruths(fname: os.PathLike) -> dict:
    image = Image.open(fname).convert('L')
    mask = np.array(image).astype(bool)
    label_mask = ski.measure.label(mask)
    props = ski.measure.regionprops(label_mask)
    
    num_lakes = len(props)
    shape = mask.shape[::-1] # Store in format of PIL Image (Width, Height)
    centers = []
    labels = []
    pxcount_groundtruth = []
    for region in props:
        y, x = region.centroid
        centers.append((int(x), int(y)))
        label = region.label
        labels.append(label)
        pxcount_groundtruth.append(np.sum(label_mask == label))

    return {
        'num_lakes': num_lakes,
        'shape': shape,
        'labels': labels,
        'pxcount_groundtruth': pxcount_groundtruth,
        'centers': centers,
        'label_mask': label_mask.astype(np.uint8)
    }

def compare_prediction_and_groundtruth(bboxes: npt.ArrayLike, gt_file: os.PathLike, threshold: float = 0.6) -> dict:
    #-- handle faulty threshold input
    if threshold <= 0.0 or threshold > 1.0:
        print(f"threshold must fulfill condition: 0 < threshold <= 1, but received {threshold}.", file=sys.stderr)
        sys.exit(1)
    
    propdict = get_groundtruths(gt_file)
    prediction_mask = utils.bboxes2mask(bboxes, propdict['shape'])
    
    captured = propdict['label_mask'][prediction_mask]
    pxcount_captured = [np.sum(captured == label) for label in propdict['labels']]
    propdict['pxcount_captured'] = pxcount_captured
    
    is_captured = [px_cap >= threshold*px_gt for (px_cap, px_gt) in zip(propdict['pxcount_captured'], propdict['pxcount_groundtruth'])]
    num_found = sum(is_captured)
    propdict['is_captured'] = is_captured
    propdict['num_found'] = num_found
    
    empty_bboxes = [not np.any(propdict['label_mask'][y_min:y_max, x_min:x_max]) for x_min, y_min, x_max, y_max in bboxes]
            
    count_dict = {
        'num_bboxes': int(len(bboxes)),
        'num_empty': int(sum(empty_bboxes)),
        'num_found': int(propdict['num_found']),
        'num_lakes': int(propdict['num_lakes'])
    }
    lake_dict = {
        'bboxes': bboxes,
        'empty_bboxes': empty_bboxes,
        'centers': propdict['centers'],
        'is_captured': propdict['is_captured'],
        'label_mask': propdict['label_mask'],
        'labels': propdict['labels'],
        'pxcount_captured': propdict['pxcount_captured'],
        'pxcount_groundtruth': propdict['pxcount_groundtruth']
    }
    
    return count_dict, lake_dict

if __name__ == '__main__':
    TESTING = False
    #-- Get input
    parser = argparse.ArgumentParser(
        'compute_lake_count', 
        description=("Evaluate number of ground truth lakes captured by eventCoords."
                     " Lakes are only counted as being detected if they are fully"
                     " included in the eventCoords boundary boxes, as these"
                     " boxes already include an additional padding."),
        parents=[get_args_parser()])
    if TESTING:
        print("TESTING!", file=sys.stderr)
        args = parser.parse_args((
            # "-i"
            # r" D:\dtu\speciale\ipp\processed\track_065_descending\iff_eventTracking"
            # " -gt"
            # r" D:\dtu\speciale\Data\testData\PineIsland\track_065_descending\ground_truth"
            # " -tr"
            # r" D:\dtu\speciale\Data\testData\PineIsland\track_065_descending\segmentation_v3"
            "-i"
            r" D:\dtu\speciale\ipp\processed\track_010_ascending\iff_eventCoords"
            " -gt"
            r" D:\dtu\speciale\Data\testData\Cook\track_010_ascending\ground_truth"
            " -o"
            r" D:\dtu\speciale\ipp\processed\track_010_ascending\iff_eventTracking"
            " -w"
            " False"
        ).split())
    else: 
        args = parser.parse_args()
        
    main(args)
    
    