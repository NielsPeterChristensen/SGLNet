#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:31:45 2025

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
import argparse
from PIL import Image
from pathlib import Path
#-- SGLNet functions
import SGLNet.Corefunc.utils as utils
import SGLNet.Corefunc.chunk_masking as chunk_masking
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
    parser = argparse.ArgumentParser('compute_confusion_matrix', add_help=False)
    
    parser.add_argument(
        "-i",
        "--iff_eventTracking_dir",
        metavar = "/path/to/iff_eventTracking/ main directory",
        type = str,
        help = ("Path to the iff_eventTracking directory. Directory is located inside the"
                " track_[XYZ]_[DIRECTION] directory, and it contains subdirectories"
                " with names corresponding to iff_dd acquisitions. These subdirectories"
                " should contain files with names ending in '_bbox.pkl', 'pred.pkl'"
                " and '_meta.json' for this script to work. '_embed.pkl' is currently"
                " only used for segmentation."
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
        "-gl",
        "--grounding_line_mask",
        metavar = "/path/to/grounding_line_mask",
        default = None,
        type = str,
        help = ("Optional. Path to grounding line mask file. Use only if bboxes have"
                " not already filtered to exclude grounding lines."
                " Masks are expected to be as described for --ground_truth_dir."
                " Default is None.")
        )
    parser.add_argument(
        "-tr",
        "--training_dir",
        metavar = "/path/to/training_segmentation_dir",
        default = None,
        type = str,
        help = ("Optional. Path to the training segmentation dir. Mask are expected"
                " to be as described for --ground_truth_dir, but with pixel values"
                " x<85 for 'no lake', values 85<=x<170 for 'exclude' and values"
                " x>=170 for 'lake'. Default is None, meaning these metrics are not"
                " included in the output.")
                )
    parser.add_argument(
        "-o",
        "--output_dir",
        metavar = "/path/to/output_dir",
        default = None,
        type = str,
        help = ("Path to output directory to save performance metrics."
                " Default is None, meaning in the same subdir as _pred.pkl and"
                " _meta.json etc. are stored.")
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
    
    print("\nRunning compute_confusion_matrix.py", file=sys.stderr)
    
    #-- Make paths to Path objects
    args.iff_eventTracking_dir = Path(args.iff_eventTracking_dir).resolve()
    args.ground_truth_dir = Path(args.ground_truth_dir).resolve()
    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir).resolve()
    if args.grounding_line_mask is not None:
        args.grounding_line_mask = Path(args.grounding_line_mask).resolve()
    if args.training_dir is not None:
        args.training_dir = Path(args.training_dir).resolve()
    
    #-- Check inputs
    if not args.iff_eventTracking_dir.is_dir():
        print(f"Invalid eventTracking directory {str(args.iff_eventTracking_dir)}", file=sys.stderr)
        sys.exit(1)
    if not args.ground_truth_dir.is_dir():
        print(f"Invalid ground truth directory {str(args.ground_truth_dir)}", file=sys.stderr)
        sys.exit(1)
    if args.grounding_line_mask is not None and not args.grounding_line_mask.is_file():
        print(f"Invalid grounding line mask file {str(args.grounding_line_mask)}", file=sys.stderr)
        sys.exit(1)
    if args.training_dir is not None and not args.training_dir.is_dir():
        print(f"Invalid training segmentation directory {str(args.grounding_line_mask)}", file=sys.stderr)
        sys.exit(1)
   
    #-- Check output
    if args.output_dir is not None and not args.output_dir.is_dir():
        print(f"Generating output directory {str(args.output_dir)}", file=sys.stderr)
        args.output_dir.mkdir(exist_ok=True)
        
    #-- Get derived files
    basenames = np.array([str(d.name) for d in args.iff_eventTracking_dir.iterdir() if d.is_dir()])
    iff_subdirs = np.array([args.iff_eventTracking_dir / name for name in basenames])
    gt_files = np.array([args.ground_truth_dir / f"{name}.png" for name in basenames])
    bbox_files = np.array([subdir / f"{name}_bbox.pkl" for (subdir, name) in zip(iff_subdirs, basenames)])
    meta_files = np.array([subdir / f"{name}_meta.json" for (subdir, name) in zip(iff_subdirs, basenames)])
    pred_files = np.array([subdir / f"{name}_pred.pkl" for (subdir, name) in zip(iff_subdirs, basenames)])
    if args.training_dir is not None:
        # tr_files = np.array([list(args.training_dir.glob(f"{str(name)}*.png"))[0] for name in basenames])
        tr_files = np.array([next((f for f in args.training_dir.glob("*.png") if str(name) in f.name), None) for name in basenames], dtype=object)
        if len(tr_files) != len(gt_files) or any(f is None for f in tr_files):
            print("Unable to find all training segmentation files", file=sys.stderr)
            sys.exit(1)
    else:
        tr_files = np.zeros_like(gt_files)
        tr_files[:] = None
    
    #-- Check derived files
    gt_exists = np.array([f.is_file() for f in gt_files])
    if not all(gt_exists):
        print(f"Not all gt files were found. Missing are {gt_files[~gt_exists]}.", file=sys.stderr)
        sys.exit(1)
    bbox_exists = np.array([f.is_file() for f in bbox_files])
    if not all(bbox_exists):
        print(f"Not all _bbox.pkl files were found. Missing are {bbox_files[~bbox_exists]}.", file=sys.stderr)
        sys.exit(1)
    meta_exists = np.array([f.is_file() for f in meta_files])
    if not all(meta_exists):
        print(f"Not all _meta.json files were found. Missing are {meta_files[~meta_exists]}.", file=sys.stderr)
        sys.exit(1)
    pred_exists = np.array([f.is_file() for f in pred_files])
    if not all(pred_exists):
        print(f"Not all _pred.pkl files were found. Missing are {pred_files[~pred_exists]}.", file=sys.stderr)
        sys.exit(1)
    
    #-- Loop over directories
    num_files = len(pred_files)
    for i, (bbox_file, meta_file, pred_file, gt_file, tr_file) in enumerate(zip(bbox_files, meta_files, pred_files, gt_files, tr_files)):
        evaluate(bbox_file, meta_file, pred_file, gt_file, tr_file, args.grounding_line_mask, args.output_dir, args.warnings)
        if (i+1)%5 == 0:
            print(f"{i+1} / {num_files} Completed")
    print("Done!", file=sys.stderr)
        
def evaluate(bbox_file: os.PathLike, meta_file: os.PathLike, pred_file: os.PathLike, gt_file: os.PathLike, tr_file: os.PathLike, gl_file: os.PathLike, output_dir: os.PathLike, show_warnings: bool) -> None:
    
    #-- Load data
    with open(bbox_file, 'rb') as f:
        bboxes = pickle.load(f)
    with open(pred_file, 'rb') as f:
        preds = pickle.load(f)
    with open(meta_file, 'r') as f:
        meta = json.load(f)
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
    if tr_file is not None:
        tr_mask = Image.open(tr_file).convert('L')
        if list(tr_mask.size) != meta['imsize']:
            if show_warnings:
                print((f"Warning: size of training segmentation mask {tr_mask.size} does not"
                       f" fit size of iff_dd product {tuple(meta['imsize'])} so the mask is"
                       " being resized. Be cautious as this might cause trouble."),
                      file = sys.stderr)
            tr_mask = tr_mask.resize(meta['imsize'], Image.LANCZOS)
        tr_mask = np.array(tr_mask, dtype=np.int32)
        tr_mask = utils.pad_array(tr_mask, meta['imsize'][::-1], meta['chunk_size'], meta['overlap'])
        positive_mask = tr_mask >= 170
        ambiguous_mask = (tr_mask >= 85) & (tr_mask <170)
    if gl_file is None:
        gl_mask = np.zeros_like(gt_mask, dtype=bool)
    else:
        gl_mask = Image.open(gl_file).convert('L')
        if list(gl_mask.size) != meta['imsize']:
            print((f"Warning: size of grounding line mask ({gl_mask.size}) does not "
                   " fit size of iff_dd image ({meta['imsize']}) so the mask is "
                   " being resized. Be cautious as this might cause trouble."),
                  file = sys.stderr)
            gl_mask = gl_mask.resize(meta['imsize'], Image.LANCZOS)
        gl_mask = np.array(gl_mask) >= 127
        gl_mask = utils.pad_array(gl_mask, meta['imsize'][::-1], meta['chunk_size'], meta['overlap'])

    #-- Check if chunk bbox overlap any part of the grounding line mask and remove
    gl_overlap_chunk = chunk_masking.does_mask_overlap_chunk(bboxes, gl_mask)
    bboxes = bboxes[~gl_overlap_chunk]
    preds = preds[~gl_overlap_chunk]
    #-- Check if chunk bbox overlap any part of the ground truth mask
    gt_overlap_chunk = chunk_masking.does_mask_overlap_chunk(bboxes, gt_mask)
    #-- Check if chunk bbox overlap center 50% part of the ground truth mask
    gt_overlap_chunk_center = chunk_masking.does_mask_overlap_chunk_center(bboxes, gt_mask)
    
    #-- Get metrics
    conf_matx = stats.ConfusionMatrix(preds, gt_overlap_chunk_center)
    fp_near_mask = conf_matx.fp_arr & gt_overlap_chunk
    output_dict = {'test': {**conf_matx(), 'fp_near_mask': int(sum(fp_near_mask))}}
    
    #-- Compute training score if available
    if tr_file is not None:
        lake_in_chunk = chunk_masking.does_mask_overlap_chunk(bboxes, positive_mask)[~gl_overlap_chunk]
        lake_in_chunk_center = chunk_masking.does_mask_overlap_chunk_center(bboxes, positive_mask)[~gl_overlap_chunk]
        amb_in_chunk = chunk_masking.does_mask_overlap_chunk(bboxes, ambiguous_mask)[~gl_overlap_chunk]
        
        #-- (-1: exclude   0: nolake   1: lake)
        ind = np.zeros_like(lake_in_chunk_center, dtype=int)
        ind[amb_in_chunk] = -1
        ind[lake_in_chunk] = -1
        ind[lake_in_chunk_center] = 1
        
        valid_chunks = (ind > -1)
        lake_chunks = ind == 1
    
        tr_conf_matx  = stats.ConfusionMatrix(preds[valid_chunks], lake_chunks[valid_chunks])
        output_dict = {**output_dict, 'train': tr_conf_matx()}
    
    #-- Store results
    output_dir = output_dir or pred_file.parent
    with open(output_dir / f"{pred_file.parent.name}_confMatx.json", 'w') as f:
        json.dump(output_dict, f, indent=4)

if __name__ == '__main__':
    TESTING = False
    #-- Get input
    parser = argparse.ArgumentParser(
        'compute_confusion_matrix', 
        description=("Evaluate predictions against ground truths to get false positive (fp),"
                     " false negative (fn), true positive (tp) and true negative (tn)."
                     " These are computed by comparing to ground truths where the"
                     " central 50% of chunks contain lake features. Output also"
                     " includes a fp_near_mask which contains the number of fp"
                     " that are likely due to the true/false threshold of 50% overlap."),
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
            r" D:\dtu\speciale\ipp\processed\track_010_ascending\iff_eventTracking"
            " -gt"
            r" D:\dtu\speciale\Data\testData\Cook\track_010_ascending\ground_truth"
            " -tr"
            r" D:\dtu\speciale\Data\testData\Cook\track_010_ascending\segmentation_v3"
            " -w"
            " False"
        ).split())
    else: 
        args = parser.parse_args()
        
    main(args)
    
    