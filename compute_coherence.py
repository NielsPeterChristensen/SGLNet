#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 8 2025

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
import warnings
from PIL import Image
from pathlib import Path
#-- SGLNet functions
import SGLNet.Corefunc.utils as utils
import SGLNet.Corefunc.chunk_masking as chunk_masking
import SGLNet.Corefunc.stats as stats
import SGLNet.PyIPP.ipp_io as io


# =============================================================================
# Macros
# =============================================================================


dtype = [
    ('invalid', bool),
    ('tp', bool),
    ('tn', bool),
    ('fp', bool),
    ('fn', bool),
    ('fp_near_mask', bool),
    ('cor1', np.float32),
    ('cor2', np.float32),
    ('mult_cor', np.float32),
    ('mean_cor', np.float32)
]


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
    parser = argparse.ArgumentParser('compute_coherence', add_help=False)
    
    parser.add_argument(
        "-i",
        "--iff_eventTracking_dir",
        metavar = "/path/to/iff_eventTracking/ main directory",
        type = str,
        help = ("Path to the iff_eventTracking directory. Directory is located inside the"
                " track_[XYZ]_[DIRECTION] directory, and it contains subdirectories"
                " with names corresponding to iff_dd acquisitions. These subdirectories"
                " should contain files with names ending in '_bbox.pkl', 'pred.pkl'"
                " and '_meta.json' for this script to work.")
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
                " i.e. [DATE1]_[DATE2]_[DATE2]_[DATE3].png")
        )
    parser.add_argument(
        "-cor",
        "--iff_dir",
        metavar = "/path/to/iff_dir",
        type = str,
        help = ("Path to iff directory. This directory should have subdirectories",
                " named according to the relevant acquistions, e.g."
                " 20170722_100711_20170722_100805_HH_20170728_100630_20170728_100724_HH,"
                " each containing a .cor file with the same name as the subdirectory"
                " it resides in.")
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
    
    print("\nRunning compute_coherence.py", file=sys.stderr)
    
    #-- Make paths to Path objects
    args.iff_eventTracking_dir = Path(args.iff_eventTracking_dir).resolve()
    args.ground_truth_dir = Path(args.ground_truth_dir).resolve()
    args.iff_dir = Path(args.iff_dir).resolve()
    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir).resolve()
    if args.grounding_line_mask is not None:
        args.grounding_line_mask = Path(args.grounding_line_mask).resolve()
    
    #-- Check inputs
    if not args.iff_eventTracking_dir.is_dir():
        print(f"Invalid eventTracking directory {str(args.iff_eventTracking_dir)}", file=sys.stderr)
        sys.exit(1)
    if not args.ground_truth_dir.is_dir():
        print(f"Invalid ground truth directory {str(args.ground_truth_dir)}", file=sys.stderr)
        sys.exit(1)
    if not args.iff_dir.is_dir():
        print(f"Invalid iff directory {str(args.iff_dir)}", file=sys.stderr)
        sys.exit(1)
    if args.grounding_line_mask is not None and not args.grounding_line_mask.is_file():
        print(f"Invalid grounding line mask file {str(args.grounding_line_mask)}", file=sys.stderr)
        sys.exit(1)
   
    #-- Check output
    if args.output_dir is not None and not args.output_dir.is_dir():
        print(f"Generating output directory {str(args.output_dir)}", file=sys.stderr)
        args.output_dir.mkdir(exist_ok=True)
        
    #-- Get derived files
    basenames = np.array([str(d.name) for d in args.iff_eventTracking_dir.iterdir() if d.is_dir()])
    subdirs = np.array([args.iff_eventTracking_dir / name for name in basenames])
    gt_files = np.array([args.ground_truth_dir / f"{name}.png" for name in basenames])
    bbox_files = np.array([subdir / f"{name}_bbox.pkl" for (subdir, name) in zip(subdirs, basenames)])
    meta_files = np.array([subdir / f"{name}_meta.json" for (subdir, name) in zip(subdirs, basenames)])
    pred_files = np.array([subdir / f"{name}_pred.pkl" for (subdir, name) in zip(subdirs, basenames)])
    cor_pairs = np.array([[[d1, d2], [d3, d4]] for names in basenames for d1, d2, d3, d4 in [names.split('_')]])
    cor_first_files = []
    for m1, m2 in cor_pairs[:,0,:]:
        matching = [
            file for file in args.iff_dir.rglob("*.cor")
            if m1 in file.name and m2 in file.name and "_HH_" in file.name
        ]
        cor_first_files.extend(matching)
    cor_second_files = []
    for m1, m2 in cor_pairs[:,1,:]:
        matching = [
            file for file in args.iff_dir.rglob("*.cor")
            if m1 in file.name and m2 in file.name and "_HH_" in file.name
        ]
        cor_second_files.extend(matching)
    cor_files = np.array([(f1, f2) for f1, f2 in zip(cor_first_files, cor_second_files)])

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
    cor_exists = np.array([f.is_file() for f in cor_files.reshape((-1,))])
    if not all(cor_exists):
        print(f"Not all .cor files were found. Missing are {cor_files.reshape((-1,))[~cor_exists.reshape((-1,))]}.", file=sys.stderr)
        sys.exit(1)
    
    #-- Loop over directories
    num_files = len(pred_files)
    for i, (bbox_file, meta_file, pred_file, gt_file, cor_pair) in enumerate(zip(bbox_files, meta_files, pred_files, gt_files, cor_files)):
        evaluate(bbox_file, meta_file, pred_file, gt_file, cor_pair, args.grounding_line_mask, args.output_dir, args.warnings)
        if (i+1)%5 == 0:
            print(f"{i+1} / {num_files} Completed")
    print("Done!", file=sys.stderr, file=sys.stderr)
        
def evaluate(bbox_file: os.PathLike, meta_file: os.PathLike, pred_file: os.PathLike, gt_file: os.PathLike, cor_pair: tuple[os.PathLike], gl_file: os.PathLike, output_dir: os.PathLike, show_warnings: bool) -> None:
    
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
    if gl_file is None:
        gl_mask = np.zeros_like(gt_mask, dtype=bool)
    else:
        gl_mask = Image.open(gl_file).convert('L')
        if list(gl_mask.size) != meta['imsize']:
            if show_warnings:
                print((f"Warning: size of grounding line mask ({gl_mask.size}) does not "
                       " fit size of iff_dd image ({meta['imsize']}) so the mask is "
                       " being resized. Be cautious as this might cause trouble."),
                      file = sys.stderr)
            gl_mask = gl_mask.resize(meta['imsize'], Image.LANCZOS)
        gl_mask = np.array(gl_mask) >= 127
        gl_mask = utils.pad_array(gl_mask, meta['imsize'][::-1], meta['chunk_size'], meta['overlap'])
    cor1_mask = io.read_big_endian_float32(cor_pair[0], meta['imsize'][1], meta['imsize'][0])
    cor2_mask = io.read_big_endian_float32(cor_pair[1], meta['imsize'][1], meta['imsize'][0])
    cor1_mask[cor1_mask == 9.99999968266e-21] = np.nan
    cor2_mask[cor2_mask == 9.99999968266e-21] = np.nan
    cor1_mask = utils.pad_array(cor1_mask, meta['imsize'][::-1], meta['chunk_size'], meta['overlap'], np.nan)
    cor2_mask = utils.pad_array(cor2_mask, meta['imsize'][::-1], meta['chunk_size'], meta['overlap'], np.nan)
    
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
    
    cor1 = np.array([cor1_mask[y1:y2, x1:x2] for (x1, y1, x2, y2) in bboxes])
    cor2 = np.array([cor2_mask[y1:y2, x1:x2] for (x1, y1, x2, y2) in bboxes])
    
    output = np.zeros(len(fp_near_mask), dtype=dtype)
    output['invalid'] = preds == -1
    output['tp'] = conf_matx.tp_arr
    output['tn'] = conf_matx.tn_arr
    output['fp'] = conf_matx.fp_arr
    output['fn'] = conf_matx.fn_arr
    output['fp_near_mask'] = fp_near_mask
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        output['cor1'] = np.nanmean(cor1, axis=(1,2))
        output['cor2'] = np.nanmean(cor2, axis=(1,2))
        output['mult_cor'] = np.nanmean(cor1*cor2, axis=(1,2))
        output['mean_cor'] = np.nanmean((cor1+cor2)/2, axis=(1,2))
    
    #-- Store results
    output_dir = output_dir or pred_file.parent
    with open(output_dir / f"{pred_file.parent.name}_corStats.pkl", 'wb') as f:
        pickle.dump(output, f)

if __name__ == '__main__':
    TESTING = False
    #-- Get input
    parser = argparse.ArgumentParser(
        'compute_coherence', 
        description=("Evaluate predictions against ground truths  where the"
                     " central 50% of chunks contain lake features. Outputs"
                     " coherence of each chunk and whether prediction was correct"
                     " or not."),
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
            " -cor"
            r" D:\dtu\speciale\ipp\processed\track_010_ascending\coherence"
            " -w"
            " False"
        ).split())
    else: 
        args = parser.parse_args()
        
    main(args)
    
    