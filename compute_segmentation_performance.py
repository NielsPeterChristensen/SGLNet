#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 7 2025

@author: niels
"""

# =============================================================================
# Imports
# =============================================================================


import sys
import json
import numpy as np
import argparse
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
    parser = argparse.ArgumentParser('compute_classification_performance', add_help=False)
    
    parser.add_argument(
        "segmentation_dirs",
        nargs = "+",
        action = "extend",
        metavar = "/path/to/segmentation/ main directory",
        type = str,
        help = ("If compute_segmentation_overlap.py was run with default setting,"
                " then the 'segmentation_parent_dir' would be iff_eventTracking"
                " directories. If a non-default output directory was used, then"
                " that directory should be given instead. In all cases, the"
                " segmentation dir(s) must contains subdirectories with names"
                " corresponding to iff_dd acquisitions. These subdirectories"
                " should each contain a file ending in '_segStas.json' for this"
                " script to work. Performance metrics are computes from all"
                " statistics contained in _confMatx.json files across the given"
                " iff_eventTracking directories."
               )
        )
    parser.add_argument(
        "-o",
        "--output_dir",
        metavar = "/path/to/output_dir",
        type = str,
        help = ("Path to output directory to save performance metrics."
                " It is created if it does not exist.")
        )
    
    return parser

def main(args: argparse.Namespace) -> None:
    
    print("\nRunning compute_segmentation_performance.py", file=sys.stderr)
    
    #-- Make paths to Path objects
    args.segmentation_dirs = np.array([Path(d).resolve() for d in args.segmentation_dirs])
    args.output_dir = Path(args.output_dir).resolve()
    
    #-- Check inputs
    input_is_dir = np.array([d.is_dir() for d in args.segmentation_dirs])
    if not all(input_is_dir):
        print(f"Invalid segmentation directory {args.segmentation_dirs[~input_is_dir]}", file=sys.stderr)
        sys.exit(1)
    #-- Check output
    if not args.output_dir.is_dir():
        print(f"Generating output directory {str(args.output_dir)}", file=sys.stderr)
        args.output_dir.mkdir(exist_ok=True, parents=True)
        
    #-- Get derived files
    subdirs = np.array([p for d in args.segmentation_dirs for p in d.iterdir() if p.is_dir()])
    segStats_files = np.array([d / f"{d.name}_segStats.json" for d in subdirs])
    
    #-- Check derived files
    segStats_exists = np.array([f.is_file() for f in segStats_files])
    if not all(segStats_exists):
        print(f"Not all _segStats.json files were found. Missing are {segStats_files[~segStats_exists]}.", file=sys.stderr)
        sys.exit(1)
        
    #-- Loop over directories
    num_files = len(segStats_files)
    test_perf = stats.SegmentationMetrics()
    print(f"Extracting confusion matrix statistics from {num_files} files.", file=sys.stderr)
    for i, ss_file in enumerate(segStats_files):
        with open(ss_file, 'r') as f:
            seg_stats = json.load(f)
        test_perf.update(
            seg_stats['seg_size'], 
            seg_stats['tgt_size'], 
            seg_stats['intersect'], 
            seg_stats['union'], 
            seg_stats['dice'], 
            seg_stats['jaccard']
        )
       
    ofile = args.output_dir / "segmentation_performance.json"
    print(f"Saving output as: {ofile}", file=sys.stderr)
    with open(ofile, 'w') as f:
        json.dump(test_perf(), f, indent=4)

if __name__ == '__main__':
    TESTING = False
    #-- Get input
    parser = argparse.ArgumentParser(
        'compute_classification_performance', 
        description=(
        "Extract _confMatx.json statistics from iff_eventTracking"
        " directories and compute classification performance across"
        " all files."
        ),
        parents=[get_args_parser()])
    if TESTING:
        print("TESTING!", file=sys.stderr)
        args = parser.parse_args((
            # r"D:\dtu\speciale\ipp\processed\track_065_descending\iff_eventTracking"
            r" D:\dtu\speciale\ipp\processed\track_010_ascending\segmentation\dino"
            " -o"
            r" D:\dtu\speciale\ipp\processed\track_065_descending\iff_eventTracking"
        ).split())
    else: 
        args = parser.parse_args()
        
    main(args)
    
    