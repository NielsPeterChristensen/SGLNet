#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 6 2025

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
        "iff_eventTracking_dirs",
        nargs = "+",
        action = "extend",
        metavar = "/path/to/iff_eventTracking/ main directory",
        type = str,
        help = ("Path to one or more iff_eventTracking directory. Directory is located inside the"
                " track_[XYZ]_[DIRECTION] directory, and it contains subdirectories"
                " with names corresponding to iff_dd acquisitions. These subdirectories"
                " should each contain a file ending in '_confMatx.json' for this"
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
    parser.add_argument(
        "-tr",
        "--include_training",
        metavar = "<bool>",
        type = utils.bool_flag,
        default = True,
        help = ("Whether to also compute performance on training scenario."
                " Must be set to False if _ConfMatx.json files does not contain"
                " 'train' entry. Default is True.")
        )
    
    return parser

def main(args: argparse.Namespace) -> None:
    
    print("\nRunning compute_classification_performance.py", file=sys.stderr)
    
    #-- Make paths to Path objects
    args.iff_eventTracking_dirs = np.array([Path(d).resolve() for d in args.iff_eventTracking_dirs])
    args.output_dir = Path(args.output_dir).resolve()
    
    #-- Check inputs
    input_is_dir = np.array([d.is_dir() for d in args.iff_eventTracking_dirs])
    if not all(input_is_dir):
        print(f"Invalid eventTracking directory {args.iff_eventTracking_dirs[~input_is_dir]}", file=sys.stderr)
        sys.exit(1)
    #-- Check output
    if not args.output_dir.is_dir():
        print(f"Generating output directory {str(args.output_dir)}", file=sys.stderr)
        args.output_dir.mkdir(exist_ok=True)
        
    #-- Get derived files
    iff_subdirs = np.array([p for d in args.iff_eventTracking_dirs for p in d.iterdir() if p.is_dir()])
    confMatx_files = np.array([d / f"{d.name}_confMatx.json" for d in iff_subdirs])
    
    #-- Check derived files
    confMatx_exists = np.array([f.is_file() for f in confMatx_files])
    if not all(confMatx_exists):
        print(f"Not all _confMatx.json files were found. Missing are {confMatx_files[~confMatx_exists]}.", file=sys.stderr)
        sys.exit(1)
        
    #-- Loop over directories
    num_files = len(confMatx_files)
    test_perf = stats.ClassificationMetrics()
    fp_near_mask = []
    if args.include_training:
        train_perf = stats.ClassificationMetrics()
    print(f"Extracting confusion matrix statistics from {num_files} files.", file=sys.stderr)
    for i, cm_file in enumerate(confMatx_files):
        with open(cm_file, 'r') as f:
            conf_matx = json.load(f)
        test_perf.update(
            conf_matx['test']['tp'],
            conf_matx['test']['tn'],
            conf_matx['test']['fp'],
            conf_matx['test']['fn']
        )
        fp_near_mask.append(conf_matx['test']['fp_near_mask'])
        if args.include_training:
            train_perf.update(
                conf_matx['train']['tp'],
                conf_matx['train']['tn'],
                conf_matx['train']['fp'],
                conf_matx['train']['fn']
            )
            
    output_dict = {'test': {'fp_near_mask': sum(fp_near_mask), **test_perf()}}
    if args.include_training:
        output_dict = {**output_dict, 'train': train_perf()}
    
    ofile = args.output_dir / "classification_performance.json"
    print(f"Saving output as: {ofile}", file=sys.stderr)
    with open(ofile, 'w') as f:
        json.dump(output_dict, f, indent=4)

if __name__ == '__main__':
    TESTING = False
    #-- Get input
    parser = argparse.ArgumentParser(
        'compute_classification_performance', 
        description=("Extract _confMatx.json statistics from iff_eventTracking"
                     " directories and compute classification performance across"
                     " all files."),
        parents=[get_args_parser()])
    if TESTING:
        print("TESTING!", file=sys.stderr)
        args = parser.parse_args((
            r"D:\dtu\speciale\ipp\processed\track_065_descending\iff_eventTracking"
            r" D:\dtu\speciale\ipp\processed\track_010_ascending\iff_eventTracking"
            " -o"
            r" D:\dtu\speciale\ipp\processed\track_065_descending\iff_eventTracking"
        ).split())
    else: 
        args = parser.parse_args()
        
    main(args)
    
    