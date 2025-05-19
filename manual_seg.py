#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manually segment gim events into binary masks and contours

@author: niels, 2025
"""

# =============================================================================
# Packages
# =============================================================================

import os
import sys
import numpy as np
from argparse import ArgumentParser
from typing import Union
from glob import glob
import SGLNet.PyIPP.gim_events as IPP_gim
import SGLNet.PyIPP.grd as IPP_grd
import SGLNet.Segment.threshold_segmentation as thres_seg
import SGLNet.Corefunc.utils as utils
from datetime import datetime, timezone
from configparser import ConfigParser

# =============================================================================
# Functions
# =============================================================================

def parse_arguments():
    """
    Import arguments parsed through the command line call.

    Returns
    -------
    ArgumentParser.parse_args()
        Class containing the parsed arguments.

    """
    parser = ArgumentParser(description="Manually segment cropped gim events" \
                            + " into binary masks and contours.")
    
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
        type = utils.bool_flag,
        default = True,
        metavar='<bool>',
        help="Use unwrapped phase image for visualization during" \
            +"segmentation. If False, the wrapped phase is shown instead.")
    return parser.parse_args()
    

def get_input(event_dirs: Union[list, str]) -> list:
    """
    Takes the parsed event directories and returns a list of path strings
    for further digestion.

    Parameters
    ----------
    event_dirs : list | str
        Parsed event directories. Can be a string, a list of strings,
        or a glob wildcard.

    Returns
    -------
    list
        List containing path strings to event directories.

    """
    #-- Get input dir(s) as list of paths
    if isinstance(event_dirs, list):
        for item in event_dirs: 
            if not isinstance(item, str):
                sys.exit("event_dirs must be a string or a list of strings.")
        paths = event_dirs
    elif isinstance(event_dirs, str):
        paths = glob(event_dirs)
    else:
        sys.exit("event_dirs must be a string or a list of strings.")
    #-- Check that paths exist
    for p in paths:
        if not os.path.isdir(p):
            sys.exit("path {p} does not exists.")
    return paths


def make_dir(directory: str):
    """
    Check if directory exists and create it if not.

    Parameters
    ----------
    directory : str
        Path string for directory to be created.

    Returns
    -------
    None.

    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


def parent_dir(path: str) -> str:
    """
    Get parent directory from a path using os.path functions.

    Parameters
    ----------
    path : str
        A string path.

    Returns
    -------
    str
        Path to parent directory.

    """
    return os.path.abspath(os.path.join(path, '..'))


def make_grd(file_path, GimGrdObj, EventObj, BTS):
    ic = ConfigParser()
    ic.optionxform = str
    ic['ippVersion'] = {
        '  module': 'segmentation',
        '  compileDate': f'{GimGrdObj.ippVersion.compileDate}',
        '  revision': f'{GimGrdObj.ippVersion.revision}',
        '  cmdDate': f'{command_line}',
        '  cmdLine': f'{command_date}'
    }
    ic['fileDesc'] = {
        '  endian': f'{GimGrdObj.fileDesc.endian}',
        '  NaN': f'{GimGrdObj.fileDesc.NaN}',
    }
    ic['datum'] = {
        '  verDatum': f'{GimGrdObj.datum.verDatum}',
        '  horDatum': f'{GimGrdObj.datum.horDatum}',
        '  projection': f'{GimGrdObj.datum.projection}',
        '  latTS': f'{GimGrdObj.datum.latTS}',
        '  lon0': f'{GimGrdObj.datum.lon0}'
    }
    ic['eventInfo'] = {
        '  nrows': f'{EventObj.Phu.event_nrows}',
        '  ncols': f'{EventObj.Phu.event_ncols}',
        '  li0': f'{EventObj.Phu.Coords.li0}',
        '  sa0': f'{EventObj.Phu.Coords.sa0}',
        '  li1': f'{EventObj.Phu.Coords.li1}',
        '  sa1': f'{EventObj.Phu.Coords.sa1}',
        '  nrowsIff': f'{EventObj.nrows}',
        '  ncolsIff': f'{EventObj.ncols}',
        '  nEvent': f'{EventObj.nr_event}'
    }
    ic['segmentation'] = {
        '  minArea': f'{BTS.min_area}',
        '  posThres': f'{BTS.positive_threshold}',
        '  negThres': f'{BTS.negative_threshold}',
        '  count': f'{BTS.count}'
    }
    
    with open(file_path + '.grd', 'w') as configfile:
        ic.write(configfile)


def write_output(
        mask: np.ndarray, 
        contours: list, 
        event_dir: str, 
        out_dir: str,
        GimGrdObj: IPP_grd.GimGrd,
        EventObj: IPP_gim.GimEvent,
        BTS: thres_seg.BasicThresholdSegmentation):
    """
    Write segmentation output to files stored in event_segmentation directory.
    Binary masks are stored as 32-bit big-endian .seg files.
    Contours are stored as .txt files.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask from segmentation.
    contours : list
        List of (N,2)-np.ndarrays with segmentation contours.
    event_dir : str
        Path to the event directory of the current segmentation.
        If no out_dir is given, then the event_segmentation directory is
        created in the grandparent directory of the event_dir, corresponding
        to the track_<xxx>_<pass> directory.
    out_dir : str
        Custom directory for storing output files. If None is parsed, a 
        new default directory is created using event_dir (see above).
    GimGrdObj : GimGrd
        GimGrd instance for the given event.
    EventObj : GimEvent
        GimEvent instance for the given event.
    BTS : BasicThresholdSegmentation
        BasicThresholdSegmentation instance for the given event segmentation.
    
    Returns
    -------
    None.

    """
    base_name = os.path.basename(os.path.dirname(event_dir))
    #-- Create output directory in event_dir grandparent if out_dir is None
    if out_dir is None:
        out_dir = EventObj.segmentation_path
    else:
        if not isinstance(out_dir, str):
            sys.exit("out_dir must be a string.")
    #-- Create directory and prepare file path
    make_dir(out_dir)
    file_path = os.path.join(out_dir, f"{base_name}")
    #-- Store mask as 32-bit big-endian file
    mask.astype(">f4").tofile(file_path + ".seg")
    #-- Store contours as text file
    with open(file_path + ".txt", "w") as file:
        for c in contours:
            np.savetxt(file, c, fmt="%.6f")
            file.write("\n")
    #-- Store grd file
    make_grd(file_path, GimGrdObj, EventObj, BTS)


def main(event_dirs: Union[str, list], 
         out_dir: Union[str, None], 
         min_area: int, 
         show_wrap: bool,
         command_line: str, 
         command_date: str):
    """
    Perform manual threshold segmentation and write output to files.

    Parameters
    ----------
    event_dirs : str | list
        One or more directories with gim events.
    out_dir : str | None
        Output directory to write files to. Default from argparse is None.
    min_area : int
        Minimum pixel area for segmentations. Default from argparse is 100.
    show_wrap : bool
        Boolean flag to choose whether wrapped or unwrapped phase image
        is used for visualization during manual segmentation.
        Default from argparse is True.
    command_line : str
        String with the command line parsed.
    command_date : str
        String with the date and time (UTC) of the command parsed.

    Returns
    -------
    None.

    """
    #-- Get directories as list of strings
    event_dirs = get_input(event_dirs)
    #-- Loop over directories
    for path in event_dirs:
        #-- Import event
        EventObj = IPP_gim.GimEvent(path)
        unwrap_img = EventObj.Phu.data
        if show_wrap is True:
            wrap_img = EventObj.Phr.data
        else:
            wrap_img = None
        #-- Perform segmentation
        BTS = thres_seg.BasicThresholdSegmentation(unwrap_img, wrap_img, min_area)
        segmentation = BTS.mask
        contours = BTS.contours
        #-- Write output
        GimGrdObj = IPP_grd.GimGrd(os.path.join(os.path.join(
            EventObj.gim_path, EventObj.basename), EventObj.basename + '.grd'))
        write_output(segmentation, contours, path, out_dir, 
                     GimGrdObj, EventObj, BTS)

# =============================================================================
# Execution
# =============================================================================

if __name__ == "__main__":
    #-- FOR TESTING PURPOSE ONLY
    if False:
        os.chdir("C:\\Users\\niels\\Documents\\Skole\\Kandidat\\Speciale\\" \
                 + "Code\\WSL_func")
        class Args:
            def __init__(self):
                self.event_dirs = "C:\\Users\\niels\\Documents\\Skole\\" \
                    + "Kandidat\\Speciale\\Data\\track_099_ascending\\" \
                    + "gim_events\\20171101_20171107_20171107_20171113_0"
                self.out_dir = None
                self.lowpass = 100
                self.show_unwrap = True
        args = Args()
    #-- Get parsed arguments
    args = parse_arguments()
    #-- Get command metadata
    command_line = ' '.join(sys.argv)
    command_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # command_line = '' '
    #-- Execute main()
    main(
        args.event_dirs, 
        args.out_dir, 
        args.lowpass, 
        args.show_unwrap, 
        command_line, 
        command_date)