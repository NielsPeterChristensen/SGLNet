#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THIS FILE IS JUST A MANUAL HELPER. PROBABLY DELETE LATER"""

from glob import glob
from SGLNet.PyIPP.iff_dd import IffDD
from argparse import ArgumentParser, Namespace
import SGLNet.Corefunc.utils as utils
from pathlib import Path
from typing import Union


DEFAULT_OUTDIR = 'segmentation'


def parse_arguments() -> Namespace:
    """
    Import arguments parsed through the command line call.

    Returns
    -------
    ArgumentParser.parse_args()
        Class containing the parsed arguments.

    """
    parser = ArgumentParser(description="Store iff_dd phase as images for easy segmentation.")
    
    parser.add_argument(
        "iff_dd_paths",
        nargs = '*',
        metavar = "./track_<xxx>_<pass>/iff_dd/<iff_dd_interval>.pha",
        help = "Path/to/iff_dd_images. Wildcards are possible.")
    parser.add_argument(
        "-o",
        "--out_dir",
        type = str,
        default = None,
        metavar = "Path/to/out_dir",
        help = "Path to output directory. It is created if it does not exist.\n" \
               + "Default location is ../../segmentation relative to iff_dd_paths.")
    parser.add_argument(
        "-f",
        "--form",
        type = str,
        default = 'phase',
        metavar = '<class str>',
        help = "Data form. Options are ['phase', 'recta', 'polar']. Default is 'phase'.")
    parser.add_argument(
        "-c",
        "--channels",
        type = list,
        default = [0,1,2],
        metavar = '<class list>',
        help = "Color channels where phase is included when --form='phase'.")
    parser.add_argument(
        "-m",
        "--mag",
        type = utils.bool_flag,
        default = False,
        metavar = '<class bool>',
        help = "Bool flag whether to include magnitude when --form='recta'.")
    
    return parser.parse_args()


def make_dir(path: Path):
    path = Path(str(path)) # Make sure input is type Path
    if not path.is_dir():
        path.mkdir()


def main(iff_dd_paths: str, out_dir: str = None, form: str = 'phase', channels: list = [0,1,2], mag: bool = False) -> None:
    
    files = []
    if isinstance(iff_dd_paths, list):
        for p in iff_dd_paths:
            files.extend(glob(p))
    if isinstance(iff_dd_paths, str):
        files.extend(glob(iff_dd_paths))
        
    nfiles = len(files)
    for i, file in enumerate(files):
        print(f"({i+1}/{nfiles}): Evaluating {file}")
        Obj = IffDD(file)
        odir = out_dir or (Obj.track_dir / DEFAULT_OUTDIR)
        odir = Path(str(odir))
        make_dir(odir)  
        ofile = (odir / (Obj.file_name + f'_{form}')).with_suffix('.png')
        if form == 'phase':
            utils.phase_image(Obj.pha(), channels).save(ofile)
        if form == 'recta':
            if mag:
                utils.recta_image(Obj.pha(), Obj.mag()).save(ofile)
            else:
                utils.recta_image(Obj.pha()).save(ofile)
        if form == 'polar':
            utils.polar_image(Obj.pha(), Obj.mag()).save(ofile) 
        print("  Success")


if __name__ == "__main__":
    
    args = parse_arguments()
    main(
        iff_dd_paths = args.iff_dd_paths,
        out_dir = args.out_dir,
        form = args.form,
        channels = args.channels,
        mag = args.mag)