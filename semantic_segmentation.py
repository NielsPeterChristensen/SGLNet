#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Docstring
# =============================================================================


"""
[Description]

:Authors
    NPKC / 01-05-2025 / creation / s203980@dtu.dk

:Todo
    Create semantic segmentation from dino features

:References
    [1] Mathilde Caron et. al. Emerging Properties in Self-Supervised Vision 
    Transformers https://arxiv.org/abs/2104.14294
    [2] Oriane Siméoni et al. Localizing Objects with Self-Supervised
    Transformers and no Labels https://arxiv.org/abs/2109.14279

:Note:
    Requires features extracted with DINO [1] and uses object detection from LOST [2]
"""

# =============================================================================
# Packages
# =============================================================================

#-- Utilities
import argparse
import sys
import os
import numpy as np
import numpy.typing as npt
import pickle
import json
import torch
from pathlib import Path
from PIL import Image
#-- My own
import SGLNet.Corefunc.utils as utils
import SGLNet.Plotting.plot_utils as plot_utils
import SGLNet.Segmentation.semantic as semseg

# =============================================================================
# Constants
# =============================================================================

VALID_METHODS = ['dino', 'pca', 'correlation', 'degree']
VALID_EMBEDDINGS = ['query', 'key', 'value', 'patch']
VALID_ACCUMULATION = ['max', 'mean']

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
    parser = argparse.ArgumentParser('semantic_segmentation', add_help=False)
    
    parser.add_argument(
        "iff_eventTracking_dir",
        metavar = "/path/to/iff_eventTracking/ main directory",
        type = str,
        help = ("Path to the iff_eventTracking directory. Directory is located inside the"
                " track_[XYZ]_[DIRECTION] directory, and it contains subdirectories"
                " with names corresponding to iff_dd acquisitions. These subdirectories"
                " should each contain files ending in '_embed.pkl' and '_meta.json"
                " for this script to work.")
        )
    parser.add_argument(
        "-o",
        "--output_dir",
        metavar = "/path/to/output_dir",
        default = None,
        type = str,
        help = ("Path to output directory to save performance metrics."
                " Default is None, meaning in the same subdirectory as _embed.pkl"
                " are stored. If output directory doesn't exist, it is created.")
        )
    parser.add_argument(
        '--which_method',
        type = str,
        default = 'corr',
        metavar = "<str>",
        help = ("Specify which method to use. Options are\n"
                "  dino:            Using DINO self-attention\n"
                "  pca:             Using first principle component of embeddings specified by --which_embeddings\n"
                "  corr(elation):   Using LOST correlation on embeddings specified by --which_embeddings\n"
                "  deg(ree):        Using LOST binary inverse correlation (degree) on embeddings specified by which_embeddings\n"
                "Note: methods behave differents, and some embeddings or better"
                " suited for some methods than others (see --which_embeddings)."
                " Default is 'corr'."))
    parser.add_argument(
        '--which_embeddings',
        type = str,
        default = 'k',
        metavar = "<str>",
        help = ("Specify which embeddings to use. Options are\n"
                "  q(uery):     Query matrix\n"
                "  k(ey):       Key matrix\n"
                "  v(alue):     Value matrix\n"
                "  p(atch):     Patch tokens\n"
                "Note: Some embeddings are better suited than others in certain scenarios."
                " For instance, query and patch embeddings does not work with LOST"
                " (i.e. corr and deg in --which_method). Default is 'k'."))
    parser.add_argument(
        '--which_accumulation',
        type = str,
        default = 'max',
        metavar = "<str>",
        help = ("Specify how salient embeddings are accumulated for overlapping chunks. Options are\n"
                "  max:     Take max of salient embeddings for overlapping chunks\n"
                "  mean:    Compute mean of salient embedings for overlapping chunks\n"
                "Note: Using 'max' is generally better at picking up small/faint features,"
                " but also picks up more noise. Default is 'max'"))
    parser.add_argument(
        '--dino_mass_frac',
        type = float,
        default = 0.6,
        metavar = "<float>",
        help = ("Fractional mass of self-attention to keep when which_method = dino."
                " Default is 0.6 (recommented by authors of DINO)."))
    parser.add_argument(
        '--min_patch',
        type = int,
        default = 10,
        metavar = "<int>",
        help = ("Filter connected components to only keep segmentations with"
                " a minimum of min_patch number of connected patches."
                " Default is 10."))
    parser.add_argument(
        '--connectivity',
        type = int,
        default = 1,
        choices = [1,2],
        metavar = "<int>",
        help = ("Connectivity to use for connected components analysis."
                " 1: 4-nearest and 2: 8-nearest. Default is 1."))
    parser.add_argument(
        '--threshold',
        type = float,
        default = None,
        metavar = "<float>",
        help = ("Threshold used to produce binary mask from salient map."
                " Depending on --which_method different thresholds should be used."
                " If threshold = None default values are used:\n"
                "  if which_method=dino:    0.5 (meaning every 1 in 2"
                " self-attn-heads should have attented to a patch before it is segmented,"
                " which is 3/6 heads for vit_small and 6/12 heads for vit_base).\n"
                "  if which_method=pca:     5 (could also be 0, ie. all"
                " patches where the first principle component is positive, but using"
                " 5 reduces noise slightly).\n"
                "  if which_method=corr:    10 (could also be 0, ie. all patches"
                " where the correlation with the seed patch is positive, but using"
                " 10 reduces noise slightly.)\n"
                "  if which_method=deg:     1.8 (somewhat arbitrary, but completely"
                " uncorrelated patches will have a value of 1, while strongly correlated"
                " pixels are around 3.\n"
                "Default is None."))
    parser.add_argument(
        '--sigma',
        type = float,
        default = None,
        metavar = "<float>",
        help = ("Sigma used to apply Gaussian smoothing to salient map before thresholding."
                " Depending on --which_method different alphas should be used."
                " If sigma = None default values are used:\n"
                "  if which_method=dino:    1 (to reduce noise)\n"
                "  if which_method=pca:     1 (to reduce noise)\n"
                "  if which_method=corr:    0 (smoothing breaks segmentation)\n"
                "  if which_method=deg:     1 (to reduce noise)\n"
                "Default is None."))
    parser.add_argument(
        '--overwrite',
        type = utils.bool_flag,
        default = False,
        metavar = "<bool>",
        help = ("Specify whether to overwrite existing data and images."
                " Default is False.")
        )
    parser.add_argument(
        '--save_salient',
        type = utils.bool_flag,
        default = False,
        metavar = "<bool>",
        help = ("Specify whether to save salient feature masks (intermediate"
                " product of segmentation). They are currently not used for any"
                " downstream tasks outside this script. Default is False.")
        )
    parser.add_argument(
        '--iff_dd_dir',
        type = str,
        default = None,
        metavar = "<str>",
        help = ("Optional. Path to directory containing iff_dd .pha file."
                " This should only be given if user wishes to have visualisations"
                " of segmentation masks overlaid on phase images."
                " Otherwise leave as default None.")
        )
    parser.add_argument(
        '--imscale',
        type = float,
        default = 1.0,
        metavar = "<float>",
        help = ("Optional. Set scale of visualisation images. Only applies to"
                " _overlay.png and _salient.png and not _binaryMask.png. Useful"
                " to use 0.2 for 'draft' images, as this speeds up the saving"
                " process significantly. Must be [0, 1]. Default is 1.0.")
        )
    
    return parser

def main(args: argparse.Namespace) -> None:
    
    print("\nRunning semantic_segmentation.py", file=sys.stderr)
    
    #-- Check input
    if not any(args.which_method.lower() in method for method in VALID_METHODS):
        print(f"which_method={args.which_method} not found amongst valid methods {VALID_METHODS}.", file=sys.stderr)
        sys.exit(1)
    if not any(args.which_embeddings.lower() for embedding in VALID_EMBEDDINGS):
        print(f"which_embeddings={args.which_embeddings} not found amongst valid embeddings {VALID_EMBEDDINGS}.", file=sys.stderr)
        sys.exit(1)
    if not any(args.which_accumulation.lower() in accumulation for accumulation in VALID_ACCUMULATION):
        print(f"which_accumulation={args.which_accumulation} not found amongst valid accumulation methods {VALID_ACCUMULATION}.", file=sys.stderr)
        sys.exit(1)
        
    if args.threshold is None:
        if args.which_method.lower() in 'dino':
            args.threshold = 1/2
        elif args.which_method.lower() in 'pca':
            args.threshold = 5 
        elif args.which_method.lower() in 'correlation':
            args.threshold = 10
        elif args.which_method.lower() in 'degree':
            args.threshold = 1.8
    
    if args.sigma is None:
        if args.which_method.lower() in 'correlation':
            args.sigma = 0
        else:
            args.sigma = 1

    #-- Set paths to pathlib.Path
    args.iff_eventTracking_dir = Path(args.iff_eventTracking_dir).resolve()
    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir).resolve()
    if args.iff_dd_dir is not None:
        args.iff_dd_dir = Path(args.iff_dd_dir).resolve()

    #-- Check input
    if args.which_method.lower() in 'correlation' or args.which_method.lower() in 'degree':
        if args.which_embeddings.lower() in 'query' or args.which_embeddings.lower() in 'patch':
            print('When using which_method with "corr" or "deg" then which_embedding '
                  + f'must use "k" or "v", but was {args.which_embeddings}.', file=sys.stderr)
            sys.exit(1)
    if not args.iff_eventTracking_dir.is_dir():
        print(f"Invalid eventTracking directory {str(args.iff_eventTracking_dir)}", file=sys.stderr)
        sys.exit(1)
    if args.iff_dd_dir is not None and not args.iff_dd_dir.is_dir():
        print(f"Invalid iff dd directory {str(args.iff_dd_dir)}", file=sys.stderr)
        sys.exit(1)
        
    #-- Check output
    if args.output_dir is not None and not args.output_dir.is_dir():
        print(f"Generating output directory {str(args.output_dir)}", file=sys.stderr)
        args.output_dir.mkdir(exist_ok=True, parents=True)

    #-- Get derived files
    basenames = np.array([str(d.name) for d in args.iff_eventTracking_dir.iterdir() if d.is_dir()])
    iff_subdirs = np.array([args.iff_eventTracking_dir / name for name in basenames])
    embed_files = np.array([subdir / f"{name}_embed.pkl" for (subdir, name) in zip(iff_subdirs, basenames)])
    meta_files = np.array([subdir / f"{name}_meta.json" for (subdir, name) in zip(iff_subdirs, basenames)])
    if args.iff_dd_dir is not None:
        pha_files = np.array([next((f for f in args.iff_dd_dir.glob("*.pha") if str(name) in f.name), None) for name in basenames], dtype=object)
        # pha_files = np.array([list(args.iff_dd_dir.glob(f"{str(name)}*.pha"))[0] for name in basenames])
        if len(pha_files) != len(embed_files) or any(f is None for f in pha_files):
            print("Unable to find all iff_dd phase files corresponding to embed files.", file=sys.stderr)
            sys.exit(1)
    else:
        pha_files = [None]*len(embed_files)
            
    #-- Check derived files
    embed_exists = np.array([f.is_file() for f in embed_files])
    if not all(embed_exists):
        print(f"Not all _embed.pkl files were found. Missing are {embed_files[~embed_exists]}.", file=sys.stderr)
        sys.exit(1)
    meta_exists = np.array([f.is_file() for f in meta_files])
    if not all(meta_exists):
        print(f"Not all _meta.json files were found. Missing are {meta_files[~meta_exists]}.", file=sys.stderr)
        sys.exit(1)
           
    #-- Loop over files
    print(f"Looping over files", file=sys.stderr)
    num_files  = len(embed_files)
    for i, (embed_file, meta_file, pha_file) in enumerate(zip(embed_files, meta_files, pha_files)):
        print(f"  ({i+1}/{num_files}): Segmenting {basenames[i]}", file=sys.stderr)
        perform_segmentation(embed_file, meta_file, pha_file, args)
    print("All files completed")
    
    
def perform_segmentation(embed_file: os.PathLike, meta_file: os.PathLike, pha_file: os.PathLike, args: argparse.Namespace) -> None:
    
    #-- Load data
    with open(embed_file, 'rb') as f:
        embed_data = pickle.load(f)
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)
    #-- Extract useful parameters
    ps = meta_data['patch_size']
    cs = meta_data['chunk_size']
    nh = meta_data['num_heads']
    nt = meta_data['num_tokens']
    h_dim = meta_data['head_dim']
    e_dim = meta_data['embed_dim']
    imsize = meta_data['imsize']
    #-- Setup mask
    mask_object = semseg.SalientMask(imsize, ps, cs, accum=args.which_accumulation)
    
    #-- Loop over chunks
    saliency_map = []
    embeds = []
    bboxes = []
    for chunk_embed in embed_data:
        
        #-- Extract data from structured array
        id_ = chunk_embed['id'].astype(np.int32)
        bbox = chunk_embed['bbox'].astype(np.int32)
        cls_embed = chunk_embed['cls_embed'].astype(np.float32)
        patch_embed = chunk_embed['patch_embed'].astype(np.float32)
        output_embed = chunk_embed['output_embed'].astype(np.float32)
        qkv = torch.Tensor(chunk_embed['qkv'].astype(np.float32))
        attn = torch.Tensor(chunk_embed['attn'].astype(np.float32))
        bboxes.append(bbox)
        
        #-- Get dimensions of feature map
        w_featmap = h_featmap = int((nt-1) ** 0.5)
        dim_featmap = (w_featmap, h_featmap)
        
        #-- query, key, and value matrices
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.transpose(0, 1).reshape(nt, -1)
        k = k.transpose(0, 1).reshape(nt, -1)
        v = v.transpose(0, 1).reshape(nt, -1)

        #-- embedding selection (excluding CLS token at idx 0 for qkv)
        if args.which_embeddings[0].lower() == "p":
            embed = patch_embed
        elif args.which_embeddings[0].lower() == "q":
            embed = q[1:, :]
        elif args.which_embeddings[0].lower() == "k":
            embed = k[1:, :]
        elif args.which_embeddings[0].lower() == "v":
            embed = v[1:, :]
        else:
            print(f"    {args.which_embeddings} not recognized as a valid input for which_embeddings.", file=sys.stderr)
            sys.exit(1)
        
        if 'dino' in args.which_method.lower():
            salient = semseg.DINO_segmentation(attn, nh, dim_featmap, args.dino_mass_frac)
            saliency_map.append(salient)
            
        elif 'pca' in args.which_method.lower():
            embeds.append(embed.reshape(*dim_featmap, e_dim))
            
        elif 'corr' in args.which_method.lower():
            salient, _ = semseg.LOST_segmentation(embed, dim_featmap, (ps, ps), (3, cs, cs))
            saliency_map.append(salient)
        
        elif 'deg' in args.which_method.lower():
            _, salient = semseg.LOST_segmentation(embed, dim_featmap, (ps, ps), (3, cs, cs))
            saliency_map.append(salient)
        
        else:
            print(f"    {args.which_method} not recognized as a valid input for which_method.", file=sys.stderr)
            sys.exit(1)
            
    if 'pca' in args.which_method.lower():
        saliency_map = semseg.consistent_PCA_segmentation(embeds)
        
    for (bbox, mask) in zip(bboxes, saliency_map):
        mask_object[bbox] = mask
        
    #-- Setup output dir
    bname = embed_file.parent.name
    output_dir = args.output_dir or embed_file.parent
    output_dir = output_dir / bname
    output_dir.mkdir(exist_ok=True)
    #-- Save binary mask
    binary_mask = mask_object(args.threshold, args.sigma, args.min_patch, args.connectivity)
    fname = output_dir / f"{bname}_binaryMask.png"
    if args.overwrite or not fname.exists():
        print("    Saving binary mask", file=sys.stderr)
        binary_mask.crop((0, 0, imsize[0], imsize[1])).save(fname)
    #-- Save outlines
    outlines = mask_object.get_outlines(sigma=20, smoothness=50.0, n_points=200)
    fname = output_dir / f"{bname}_outlines.pkl"
    if args.overwrite or not fname.exists():
        print("    Saving outlines", file=sys.stderr)
        with open(fname, 'wb') as f:
            pickle.dump(outlines, f)
    #-- Save segmentation overlayed on phase image (optional)
    fname = output_dir / f"{bname}_overlay.png"
    if pha_file is not None and (args.overwrite or not fname.exists()):
        print("    Saving overlay image", file=sys.stderr)
        phase_img = utils.pad_image(utils.direct_phase_image(pha_file), cs)
        # phase_img = utils.direct_phase_image(pha_file)
        img = mask_object.apply_to(phase_img, args.threshold, args.sigma, args.min_patch, args.connectivity)
        plot_utils.draw_outlines(img, outlines[:, :, ::-1], color='red')
        if args.imscale > 0 and args.imscale < 1:
            img = plot_utils.rescale_img(img, args.imscale)
        img.crop((0, 0, imsize[0], imsize[1])).save(fname)  
    #-- Save salient mask (optional)
    fname = output_dir / f"{bname}_salient.png"
    if args.save_salient and (args.overwrite or not fname.exists()):
        print("    Saving salient mask", file=sys.stderr)
        img = mask_object.get_salient()
        if pha_file is not None:
            img = np.array(img)
            img[img[:,:,3]>0, 3] = 180
            img = Image.fromarray(img, mode='RGBA')
            img = Image.alpha_composite(phase_img, img)
        if args.imscale > 0 and args.imscale < 1:
            img = plot_utils.rescale_img(img, args.imscale)
        img.crop((0, 0, imsize[0], imsize[1])).save(fname)
        

if __name__ == "__main__":
    TESTING = False
    #-- Get input
    parser = argparse.ArgumentParser('semantic_segmentation', description="Perform semantic segmentation.", parents=[get_args_parser()])
    if TESTING:
        print("TESTING!", file=sys.stderr)
        args = parser.parse_args((
            r"D:\dtu\speciale\ipp\processed\track_010_ascending\iff_eventTracking"
            " -o"
            r" D:\dtu\speciale\ipp\processed\track_010_ascending\segmentation\dino"
            # " --save_salient"
            # " True"
            # " --iff_dd_dir"
            # r" D:\dtu\speciale\ipp\processed\track_010_ascending\iff_dd"
            " --which_method"
            " dino"
            # " --which_embeddings"
            # " q"
            " --imscale"
            " 0.2"
            " --overwrite"
            " True"
        ).split())
    else: 
        args = parser.parse_args()
        
    main(args)