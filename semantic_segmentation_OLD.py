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
import datetime
import pickle
import pkg_resources
import torch
import torch.nn as nn
import skimage as ski
import matplotlib as mpl
import scipy
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from glob import glob
from PIL import Image, ImageOps
#-- My own
import SGLNet.Corefunc.utils as utils
import SGLNet.Plotting.plot_utils as plot_utils
import SGLNet.Segmentation.semantic as semseg
#-- From LOST [2]
import SGLNet.lost.object_discovery as object_discovery
import SGLNet.lost.visualizations as visualizations

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
        "pkl_files",
        nargs = '+',
        metavar = "/path/to/embed_data.pkl",
        help = "Example: /path/to/network_test/vit_base_16_phase_224/*_embed.pkl. " \
               + "Wildcards are possible.\n")
    parser.add_argument(
        "-o",
        "--output_dir",
        type = str,
        metavar = "<str>",
        default = "../semantic_segmentation",
        help = "Path to output directory to save segmentations. " \
               + "Default is './semantic_segmentation' relative to pickle files.")
    parser.add_argument(
        '--iff_dd_dir',
        type = str,
        default = '../../../iff_dd',
        metavar = "<str>",
        help = "Path to directory containing iff_dd .pha file. Can be absolute or relative. " \
               + "Default is ../../../iff_dd relative to pkl files.")
    parser.add_argument(
        '--test_segmentation_dir',
        type = str,
        default = None,
        metavar = "<str>",
        help = "Path to directory containing testing mask file. Can be absolute or relative. " \
               + "Default is 'None' (no testing).\nNote: Mask filenames are " \
               + "expected to be the basename of corresponding .pha file " \
               + "and potentially some arbitrary extra string. For example " \
               + "20160726_20160807_20160807_20160819_phase-01.png.")
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
                "  if which_method=dino:    0.67 (meaning every 2 in 3"
                " self-attn-heads should have attented to a patch before it is segmented,"
                " which is 4/6 heads for vit_small and 8/12 heads for vit_base).\n"
                "  if which_method=pca:     5 (could also be 0, ie. all"
                " patches where the first principle component is positive, but using"
                " 5 reduces noise slightly).\n"
                "  if which_method=corr:    10 (could also be 0, ie. all patches"
                " where the correlation with the seed patch is positive, but using"
                " 10 reduces noise slightly.)\n"
                "  if which_method=deg:     1.5 (somewhat arbitrary, but completely"
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
        '--save_image',
        type = utils.bool_flag,
        default = False,
        metavar = "<bool>",
        help = "Specify whether to save each segmentation. Default is False.")
    parser.add_argument(
        '--overwrite',
        type = utils.bool_flag,
        default = False,
        metavar = "<bool>",
        help = "Specify whether to overwrite existing data and images." \
               + "Default is False.")
    
    return parser


def main(args: argparse.Namespace) -> None:
    
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
            args.threshold = 2/3
        elif args.which_method.lower() in 'pca':
            args.threshold = 5 
        elif args.which_method.lower() in 'correlation':
            args.threshold = 10
        elif args.which_method.lower() in 'degree':
            args.threshold = 1.5
    
    if args.sigma is None:
        if args.which_method.lower() in 'correlation':
            args.sigma = 0
        else:
            args.sigma = 1
    
    #-- Check input
    if args.which_method.lower() in 'correlation' or args.which_method.lower() in 'degree':
        if args.which_embeddings.lower() in 'query' or args.which_embeddings.lower() in 'patch':
            print('When using which_method with "corr" or "deg" then which_embedding '
                  + f'must use "k" or "v", but was {args.which_embeddings}.', file=sys.stderr)
            sys.exit(1)

    #-- Get pickle files
    if isinstance(args.pkl_files, str):
        args.pkl_files = [args.pkl_files]
    if os.name == "nt": # For windows
        if len(args.pkl_files) == 1:
            args.pkl_files = glob(args.pkl_files[0])
    args.pkl_files = [Path(f) for f in args.pkl_files]

    #-- Get phase files
    args.iff_dd_dir = Path(args.iff_dd_dir)
    pha_files = []
    for file in args.pkl_files:
        bname = "_".join(file.stem.split("_")[:4])
        if args.iff_dd_dir.is_absolute():
            pha_dir = args.iff_dd_dir
        else:
            pha_dir = file / args.iff_dd_dir
        pha_file = (pha_dir / f"{bname}.pha").resolve()
        if not pha_file.exists():
            print(f"No .pha file found at {pha_file}.", file=sys.stderr)
            sys.exit(1)
        pha_files.append(pha_file)
        
    #-- Check output
    args.output_dir = Path(args.output_dir)
    if not args.output_dir.is_absolute():
        args.output_dir = (args.pkl_files[0] / args.output_dir).resolve()
    args.output_dir.mkdir(exist_ok=True)
        
    #-- Get model parameters
    parent_name = args.pkl_files[0].parent.stem
    try: 
        setup = parent_name.split("_")
        arch = "_".join(setup[:2])
        ps = int(setup[2])
        form = setup[3]
        cs = int(setup[4])
    except:
        print(f"Unable to retrieve model parameters from name of directory: {parent_name}", file=sys.stderr)
        sys.exit(1)
        
    # TODO: CONTINUE FROM HERE!!

    #-- List files
    # patch_files = list(patch_dir.glob('*.pkl'))[:1]
    # attn_files = list(attn_dir.glob('*.pkl'))[:1]
    # pha_files = list(pha_dir.glob('*.pha'))[:1]

    for i, (pkl_file, pha_file) in enumerate(zip(args.pkl_files, pha_files)):
        name = pha_file.stem
        print(f"Evaluating {name}")
        #-- Load files
        img = utils.pad_image(utils.direct_phase_image(pha_file), ps)
        embed_data = import_embeddings(pkl_file)
        #-- Setup mask
        mask_object = semseg.SalientMask(img.size, patch_size=ps, accum=args.which_accumulation)
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
            
            #-- Configure output
            ofile = args.output_dir / f"{name}_{id_}.png"
            if ofile.exists():
                print("  File exists -> Skipping")
                continue
            
            #-- Get dimensions
            num_heads = qkv.shape[1]
            num_tokens = qkv.shape[2]
            head_dim = qkv.shape[3]
            embed_dim = num_heads * head_dim
            w_featmap = h_featmap = int((num_tokens-1) ** 0.5)
            dim_featmap = (w_featmap, h_featmap)
            
            #-- query, key, and value matrices
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q.transpose(0, 1).reshape(num_tokens, -1)
            k = k.transpose(0, 1).reshape(num_tokens, -1)
            v = v.transpose(0, 1).reshape(num_tokens, -1)

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
                print(f"{args.which_embeddings} not recognized as a valid input for which_embeddings.", file=sys.stderr)
                sys.exit(1)
            
            if 'dino' in args.which_method.lower():
                salient = semseg.DINO_segmentation(attn, num_heads, dim_featmap, args.dino_mass_frac)
                saliency_map.append(salient)
                
            elif 'pca' in args.which_method.lower():
                embeds.append(embed.reshape(*dim_featmap, embed_dim))
                
            elif 'corr' in args.which_method.lower():
                salient, _ = semseg.LOST_segmentation(embed, dim_featmap, (ps, ps), (3, cs, cs))
                saliency_map.append(salient)
            
            elif 'deg' in args.which_method.lower():
                _, salient = semseg.LOST_segmentation(embed, dim_featmap, (ps, ps), (3, cs, cs))
                saliency_map.append(salient)
            
            else:
                print(f"{args.which_method} not recognized as a valid input for which_method.", file=sys.stderr)
                sys.exit(1)
                
        if 'pca' in args.which_method.lower():
            saliency_map = semseg.consistent_PCA_segmentation(embeds)
            
        for (bbox, mask) in zip(bboxes, saliency_map):
            mask_object[bbox] = mask
        masked_img = mask_object.apply_to(img, args.threshold, args.sigma, args.min_patch, args.connectivity)
        resized_img = plot_utils.rescale_img(masked_img, 0.25)
        resized_img.show()        
        
        sys.exit(1)
        
        # import matplotlib.pyplot as plt
        # import skimage
        # fig, axes = plt.subplots(3,len(patches))
        # for i, (mask, salient, chunk) in enumerate(zip(masks, saliency_map, patches)):
        #     bbox = chunk['bbox']
        #     im = np.array(img.crop(bbox).convert('L'))
        #     axes[0,i].imshow(im, cmap='gray')
        #     axes[1,i].imshow(mask)
        #     axes[2,i].imshow(salient)
        #     plot_utils.remove_axis_elements(axes[0,i])
        #     plot_utils.remove_axis_elements(axes[1,i])
        #     plot_utils.remove_axis_elements(axes[2,i])
        # plot_utils.plot_maximized()
        
        import matplotlib.pyplot as plt
        import skimage
        fig, axes = plt.subplots(2,len(patches))
        for i, (salient, bbox) in enumerate(zip(saliency_map, bboxes)):
            im = np.array(img.crop(bbox).convert('L'))
            axes[0,i].imshow(im, cmap='gray')
            axes[1,i].imshow(salient)
            plot_utils.remove_axis_elements(axes[0,i])
            plot_utils.remove_axis_elements(axes[1,i])
        plot_utils.plot_maximized()

        masks = [to_rgba_array(mask_object.get_mask(bbox), vmin=mask_object.mask.min(), vmax=mask_object.mask.max(), colormap='viridis') for bbox in bboxes]
        norms = [to_rgba_array(mask_object.get_norm_mask(bbox), vmin=mask_object.normalized_mask.min(), vmax=mask_object.normalized_mask.max(), colormap='viridis') for bbox in bboxes]
        smooths = [to_rgba_array(mask_object.get_smooth_mask(bbox), vmin=mask_object.smoothed_mask.min(), vmax=mask_object.smoothed_mask.max(), colormap='viridis') for bbox in bboxes]
        binaries = [to_rgba_array(mask_object.get_binary_mask(bbox), vmin=mask_object.binary_mask.min(), vmax=mask_object.binary_mask.max()) for bbox in bboxes]
        filtered = [to_rgba_array(mask_object.get_filter_mask(bbox), vmin=mask_object.filtered_mask.min(), vmax=mask_object.filtered_mask.max()) for bbox in bboxes]
        images = [np.array(img.crop(bbox)) for bbox in bboxes]
        mask_images = [np.array(masked_img.crop(chunk['bbox'])) for chunk in patches]
        comb_img = grid_image(images, norms, filtered) 
        # comb_img2 = grid_image(norms, filtered, mask_images)
        comb_img.show()
        
        plt.figure()
        plt.imshow(mask_object.normalized_mask)
        plot_utils.plot_maximized()
                
    print("  All files completed")


def import_embeddings(fname: os.PathLike) -> npt.NDArray[np.void]:
    '''
    output dtype:
    name            type        dimension
    ---------------------------------------
    'id'            int32       scalar
    'bbox'          int32       (4,)
    'cls_embed'     float32     (embed_dim,)
    'patch_embed'   float32     (num_patches, embed_dim)
    'output_embed'  float32     (num_patches, output_dim)
    'qkv'           float32     (3, num_heads, num_tokens, head_dim)
    'attn'          float32     (num_heads, num_tokens, num_tokens)
    '''
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    assert isinstance(data, np.ndarray) and data.dtype.fields is not None, "Expected structured NumPy array"
    return data

# def upscale_mask(arr: npt.NDArray, scale: int) -> npt.NDArray:
#     return zoom(arr, zoom=(scale,scale), order=0)

# def rescale_img(img: Image, scale: float):
#     new_size = (int(img.width*scale), int(img.height*scale))
#     return img.resize(new_size, Image.LANCZOS)

# def pad_image(img, patch_size):
#     ps = patch_size
#     left = 0
#     top = 0
#     right = int(((ps - img.size[0]/16) % 1) * 16)
#     bottom = int(((ps - img.size[1]/16) % 1) * 16)
#     return ImageOps.expand(img, border=(left, top, right, bottom), fill = (0,0,0,0))

if __name__ == "__main__":
    TESTING = True
    #-- Get input
    parser = argparse.ArgumentParser('semantic_segmentation', description="Perform semantic segmentation.", parents=[get_args_parser()])
    if TESTING:
        print("TESTING!", file=sys.stderr)
        args = parser.parse_args((r"D:\dtu\speciale\Data\testData\PineIsland\track_065_descending\network_test\vit_base_16_phase_448\*_embed.pkl"
                                  " --iff_dd_dir ../../../iff_dd_rollingRef"
                                  " --which_method deg"
                                  # " --which_embeddings q"
                                  ).split())
    else: 
        args = parser.parse_args()
        
    main(args)