#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
image2lakefeats.py takes iff_dd phases and extracts CLS and patch feature vectors
(CLS and patch embeddings) for lake chunks usign the DINO foundational model [1]

:Authors
    NPKC / 24-04-2025 / creation / s203980@dtu.dk

:Todo
    Add more comments

:References
    [1] Mathilde Caron et. al. Emerging Properties in Self-Supervised Vision 
    Transformers https://arxiv.org/abs/2104.14294

:Note:
    Requires DINO [1]
"""


# =============================================================================
# Modules
# =============================================================================


#-- Utilities
import argparse
import sys
import os
import numpy as np
import datetime
import pickle
import pkg_resources
from pathlib import Path
from glob import glob
#-- PyTorch
import torch
from torchvision import transforms as pth_transforms
#-- My own
import SGLNet.Corefunc.utils as utils
import SGLNet.NN.classifier as classifier
import SGLNet.Corefunc.chunk_loader as chunk_loader
#-- from DINO [1]
import dino.utils as dino_utils
import dino.vision_transformer as vits

# =============================================================================
# Constants
# =============================================================================


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
    parser = argparse.ArgumentParser('image2lakefeats', add_help=False)
    
    parser.add_argument(
        "iff_dd_paths",
        nargs = '+',
        metavar = "/path/to/iff_dd/<iff_dd_interval>.pha",
        help = "/path/to/iff_dd_phases. Wildcards are possible.\n" \
               + "Note: .pha files are assumed to be in a single directory")
    parser.add_argument(
        '-o',
        '--output_dir',
        type = str,
        metavar = "Path/to/output_dir",
        help = "Path to output directory. It is created if it does not exist.")
    parser.add_argument(
        '--chunk_size',
        type = int,
        default = 224,
        metavar = "<int>",
        help = "Chunk size that each image is divided into. Default is 224")
    parser.add_argument(
        '--overlap',
        type = utils.bool_flag,
        default = True,
        metavar = "<bool>",
        help = "Whether to use 50%% overlapping chunks. Default is True")
    parser.add_argument(
        '--downscale_factor',
        type = int,
        default = 1,
        metavar = "<int>",
        choices= [1, 2, 4],
        help = "Downscaling factor. Default is 1.")
    parser.add_argument(
        '--form',
        type = str,
        default = 'recta',
        metavar = "<str>",
        choices = ['image', 'phase', 'recta', 'polar', 'phase2', 'polar2', 'recta2'],
        help = "Which form the phase should be in. Default is 'recta'.")
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 64,
        metavar = "<int>",
        help = "Batch size (in number of image chunks). Default is 64.")
    parser.add_argument(
        '--arch',
        type = str,
        default = 'vit_small',
        metavar = '<str>',
        choices = ['vit_small', 'vit_base'],
        help = "Name of architecture for inference. Default is 'vit_small'.")
    parser.add_argument(
        '--patch_size',
        type = int,
        default = 16,
        metavar = '<int>',
        choices = [8, 16],
        help = "Patch size to use for ViT. Default is 16.")
    parser.add_argument(
        '--n_last_blocks',
        type = int,
        default = 4,
        metavar = "<int>",
        help = "Concatenate [CLS] tokens for the 'n' last blocks in the ViT." \
               + "DINO recommends n=4 for ViT-Small and n=1 for ViT-Base." \
               + "Default is 4.")
    parser.add_argument(
        '--avgpool_patchtokens',
        type = utils.bool_flag,
        default = True,
        metavar = "<bool>",
        help = "Concatenate the global average pooled features to the [CLS] token." \
               + "DINO recommends False for ViT-Small and True for ViT-Base." \
               + "Default is True."
               )
    parser.add_argument(
        '--DINO_weights',
        type = str,
        default = '',
        metavar = "<str>",
        help = "Path to pretrained DINO weights file or directory to use for inference. " \
               + "Default is ''."
               )
    parser.add_argument(
        '--CLS_weights',
        type = str,
        default = '',
        metavar = "<str>",
        help = "Path to pretrained classification head weights (.pth.tar.file). " \
               + "Default is '', which means the script looks for a suitable " \
               + "weights-file in the SGLNet.Weights folder. If this fails, try " \
               + "to manually add the file path using this argument."
               )
    parser.add_argument(
        '--overwrite',
        type = utils.bool_flag,
        default = False,
        metavar = "<bool>",
        help = "Whether to recompute and overwrite features that already exist." \
               + "Default is False.")
    parser.add_argument(
        '--device',
        type = str,
        metavar = "<str>",
        default = 'cuda',
        choices = ['cuda', 'cpu'],
        help = "Specify whether to use cuda or cpu for training. Default is cuda.")
    parser.add_argument(
        '--segmentation_dir',
        type = str,
        default = '.',
        metavar = "<str>",
        help = "Path to directory containing mask file. Can be absolute or relative. " \
               + "Default is '.' (same dir as .pha file).\nNote: Mask filenames are " \
               + "expected to be the basename of corresponding .pha file and " \
               + "some custom extension given by --mask_extension. For example " \
               + "20160726_20160807_20160807_20160819_phase-01.png.")
    parser.add_argument(
        '--mask_extension',
        type = str,
        default = "_gt",
        metavar = "<str>",
        help = "Custom extension of filename when importing masks. This is to " \
               + "be used when mask files are names with the basename of the " \
               + ".pha files + mask_extension. For example " \
               + "20160726_20160807_20160807_20160819_phase-01.png. Default is " \
               + "'_phase-01'.")
    parser.add_argument(
        '--grounding_line_dir',
        type = str,
        default = '.',
        metavar = "<str>",
        help = "Path to directory containing grounding line mask files. Can be " \
               + "absolute or relative. Default is '.' (same dir as .pha file).\n" \
               + "Note: grounding line masks are expected to be names after the " \
               + "acquisition track, e.g. 'track_099_ascending.png' and thus be " \
               + "valid for all acquistions from this track.")
        
    return parser


def main(args: argparse.Namespace) -> None:
    """
    Main function for image2feat.py command. Takes .pha images and converts
    to feature vectors (embeddings) using DINO.

    Parameters
    ----------
    args : argparse.Namespace
        parsed terminal commands.

    Returns
    -------
    None.

    """         

    #-- Setup folders and files
    if isinstance(args.iff_dd_paths, str):
        args.iff_dd_paths = [args.iff_dd_paths]
    if os.name == "nt": # For windows
        if len(args.iff_dd_paths) == 1:
            args.iff_dd_paths = glob(args.iff_dd_paths[0])
    args.iff_dd_paths = [Path(f) for f in args.iff_dd_paths]
    num_files = len(args.iff_dd_paths)
    pha_dir = args.iff_dd_paths[0].parent
    
    args.output_dir = Path(str(args.output_dir)).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    segmentation_dir = Path(str(args.segmentation_dir))
    if not segmentation_dir.is_absolute():
        segmentation_dir = pha_dir / segmentation_dir
    segmentation_dir = segmentation_dir.resolve()
    if not segmentation_dir.exists():
        print(f"Mask directory {segmentation_dir} does not exist.", file=sys.stderr)
        sys.exit(1)
        
    grounding_line_dir = Path(str(args.grounding_line_dir))
    if not grounding_line_dir.is_absolute():
        grounding_line_dir = pha_dir / grounding_line_dir
    grounding_line_dir = grounding_line_dir.resolve()
    if not grounding_line_dir.exists():
        print(f"Grounding line directory {grounding_line_dir} does not exist.", file=sys.stderr)
        sys.exit(1)
        
    if args.DINO_weights != '':
        DINO_weights = Path(str(args.DINO_weights)).resolve()
        if DINO_weights.is_dir():
            DINO_weights = DINO_weights / utils.pretrained_weights_dict[args.arch][str(args.patch_size)]
        if not DINO_weights.is_file():
            print(f'No valid file found for pretrained weights {DINO_weights}.', file=sys.stderr)
            sys.exit(1)
    version = f"{args.arch.split('_')[1]}_{args.patch_size}_{args.form}_{args.chunk_size}"
    if args.CLS_weights == '':
        CLS_weights = Path(pkg_resources.resource_filename("SGLNet.Weights", f"{version}.pth.tar"))
    else:
        CLS_weights = Path(args.CLS_weights)
    if not CLS_weights.exists():
        print(f"Error: Failed to load CLS weights from {args.CLS_weights}.", file=sys.stderr)
        sys.exit(1)
    
    #-- Setup data transformer
    data_transform = pth_transforms.Compose([
        pth_transforms.Resize(args.chunk_size, interpolation=3),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    #-- Setup model
    model = vits.__dict__[args.arch](patch_size=args.patch_size)
    embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    dino_utils.load_pretrained_weights(model, DINO_weights, 'teacher', args.arch, args.patch_size)
    
    #-- Setup classifier model
    state_dict = torch.load(CLS_weights, map_location="cpu", weights_only=True)
    model_state_dict = state_dict["state_dict"]
    classifier_model = classifier.BinaryClassifier(embed_dim)
    classifier_model.load_state_dict(model_state_dict)
    classifier_model.to(device)
    classifier_model.eval()
    
    #-- Setup custom dtype for structured numpy array
    num_patches = (args.chunk_size//args.patch_size)**2
    chunk_dtype = custom_dtype(embed_dim, num_patches)

    #-- Run inference over files
    elapsed_time = datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d")
    print(f"({elapsed_time}) image2feat.py started", file=sys.stderr)
    start_string = (" " * (len(str(elapsed_time))+3)) + f"Looping over {num_files} files"
    print(start_string, file=sys.stderr)
    print("-" * len(start_string), file=sys.stderr)
    for i, file in enumerate(args.iff_dd_paths):
                
        fname = file.stem
        output_file = args.output_dir / f"{fname}.pkl"
        print(f"Evaluating ({i+1}/{num_files}): {fname}", file=sys.stderr)
        
        #-- Only do computations if features have not already been computed
        if output_file.exists():
            print("  .pkl file already exists", file=sys.stderr)
            if not args.overwrite:
                print(" Skipping file", file=sys.stderr)
                continue
            print("  Recomputing and overwriting file", file=sys.stderr)
        
        #-- Get chunk loader
        print("  Loading file", file=sys.stderr)
        Loader = chunk_loader.TensorChunkLoader(
            image_path = file, 
            transform = data_transform, 
            chunk_size = args.chunk_size, 
            overlap = args.overlap, 
            downscale_factor = args.downscale_factor, 
            form = args.form, 
            batch_size = args.batch_size)
        
        #-- Get mask
        # mask_file = (segmentation_dir / (Loader.IffDD.file_name + args.mask_extension)).with_suffix('.png')
        # gl_file = (grounding_line_dir / Loader.IffDD.track_dir.stem).with_suffix('.png')      
        # Loader.apply_masks(mask_file, gl_file, 0, 1)
        # Loader.bboxes = Loader.bboxes[Loader.is_lake]
        N = len(Loader)

        print("  Extracting features", file=sys.stderr)
        with torch.no_grad():
            cls_embedding = torch.zeros((N, model.embed_dim*args.n_last_blocks), dtype=torch.float32).to(device)
            patch_embedding_list = []
            predictions = torch.zeros((N, 1), dtype=torch.float32).to(args.device)
            n_batches = int(np.ceil(len(Loader)/args.batch_size))
            for i, batch in enumerate(Loader):
                torch.cuda.empty_cache()
                batch = batch.to(device)
                idx = Loader._interbatch_idx
                intermediate_output = model.get_intermediate_layers(batch, args.n_last_blocks)
                batch_predictions = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if args.avgpool_patchtokens:
                    batch_predictions = torch.cat((batch_predictions.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    batch_predictions = batch_predictions.reshape(batch_predictions.shape[0], -1)
                cls_embedding[idx] = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                batch_prediction = classifier_model(batch_predictions)
                predictions[idx] = batch_prediction
                patch_embedding_list.extend([x[:, 1:].cpu().numpy() for (x, p) in zip(intermediate_output, batch_prediction) if p >= 0])
                print(f"    Batch {i+1}/{n_batches}", file=sys.stderr)
            patch_embedding = np.array(patch_embedding_list)
            # patch_embedding_list.append(np.concatenate(, axis=-1))
            predictions = (predictions.cpu() >= 0).long().flatten().numpy().astype(bool)
            cls_embedding = cls_embedding.cpu().numpy()
            patch_embedding = np.vstack(patch_embedding_list)
        
        #-- Store output
        print("  Storing file", file=sys.stderr)
        chunk_data = np.empty(N, dtype=chunk_dtype)
        chunk_data['id'] = np.arange(N)[predictions]
        chunk_data['bbox'] = Loader.bboxes[predictions]
        chunk_data['patch_embed'] = patch_embedding[predictions]
        chunk_data['cls_embed'] = cls_embedding[predictions]
        with open(output_file, 'wb') as f:
            pickle.dump(chunk_data, f)
    
    elapsed_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"({elapsed_time}) Finished {i+1} / {num_files} succesfully!", file=sys.stderr)

def custom_dtype(embed_dim: int, num_patches: int):
    '''
    output dtype:
    [name]          [type]      [dimension]
    'id'            int32       scalar      
    'bbox'          int32       np.array((4,))   
    'patch_embed'   float32     np.array((num_patches, embed_dim))
    'cls_embed'     float32     np.array((embed_dim,))
    '''
    dtype = [
        ('id', np.int32), 
        ('bbox', (np.int32, 4)),  
        ('patch_embed', (np.float32, (num_patches, embed_dim))),
        ('cls_embed', (np.float32, embed_dim)) 
    ]
    return dtype

# =============================================================================
# Executable
# =============================================================================


if __name__ == '__main__':
    TESTING = True
    #-- Get input
    parser = argparse.ArgumentParser('image2lakefeats', description="Compute features and store as npy array.", parents=[get_args_parser()])
    if TESTING:
        print("TESTING!", file=sys.stderr)
        args = parser.parse_args(("D:\\dtu\\speciale\\Data\\testData\\PineIsland\\track_065_descending\\iff_dd_rollingRef\\*.pha "
                                  + "--DINO_weights D:\\dtu\\speciale\\Foundational_Models\\dino\\dino\\pretrained_weights "
                                  + "-o D:\\dtu\\speciale\\Vejledning\\Vejledning_8\\Segmentation\\Data "
                                  + "--segmentation_dir ../ground_truth "
                                  + "--grounding_line_dir D:\\dtu\\speciale\\Grounding_line_masks "
                                  + "--overwrite True "
                                  + "--arch vit_base "
                                  + "--n_last_blocks 1 "
                                  + "--form phase "
                                  + "--chunk_size 448 "
                                  + "--patch_size 16 "
                                  ).split())
    else: 
        args = parser.parse_args()
        
    main(args)