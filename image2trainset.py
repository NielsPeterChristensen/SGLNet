#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
image2trainset.py takes iff_dd phases and extracts feature vectors
(one CLS embedding per image) usign the DINO foundational model [1]

:Authors
    NPKC / 18-02-2025 / creation / s203980@dtu.dk

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
from pathlib import Path
from glob import glob
#-- PyTorch
import torch
from torchvision import transforms as pth_transforms
#-- My own
import SGLNet.Corefunc.utils as utils
import SGLNet.Corefunc.chunk_loader as chunk_loader
import SGLNet.Corefunc.chunk_masking as chunk_masking
#-- from DINO [1]
import SGLNet.dino.utils as dino_utils
import SGLNet.dino.vision_transformer as vits



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
    parser = argparse.ArgumentParser('image2trainset', add_help=False)
    
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
        default = '../../../../feature_dataset',
        metavar = "Path/to/output_dir",
        help = "Path to output directory. It is created if it does not exist." \
               + "Can be relative or absolute. Default location is '../features' relative " \
               + "to directory with .pha files.")
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
        default = False,
        metavar = "<bool>",
        help = "Concatenate the global average pooled features to the [CLS] token." \
               + "DINO recommends False for ViT-Small and True for ViT-Base." \
               + "Default is False.")
    parser.add_argument(
        '--pretrained_weights',
        type = str,
        default = '',
        metavar = "<str>",
        help = "Path to pretrained weights file or directory to use for inference. " \
               + "Default is ''.")
    parser.add_argument(
        '--overwrite',
        type = utils.bool_flag,
        default = False,
        metavar = "<bool>",
        help = "Whether to recompute and overwrite features that already exist." \
               + "Default is False.")
    parser.add_argument(
        '--ambiguous_chunk_action',
        type = chunk_masking.action_flag,
        default = -1,
        choices = ['Exclude', 'exclude', '-1', 'True', 'true', '1', 'False', 'false', '0'],
        metavar = "<str>",
        help = "Specify what to do with chunks containing 'ambiguous' mask pixels, " \
                + "i.e. weak or noisy lake patterns. Options are:\n" \
                + " -1 = exclude (exclude from training)\n" \
                + "  0 = False (label as nolake)\n" \
                + "  1 = True (label as lake).\n"\
                + "Default is -1 (exclude).")
    parser.add_argument(
        '--offcenter_chunk_action',
        type = chunk_masking.action_flag,
        default = -1,
        choices = ['Exclude', 'exclude', '-1', 'True', 'true', '1', 'False', 'false', '0'],
        metavar = "<str>",
        help = "Specify what to do with chunks containing 'lake' mask pixels " \
                + "only near the chunk edge (outside the centre 50%%). Options are:\n" \
                + " -1 = exclude (exclude from training)\n" \
                + "  0 = False (label as nolake)\n" \
                + "  1 = True (label as lake).\n"\
                + "Default is -1 (exclude).")
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
        default = "_phase-01",
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
    
    args.output_dir = Path(str(args.output_dir))
    if not args.output_dir.is_absolute():
        args.output_dir = pha_dir / args.output_dir / f"{args.arch[4:]}_{args.patch_size}_{args.form}_{args.chunk_size}"
    args.output_dir = args.output_dir.resolve()
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
        
    pretrained_weights = Path(str(args.pretrained_weights)).resolve()
    if str(pretrained_weights) != '':
        if pretrained_weights.is_dir():
            pretrained_weights = pretrained_weights / utils.pretrained_weights_dict[args.arch][str(args.patch_size)]
        if not pretrained_weights.is_file():
            print('No valid file found for pretrained weights {pretrained_weights}.', file=sys.stderr)
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
    dino_utils.load_pretrained_weights(model, pretrained_weights, 'teacher', args.arch, args.patch_size)
    
    #-- Setup output file
    setup = setup_output(args, data_transform, embed_dim)
    feature_data, feature_file, label_data, label_file, bbox_data, bbox_file, meta_file = setup

    #-- Run inference over files
    elapsed_time = datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d")
    print(f"({elapsed_time}) image2feat.py started", file=sys.stderr)
    start_string = (" " * (len(str(elapsed_time))+3)) + f"Looping over {num_files} files"
    print(start_string, file=sys.stderr)
    print("-" * len(start_string), file=sys.stderr)
    for i, file in enumerate(args.iff_dd_paths):
        
        with open(meta_file, 'rb') as f:
            metadata = pickle.load(f)
                
        #-- Only do computations if features have not already been computed
        if not file.stem in metadata.keys():
        
            #-- Get chunk loader
            Loader = chunk_loader.TensorChunkLoader(
                image_path = file, 
                transform = data_transform, 
                chunk_size = args.chunk_size, 
                overlap = args.overlap, 
                downscale_factor = args.downscale_factor, 
                form = args.form, 
                batch_size = args.batch_size)
                
            #-- Get mask
            mask_file = (segmentation_dir / (Loader.IffDD.file_name + args.mask_extension)).with_suffix('.png')
            gl_file = (grounding_line_dir / Loader.IffDD.track_dir.stem).with_suffix('.png')      
            Loader.apply_masks(mask_file, gl_file, args.ambiguous_chunk_action, args.offcenter_chunk_action)
    
            with torch.no_grad():
                output = torch.zeros((len(Loader), embed_dim), dtype=torch.float32).to(device)
                for batch in Loader:
                    torch.cuda.empty_cache()
                    batch = batch.to(device)
                    idx = Loader._interbatch_idx
                    if "vit" in args.arch:
                        intermediate_output = model.get_intermediate_layers(batch, args.n_last_blocks)
                        batch_output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                        if args.avgpool_patchtokens:
                            batch_output = torch.cat((batch_output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                            batch_output = batch_output.reshape(batch_output.shape[0], -1)
                    else:
                        batch_output = model(batch)
                    output[idx] = batch_output
                output = output.cpu().numpy()
                
            #-- Store outputs
            start_ind = int(list(metadata.values())[-1]['end_ind'])
            end_ind = start_ind + int(len(Loader))
            feature_data[start_ind:end_ind] = output
            label_data[start_ind:end_ind] = np.array(Loader.is_lake, dtype=np.float32).reshape((-1,1))
            bbox_data[start_ind:end_ind] = np.array(Loader.bboxes, dtype=np.float32).reshape((-1,4))
                    
            metadata[Loader.IffDD.file_name] = {'start_ind': start_ind, 'end_ind': end_ind}
            with open(meta_file, 'wb') as f:
                pickle.dump(metadata , f)
            
        if (i + 1) % 5 == 0:
            elapsed_time = datetime.datetime.now().strftime("%H:%M:%S")
            print(f'({elapsed_time})   {i+1} / {num_files} completed', file=sys.stderr)
    
    elapsed_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"({elapsed_time}) Finished {i+1} / {num_files} succesfully!", file=sys.stderr)
    final_length = int(list(metadata.values())[-1]['end_ind'])
    print(f"{' '*len(elapsed_time)} Truncating output length from {metadata['Setup']['total_length']} to {final_length}...", file=sys.stderr)
    metadata['Setup']['total_length'] = final_length
    with open(meta_file, 'wb') as f:
        pickle.dump(metadata , f)
    
    feature_data.flush()
    label_data.flush()
    bbox_data.flush()
    del feature_data
    del label_data
    del bbox_data
    
    final_feature_size = final_length * embed_dim * np.dtype(np.float32).itemsize
    final_label_size = final_length * 1 * np.dtype(np.float32).itemsize
    final_bbox_size = final_length * 4 * np.dtype(np.float32).itemsize
    with open(feature_file, "rb+") as f:
        f.truncate(final_feature_size)
    with open(label_file, "rb+") as f:
        f.truncate(final_label_size)
    with open(bbox_file, "rb+") as f:
        f.truncate(final_bbox_size)
    
    elapsed_time = datetime.datetime.now().strftime("%H:%M:%S")        
    print(f"({elapsed_time}) Truncation successful!", file=sys.stderr)
    
    
def does_lakefiles_exist(file: str, args: argparse.Namespace) -> bool:
    file = utils.remove_extensions(Path(str(file)).resolve())
    file_name = file.stem
    track_dir = file.parent.parent
    if args.output_dir == '':
        output_dir = track_dir / "features" / f"{args.arch[:4]}_{args.patch_size}_{args.form}_{args.chunk_size}"
    else:
        output_dir = Path(str(args.output_dir)).resolve()
        output_dir = output_dir / f"{args.arch}{args.patch_size}_{args.form}{args.chunk_size}"
    if (output_dir / "lake" / file_name).with_suffix('.npy').exists() and (output_dir / "nolake" / file_name).with_suffix('.npy').exists():
        print(f" * Skipping existing file {file_name}", file=sys.stderr)
        return True
    return False



def setup_output(args: argparse.Namespace, transform: pth_transforms.Compose, embed_dim: int) -> tuple[np.memmap, Path, np.memmap, Path, np.memmap, Path, Path]:
    
    file = args.iff_dd_paths[0]
    n_files = len(args.iff_dd_paths)
    
    Loader = chunk_loader.TensorChunkLoader(
        image_path = file, 
        transform = transform, 
        chunk_size = args.chunk_size, 
        overlap = args.overlap, 
        downscale_factor = args.downscale_factor, 
        form = args.form, 
        batch_size = args.batch_size)
    
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_file = (feature_dir / Loader.IffDD.track_dir.stem).with_suffix('.dat')
    label_dir = args.output_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_file = (label_dir / Loader.IffDD.track_dir.stem).with_suffix('.dat')
    meta_dir = args.output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_file = (meta_dir / Loader.IffDD.track_dir.stem).with_suffix('.pkl')
    bbox_dir = args.output_dir / "bboxes"
    bbox_dir.mkdir(parents=True, exist_ok=True)
    bbox_file = (bbox_dir / Loader.IffDD.track_dir.stem).with_suffix('.dat')
    
    total_length = len(Loader)*n_files
    features_shape = (total_length, embed_dim)
    labels_shape = (total_length, 1)
    bboxes_shape = (total_length, 4)
            
    if not feature_file.exists() or args.overwrite:
        features = np.memmap(feature_file, dtype=np.float32, mode='w+', shape=features_shape)
    else:
        features = np.memmap(feature_file, dtype=np.float32, mode='r+', shape=features_shape)
        
    if not label_file.exists() or args.overwrite:
        labels = np.memmap(label_file, dtype=np.float32, mode='w+', shape=labels_shape)
    else:
        labels = np.memmap(label_file, dtype=np.float32, mode='r+', shape=labels_shape)
        
    if not bbox_file.exists() or args.overwrite:
        bboxes = np.memmap(bbox_file, dtype=np.float32, mode="w+", shape=bboxes_shape)
    else:
        bboxes = np.memmap(bbox_file, dtype=np.float32, mode="r+", shape=bboxes_shape)
        
    if not meta_file.exists() or args.overwrite:
        with open(meta_file, 'wb') as f:
            pickle.dump({
                'Setup': {
                    'total_length': total_length, 
                    'embed_dim': embed_dim,
                    'n_files': n_files,
                    'track': Loader.IffDD.track_dir.stem,
                    'arch': args.arch,
                    'patch_size': args.patch_size,
                    'form': args.form,
                    'chunk_size': args.chunk_size
                },
                'Init': {
                    'start_ind': 0,
                    'end_ind': 0
                }
            }, f)
    
    return features, feature_file, labels, label_file, bboxes, bbox_file, meta_file
          
  
# =============================================================================
# Executable
# =============================================================================


if __name__ == '__main__':
    TESTING = False
    #-- Get input
    parser = argparse.ArgumentParser('image2trainset', description="Compute features and store as binary memmap array.", parents=[get_args_parser()])
    if TESTING:
        print("TESTING!", file=sys.stderr)
        args = parser.parse_args(("D:\\dtu\\speciale\\Data\\testData\\Jutulstraumen\\track_002_ascending\\iff_dd_rollingRef\\*.pha "
                                  + "--pretrained_weights D:\\dtu\\speciale\\Foundational_Models\\dino\\dino\\pretrained_weights "
                                  #+ "-o ../features_v3 "
                                  + "--segmentation_dir ../segmentation_v3 "
                                  + "--grounding_line_dir D:\\dtu\\speciale\\Grounding_line_masks "
                                  + "--overwrite False "
                                  + "--arch vit_small "
                                  + "--n_last_blocks 4 "
                                  + "--avgpool_patchtokens False "
                                  + "--form phase2 "
                                  + "--chunk_size 224 "
                                  + "--patch_size 8 "
                                  ).split())
    else: 
        args = parser.parse_args()
        
    main(args)