#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================


"""
test_network.py test network performance on (unseen) data with reference labels
(segmentations) for ground truth comparison. Consists of DINO foundational 
model [1] and binary linear classifier.

:Authors
    NPKC / 19-03-2025 / creation / s203980@dtu.dk

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
import datetime
import pickle
import json
import os
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from PIL import Image, ImageDraw
#-- PyTorch
import torch
import torch.nn as nn
import torchmetrics
import torchvision.transforms as pth_transforms
#-- My own
import SGLNet.Corefunc.chunk_loader as chunk_loader
import SGLNet.NN.classifier as classifier
import SGLNet.Corefunc.utils as utils
import SGLNet.Plotting.plot_utils as plot_utils
from SGLNet.image2trainset import pretrained_weights_dict
#-- from DINO [1]
import SGLNet.dino.vision_transformer as vits
import SGLNet.dino.utils as dino_utils


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
    parser = argparse.ArgumentParser('test_network', add_help=False)
    
    parser.add_argument(
        "iff_dd_paths",
        nargs = '+',
        metavar = "/path/to/iff_dd/<iff_dd_interval>.pha",
        help = "/path/to/iff_dd_phases. Wildcards are possible.\n" \
               + "Note: .pha files are assumed to be in a single directory")
    parser.add_argument(
        "--classifier_subdir",
        metavar = "<str>",
        required = True,
        help = "Path to classifier subdirectory from classifier head training " \
               + "output. Typically a directory named after the tracks used " \
               + "during training, e.g. classifier/002asc_099asc_038des_169des. " \
               + "This directory should contain a 'checkpoints' subdirectory " \
               + "with classifier checkpoints stored as 'epoch0.pth.tar', " \
               + "'epoch1,pth.tar' ... 'epoch[N].pth.tar'. This 'checkpoints' " \
               + "subdirectory must contain at least one such .pth.tar fiel. " \
               + "Note that by default, the grandparent of this folder is a " \
               + "directory names after the DINO configuration used, e.g. " \
               + "'base_16_phase_224. This directory name is used to set parameters" \
               + "if --arch --patch_size --form --chunk_size are not given (None)."
               + "Argument is required.")
    parser.add_argument(
        "-o",
        "--output_dir",
        type = str,
        metavar = "<str>",
        default = "./network_test",
        help = "Path to output directory to save performance metrics and images. " \
               + "Default is './network_test' relative to --classifier_subdir.")
    parser.add_argument(
        '--pretrained_weights',
        type = str,
        default = '',
        metavar = "<str>",
        help = "Path to pretrained DINO weights file or directory to use for inference. " \
               + "Default is ''.")
    parser.add_argument(
        "--batch_size",
        type = int,
        metavar = "<int>",
        default = 128,
        help = "Set the batch size to use for training. Default is 512.")
    parser.add_argument(
        "--num_workers",
        type = int,
        metavar = "<int>",
        default = 4,
        help = "Number of data loading workers. Default is 4.")
    parser.add_argument(
        '--epoch',
        type = str,
        default = '-1',
        metavar = "<str>",
        help = "Classifier head epoch to use. Default is -1 (last). Another " \
               + "option is 'best' with the log.txt file from training to " \
               + "determine the earliest epoch with the highest F1-score.")
    parser.add_argument(
        '--arch',
        type = str,
        default = None,
        metavar = '<str>',
        choices = ['vit_small', 'vit_base'],
        help = "Name of architecture for inference. Default is None " \
               + "(see --classifier_subdir description).")
    parser.add_argument(
        '--patch_size',
        type = int,
        default = None,
        metavar = '<int>',
        choices = [8, 16],
        help = "Patch size to use for ViT. Default is None " \
               + "(see --classifier_subdir description).")
    parser.add_argument(
        '--form',
        type = str,
        default = None,
        metavar = "<str>",
        choices = ['image', 'phase', 'recta', 'polar'],
        help = "Which form the phase should be in. Default is None " \
               + "(see --classifier_subdir description).")
    parser.add_argument(
        '--chunk_size',
        type = int,
        default = None,
        metavar = "<int>",
        help = "Chunk size that each image is divided into. Default is None " \
               + "(see --classifier_subdir description).")
    parser.add_argument(
        '--n_last_blocks',
        type = int,
        default = None,
        metavar = "<int>",
        help = "Concatenate [CLS] tokens for the 'n' last blocks in the ViT." \
               + "DINO recommends n=4 for ViT-Small and n=1 for ViT-Base." \
               + "Default is None (uses architecture to decide).")
    parser.add_argument(
        '--avgpool_patchtokens',
        type = utils.bool_flag,
        default = None,
        metavar = "<bool>",
        help = "Concatenate the global average pooled features to the [CLS] token." \
               + "DINO recommends False for ViT-Small and True for ViT-Base." \
               + "Default is None (uses architecture to decide).")
    parser.add_argument(
        '--downscale_factor',
        type = int,
        default = 1,
        metavar = "<int>",
        choices= [1, 2, 4],
        help = "Downscaling factor. Default is 1.")
    parser.add_argument(
        '--overlap',
        type = utils.bool_flag,
        default = True,
        metavar = "<bool>",
        help = "Whether to use 50%% overlapping chunks. Default is True")
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
    parser.add_argument(
        '--save_images',
        type = utils.bool_flag,
        default = False,
        metavar = "<bool>",
        help = "Specify whether to save each test image with corresponding " \
               + "lake detections. Default is True.")
    parser.add_argument(
        '--save_features',
        type = utils.bool_flag,
        default = True,
        metavar = "<bool>",
        help = "Specify whether to save feature vectors for each chunk in the " \
               + "test images. Default is True.")
    
    return parser


def main(args: argparse.Namespace) -> None:
        
    resolve_iff_paths(args)
    resolve_directory(args, 'segmentation_dir')
    resolve_directory(args, 'grounding_line_dir')
    resolve_classifier_subdir(args)
    resolve_checkpoint(args)
    resolve_output(args)
    infer_vit_params(args)
    resolve_vit_feature_config(args)
    validate_pretrained_weights(args)
    
    #-- Setup torch device
    args.device = torch.device(args.device)
    
    #-- Setup data transformer
    data_transform = pth_transforms.Compose([
        pth_transforms.Resize(args.chunk_size, interpolation=3),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    #-- Setup vit model
    vit_model = vits.__dict__[args.arch](patch_size=args.patch_size)
    args.embed_dim = vit_model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    vit_model.to(args.device)
    vit_model.eval()
    dino_utils.load_pretrained_weights(vit_model, args.pretrained_weights, 'teacher', args.arch, args.patch_size)

    #-- Setup classifier model
    state_dict = torch.load(args.checkpoint_file, map_location="cpu", weights_only=True)
    model_state_dict = state_dict["state_dict"]
    classifier_model = classifier.BinaryClassifier(args.embed_dim)
    classifier_model.load_state_dict(model_state_dict)
    classifier_model.to(args.device)
    classifier_model.eval()
    print("Model loaded with checkpoints!", file=sys.stderr)
    
    #-- Setup performance logger
    perf_logger = utils.PerformanceLogger(args.device)
    
    for i, file in enumerate(args.iff_dd_paths):
        print(f"Running inference for file ({i+1}/{len(args.iff_dd_paths)}): {file.stem}", file=sys.stderr)
        inference(file, vit_model, classifier_model, data_transform, perf_logger, args)
        
    performance_metrics = perf_logger.compute()
    print("Test complete!", file=sys.stderr)
    print(performance_metrics, file=sys.stderr)
    tracks = '_'.join([t.split('_')[1] for t in args.track_names])
    with (args.output_dir / f"performance_log__{tracks}.txt").open("w") as f:
        f.write(json.dumps(performance_metrics) + "\n")

def inference(file: Path, vit_model: vits.VisionTransformer, classifier_model: classifier.BinaryClassifier, transform: pth_transforms.Compose, logger: utils.PerformanceLogger, args: argparse.Namespace):
    
    Loader = chunk_loader.TensorChunkLoader(
        image_path = file,
        transform = transform,
        chunk_size = args.chunk_size,
        overlap = args.overlap,
        form = args.form, 
        downscale_factor = args.downscale_factor,
        batch_size = args.batch_size,
    )
    
    with torch.no_grad():
        features = torch.zeros((len(Loader), args.embed_dim), dtype=torch.float32).to(args.device)
        predictions = torch.zeros((len(Loader), 1), dtype=torch.float32).to(args.device)
        n_batches = int(np.ceil(len(Loader)/args.batch_size))
        for i, batch in enumerate(Loader):
            torch.cuda.empty_cache()
            batch = batch.to(args.device)
            idx = Loader._interbatch_idx
            #-- Extract features
            if "vit" in args.arch:
                intermediate_predictions = vit_model.get_intermediate_layers(batch, args.n_last_blocks)
                batch_predictions = torch.cat([x[:, 0] for x in intermediate_predictions], dim=-1)
                if args.avgpool_patchtokens:
                    batch_predictions = torch.cat((batch_predictions.unsqueeze(-1), torch.mean(intermediate_predictions[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    batch_predictions = batch_predictions.reshape(batch_predictions.shape[0], -1)
            else:
                batch_predictions = vit_model(batch)
            features[idx] = batch_predictions
            #-- Classify features
            predictions[idx] = classifier_model(batch_predictions)
            # print(f" - Batch {i+1}/{n_batches}", file=sys.stderr)
        features = features.cpu()
        predictions = (predictions.cpu() >= 0).long().flatten()
    
    apply_loader_mask(args, Loader)
    logger.update(predictions[Loader.valid_chunks], torch.tensor(Loader.is_lake))        
    if args.save_images:
        save_Image(args, predictions, Loader)
    if args.save_features:
        save_features(args, features, predictions, Loader)
    
def resolve_iff_paths(args: argparse.Namespace) -> None:
    """Resolves iff_dd_paths and sets pha_dir."""
    if os.name == "nt":
        paths = []
        for path in args.iff_dd_paths:
            paths.extend(glob(path))
        args.iff_dd_paths = paths
    args.iff_dd_paths = [Path(f) for f in args.iff_dd_paths]
    pha_dirs = [p.parent for p in args.iff_dd_paths]
    args.pha_dirs = list(dict.fromkeys(pha_dirs))
    args.track_names = [p.parent.stem for p in args.pha_dirs]

def resolve_directory(args: argparse.Namespace, attr_name: str) -> None:
    """Resolves and validates a directory path relative to pha_dir if necessary."""
    path = Path(str(getattr(args, attr_name)))
    if not path.is_absolute():
        path = [pha_dir / path for pha_dir in args.pha_dirs]
    if not isinstance(path, list):
        path = [path]
    path = [p.resolve() for p in path]
    if not all([p.exists() for p in path]):
        print(f"Directory {path} does not exist.", file=sys.stderr)
        sys.exit(1)
    
    setattr(args, attr_name, path)  # Update the args attribute

def resolve_classifier_subdir(args: argparse.Namespace) -> None:
    """Ensures classifier_subdir is an absolute path and exists."""
    args.classifier_subdir = Path(args.classifier_subdir).resolve()
    if not args.classifier_subdir.exists():
        print(f"Classifier model directory {args.classifier_subdir} does not exist.", file=sys.stderr)
        sys.exit(1)
        
def resolve_checkpoint(args: argparse.Namespace) -> None:
    """Finds the appropriate classifier checkpoint file."""
    checkpoint_files = np.array([f for f in (args.classifier_subdir / 'checkpoints').iterdir() if str(f).endswith('.pth.tar')])
    checkpoint_epochs = np.array([int(utils.remove_extensions(f).stem[5:]) for f in checkpoint_files])
    inds = checkpoint_epochs.argsort()
    checkpoint_files = checkpoint_files[inds]
    checkpoint_epochs = checkpoint_epochs[inds]
    if args.epoch.lower() == 'best':
        get_best_f1_epoch(args)
    args.epoch = int(args.epoch)
    if args.epoch < 0:
        args.epoch = checkpoint_epochs[args.epoch]
    if args.epoch in checkpoint_epochs:
        args.checkpoint_file = checkpoint_files[checkpoint_epochs == args.epoch][0]
    else:
        print(f"No checkpoint found matching epoch {args.epoch}.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Using checkpoint from epoch {args.epoch}.", file=sys.stderr)
    
def get_best_f1_epoch(args: argparse.Namespace) -> int: 
    
    def read_log(file: str) -> pd.DataFrame:
        data = []
        with open(file, "r") as f:
            for line in f:
                try:
                    timestamp, json_str = line.strip().split(": ", 1)  # Split on first ": "
                    entry = json.loads(json_str)  # Parse JSON
                    data.append(entry)
                except (ValueError, json.JSONDecodeError):
                    continue  # Skip non-JSON lines
        return pd.DataFrame(data)
    
    path = args.classifier_subdir / 'log.txt'
    data = read_log(path)
    best_f1_index = np.argmax(data['f1'] == max(data['f1']))
    args.epoch = int(data['epoch'][best_f1_index])
        
def resolve_output(args: argparse.Namespace) -> None:
    """Resolves and validates a directory path relative to pha_dir if necessary."""
    path = Path(args.output_dir)
    if not path.is_absolute():
        path = args.classifier_subdir / path
    path = path / f"epoch{args.epoch}"
    args.output_dir = path.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

def infer_vit_params(args: argparse.Namespace) -> None:
    """Determines ViT parameters based on arguments or directory structure."""
    if all([args.arch, args.patch_size, args.form, args.chunk_size]):
        return
    
    print('Using grandparent of --classifier_subdir to determine ViT parameters.', file=sys.stderr)
    model_dir = args.classifier_subdir.parent.parent
    arch, patch_size, form, chunk_size = model_dir.stem.split('_')
    args.arch = args.arch or f'vit_{arch}'
    args.patch_size = args.patch_size or int(patch_size)
    args.form = args.form or form
    args.chunk_size = args.chunk_size or int(chunk_size)

def resolve_vit_feature_config(args: argparse.Namespace) -> None:
    """Determines the ViT feature vector configuration."""
    if args.n_last_blocks is not None and args.avgpool_patchtokens is not None:
        return
    
    print('Using ViT model to set output feature configuration.', file=sys.stderr)
    if args.arch == 'vit_small':
        args.n_last_blocks = 4
        args.avgpool_patchtokens = False
    elif args.arch == 'vit_base':
        args.n_last_blocks = 1
        args.avgpool_patchtokens = True
    else:
        print(f'ViT architecture {args.arch} not recognized. Please set --n_last_blocks and --avgpool_patchtokens manually.', file=sys.stderr)
        sys.exit(1)

def validate_pretrained_weights(args: argparse.Namespace) -> None:
    """Validates and resolves the pretrained weights path."""
    if args.pretrained_weights != '':
        args.pretrained_weights = Path(str(args.pretrained_weights)).resolve()
        if args.pretrained_weights.is_dir():
            args.pretrained_weights = args.pretrained_weights / pretrained_weights_dict[args.arch][str(args.patch_size)]
        
        if not args.pretrained_weights.is_file():
            print(f'No valid file found for pretrained weights {args.pretrained_weights}.', file=sys.stderr)
            sys.exit(1)
    
def apply_loader_mask(args: argparse.Namespace, Loader: chunk_loader.ImageChunkLoader) -> None:
    "Apply manual segmentation masks for lakes and grounding lines to Loader."
    ind = np.array(args.track_names) == Loader.IffDD.track_dir.stem
    segmentation_dir = np.array(args.segmentation_dir)
    if len(segmentation_dir) == 1:
        segmentation_dir = segmentation_dir
    elif len(segmentation_dir) == len(ind):
        segmentation_dir = segmentation_dir[ind]
    else:
        print(f"Unable to determine correct segmentation_dir from {segmentation_dir}.", file=sys.stderr)
    grounding_line_dir = np.array(args.grounding_line_dir)
    if len(grounding_line_dir) == 1:
        grounding_line_dir = grounding_line_dir
    elif len(grounding_line_dir) == len(ind):
        grounding_line_dir = grounding_line_dir[ind]
    else:
        print(f"Unable to determine correct grounding_line_dir from {grounding_line_dir}.", file=sys.stderr)
        
    
    mask_file = (segmentation_dir[0] / (Loader.IffDD.file_name + args.mask_extension)).with_suffix('.png')
    gl_file = (args.grounding_line_dir[0] / Loader.IffDD.track_dir.stem).with_suffix('.png')      
    Loader.apply_masks(mask_file, gl_file)
    
def save_Image(args: argparse.Namespace, predictions: torch.Tensor, Loader: chunk_loader.ImageChunkLoader) -> None:
    """Save image with predicted lake outline drawn on."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    predictions = np.array(predictions, dtype=bool).reshape((-1,))
    Img = Loader()
    detected_bboxes = Loader.all_bboxes[predictions]
    outlines, mask = utils.get_outlines(detected_bboxes, Img.size)    
    plot_utils.draw_outlines(Img, outlines, color='red', linewidth=10)
    #-- Store Image
    output_dir = args.output_dir / Loader.IffDD.track_dir.stem / 'png'
    output_dir.mkdir(parents=True, exist_ok=True)
    Img.save((output_dir / Loader.IffDD.file_name).with_suffix('.png'))
    
def save_features(args: argparse.Namespace, features: torch.Tensor, predictions: torch.Tensor, Loader: chunk_loader.ImageChunkLoader) -> None:
    """Store feature vectors and corresponding bboxes as .pt file"""
    data_package = {
        "features": features,
        "predictions": predictions,
        "bboxes": Loader.all_bboxes,
        "lake_mask": str(Loader.path_to_lake_mask),
        "gl_mask": str(Loader.path_to_gl_mask),
        "iff_dd": str(Loader.IffDD.file_path),
        "test_tracks": args.track_names, 
        "params": {
            "chunk_size": Loader.chunk_size,
            "overlap": Loader.overlap,
            "downscale_factor": Loader.downscale_factor,
            "form": Loader.form,
            "stride": Loader.stride,
            "shape": Loader.shape
        }
    }
    output_dir = args.output_dir / Loader.IffDD.track_dir.stem / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (output_dir / Loader.IffDD.file_name).with_suffix('.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(data_package, f)
    

# =============================================================================
# Executable
# =============================================================================


if __name__ == '__main__':
    TESTING = False
    #-- Get input
    parser = argparse.ArgumentParser('test_network', description="Compute features and store as npy array.", parents=[get_args_parser()])
    if TESTING:
        print("TESTING!", file=sys.stderr)
        args = parser.parse_args(("D:\\dtu\\speciale\\Data\\testData\\PineIsland\\track_065_descending\\iff_dd_rollingRef\\*.pha "
                                  + "D:\\dtu\\speciale\\Data\\testData\\Cook\\track_010_ascending\\iff_dd_rollingRef\\*.pha "
                                  + "--classifier_subdir D:\\dtu\\speciale\\Data\\feature_dataset\\base_16_phase_224\\classifier\\002asc_099asc_038des_169des "
                                  + "--pretrained_weights D:\\dtu\\speciale\\Foundational_Models\\dino\\dino\\pretrained_weights "
                                  + "--segmentation_dir ../segmentation_v3 "
                                  + "--epoch best "
                                  + "--grounding_line_dir D:\\dtu\\speciale\\Grounding_line_masks "
                                  + "--save_images False "
                                  + "--save_features True"
                                  ).split())
    else: 
        args = parser.parse_args()
        
    main(args)