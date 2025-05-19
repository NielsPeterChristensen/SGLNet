#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================


"""
infer_test_data.py extracts network output for performance testing. Inference is 
performed on (unseen) data with reference labels
(segmentations) for ground truth comparison. Consists of DINO foundational 
model [1] and binary linear classifier.

:Authors
    NPKC / 30-04-2025 / creation / s203980@dtu.dk

:Todo
    Add more comments

:References
    [1] Mathilde Caron et. al. Emerging Properties in Self-Supervised Vision 
    Transformers https://arxiv.org/abs/2104.14294

:Note:
    * Requires DINO [1].
    * This is an updated version that uses both the training segmentation (where
    ambiguous areas are excluded) and a seperate reference segmentation with
    more careful (bul also uncertain) segmentations and not areas are excluded.
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
import numpy.typing as npt
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
import SGLNet.Corefunc.chunk_masking as chunk_masking
import SGLNet.NN.classifier as classifier
import SGLNet.Corefunc.utils as utils
import SGLNet.Plotting.plot_utils as plot_utils
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
    parser = argparse.ArgumentParser('infer_test_data', add_help=False)
    
    parser.add_argument(
        "iff_dd_paths",
        nargs = '+',
        metavar = "/path/to/iff_dd/<iff_dd_interval>.pha",
        help = "/path/to/iff_dd_phases. Wildcards are possible.\n")
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
        default = "../../network_test",
        help = "Path to output directory to save performance metrics and images. " \
               + "Default is '../../network_test' relative to iff_dd files.")
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
        default = 64,
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
        '--train_segmentation_dir',
        type = str,
        default = '.',
        metavar = "<str>",
        help = "Path to directory containing training mask file. Can be absolute or relative. " \
               + "Default is '.' (same dir as .pha file).\nNote: Mask filenames are " \
               + "expected to be the basename of corresponding .pha file " \
               + "and potentially some arbitrary extra string. For example " \
               + "20160726_20160807_20160807_20160819_phase-01.png.")
    
    parser.add_argument(
        '--test_segmentation_dir',
        type = str,
        default = '.',
        metavar = "<str>",
        help = "Path to directory containing testing mask file. Can be absolute or relative. " \
               + "Default is '.' (same dir as .pha file).\nNote: Mask filenames are " \
               + "expected to be the basename of corresponding .pha file " \
               + "and potentially some arbitrary extra string. For example " \
               + "20160726_20160807_20160807_20160819_phase-01.png.")
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
        '--save_pred',
        type = utils.bool_flag,
        default = True,
        metavar = "<bool>",
        help = "Specify whether to save predictions with corresponding" \
               + "train grount truth (gt) and test gt labels. Default is True.")
    parser.add_argument(
        '--save_embed',
        type = utils.bool_flag,
        default = True,
        metavar = "<bool>",
        help = "Specify whether to save embeddings (inkl attn) with corresponding" \
               + "for downstream segmentation testing etc. Default is True.")
    parser.add_argument(
        '--save_image',
        type = utils.bool_flag,
        default = False,
        metavar = "<bool>",
        help = "Specify whether to save each test image with corresponding " \
               + "lake detections. Default is False.")
    parser.add_argument(
        '--overwrite',
        type = utils.bool_flag,
        default = False,
        metavar = "<bool>",
        help = "Specify whether to overwrite existing data and images." \
               + "Default is False.")
    
    return parser


def main(args: argparse.Namespace) -> None:
        
    resolve_iff_paths(args)
    resolve_directory(args, 'train_segmentation_dir')
    resolve_directory(args, 'test_segmentation_dir')
    resolve_directory(args, 'grounding_line_dir')
    resolve_classifier_subdir(args)
    resolve_checkpoint(args)
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
    vit_model.to(args.device)
    vit_model.eval()
    dino_utils.load_pretrained_weights(vit_model, args.pretrained_weights, 'teacher', args.arch, args.patch_size)
    
    #-- Get relevant parameters
    args.embed_dim = vit_model.embed_dim
    args.output_dim = vit_model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    args.num_heads = vit_model.blocks[-1].attn.num_heads
    args.num_patches = (args.chunk_size // args.patch_size)**2
    args.num_tokens = args.num_patches + 1
    args.head_dim = args.embed_dim // args.num_heads
    
    #-- Setup custom dtype for structured numpy array
    args.embed_dtype = embedding_dtype(args)
    args.pred_dtype = prediction_dtype()

    #-- Setup classifier model
    state_dict = torch.load(args.checkpoint_file, map_location="cpu", weights_only=True)
    model_state_dict = state_dict["state_dict"]
    classifier_model = classifier.BinaryClassifier(args.output_dim)
    classifier_model.load_state_dict(model_state_dict)
    classifier_model.to(args.device)
    classifier_model.eval()
    print("Model loaded with checkpoints!", file=sys.stderr)
    
    #-- Setup performance logger
    # perf_logger = utils.PerformanceLogger(args.device)
    
    for i, file in enumerate(args.iff_dd_paths):
        print(f"Running inference for file ({i+1}/{len(args.iff_dd_paths)}): {file.stem}", file=sys.stderr)
        inference(file, vit_model, classifier_model, data_transform, args)
        
    # performance_metrics = perf_logger.compute()
    print("Test complete!", file=sys.stderr)
    # print(performance_metrics, file=sys.stderr)
    # tracks = '_'.join([t.split('_')[1] for t in args.track_names])
    # with (args.output_dir / f"performance_log__{tracks}.txt").open("w") as f:
    #     f.write(json.dumps(performance_metrics) + "\n")

def inference(file: Path, vit_model: vits.VisionTransformer, classifier_model: classifier.BinaryClassifier, transform: pth_transforms.Compose, args: argparse.Namespace):
    
    #-- Skip file if (relevant) outputs already exist
    if not args.overwrite:
        output_exists = []
        output_bname = get_output_basename(args, file)
        if args.save_embed:
            if Path(f"{output_bname}_embed.pkl").exists():
                output_exists.append(True)
            else:
                output_exists.append(False)
        if args.save_pred:
            if Path(f"{output_bname}_pred.pkl").exists():
                output_exists.append(True)
            else:
                output_exists.append(False)
        if args.save_image:
            if Path(f"{output_bname}_img.png").exists():
                output_exists.append(True)
            else:
                output_exists.append(False)
        if all(output_exists):
            print(" - Skipping", file=sys.stderr)
            return
    
    Loader = chunk_loader.TensorChunkLoader(
        image_path = file,
        transform = transform,
        chunk_size = args.chunk_size,
        overlap = args.overlap,
        form = args.form, 
        downscale_factor = args.downscale_factor,
        batch_size = args.batch_size,
    )
    
    num_chunks = len(Loader)
    num_batches = int(np.ceil(num_chunks/args.batch_size))
    chunk_id = np.arange(num_chunks)
    bboxes = Loader.bboxes
    embedding_data = [np.zeros(0, dtype=args.embed_dtype)]
    
    apply_loader_mask(args, Loader, 'train')
    train_gt = Loader.all_lake.astype(np.int32)
    train_gt[~Loader.valid_chunks] = -1
    apply_loader_mask(args, Loader, 'test0')
    test_gt_halfwindow = Loader.all_lake.astype(np.int32)
    test_gt_halfwindow[~Loader.valid_chunks] = -1
    apply_loader_mask(args, Loader, 'test1')
    test_gt_fullwindow = Loader.all_lake.astype(np.int32)
    test_gt_fullwindow[~Loader.valid_chunks] = -1
    Loader.bboxes = Loader.all_bboxes
        
    with torch.no_grad():
        predictions = np.zeros((num_chunks,), dtype=np.int32)
        for i, batch in enumerate(Loader):
            torch.cuda.empty_cache()
            batch = batch.to(args.device)
            idx = np.array(Loader._interbatch_idx)
            batch_size = len(idx)
            
            #-- Extract features
            intermediate_predictions = vit_model.get_intermediate_layers(batch, args.n_last_blocks)
            patch_tokens = intermediate_predictions[-1][:, 1:]
            cls_tokens = intermediate_predictions[-1][:, 0]
            output_embed = torch.cat([x[:, 0] for x in intermediate_predictions], dim=-1)
            if args.avgpool_patchtokens:
                output_embed = torch.cat((output_embed.unsqueeze(-1), torch.mean(patch_tokens, dim=1).unsqueeze(-1)), dim=-1)
                output_embed = output_embed.reshape(output_embed.shape[0], -1)
            #-- Classify features
            batch_pred = classifier_model(output_embed).flatten().long().cpu().numpy() >= 0
            predictions[idx] = batch_pred
            
            if any(batch_pred):
                #-- Exclude chunks with grounding lines in saved embedding data
                has_grounding_line = chunk_masking.does_mask_overlap_chunk(bboxes[idx], Loader.gl_mask)
                valid_batch_pred = batch_pred & ~has_grounding_line
                
                if any(valid_batch_pred):
                
                    num_true = sum(valid_batch_pred)
                    sub_idx = idx[valid_batch_pred]   
                    
                    #-- Store the outputs of qkv layer from the last attention layer
                    feat_out = {}
                    def hook_fn_forward_qkv(module, input, output):
                        feat_out["qkv"] = output
                    vit_model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
                    
                    #-- Forward pass in the model
                    attn = vit_model.get_last_selfattention(batch[valid_batch_pred])
                    
                    # Extract the qkv features of the last attention layer
                    qkv = (
                        feat_out["qkv"]
                        .reshape(num_true, args.num_tokens, 3, args.num_heads, args.head_dim)
                        .permute(2, 0, 3, 1, 4).transpose(0,1)
                    )
                    
                    batch_sample = embedding_sample(
                        args, 
                        chunk_id[sub_idx], 
                        bboxes[sub_idx], 
                        cls_tokens[valid_batch_pred].cpu().numpy(),
                        patch_tokens[valid_batch_pred].cpu().numpy(), 
                        qkv.cpu().numpy(), 
                        attn.cpu().numpy())
                    embedding_data.append(batch_sample)
            
            print(f" - Batch {i+1}/{num_batches}", file=sys.stderr)

    
    embedding_data = np.concatenate(embedding_data, axis=0)    
    prediction_data = pred_data(args, chunk_id, bboxes, predictions, train_gt, test_gt_halfwindow, test_gt_fullwindow)
    
    # var_gt = np.zeros((num_chunks, 10), dtype=np.int32)
    # mask = Loader.lake_mask == 255
    # for i in range(10):
    #     true_idx = chunk_masking.does_mask_overlap_chunk_center_frac(Loader.all_bboxes, mask, center_frac=(i+1)*0.1)
    #     var_gt[true_idx,i] = 1
    # additional_gt = np.zeros(num_chunks, dtype=[('var_gt', (np.int32, 10))])
    # additional_gt['var_gt'] = var_gt
    # with open(f"{output_bname}_additional.pkl", "wb") as f:
    #     pickle.dump(additional_gt, f)
    
    if args.save_embed:
        with open(f"{output_bname}_embed.pkl", 'wb') as f:
            pickle.dump(embedding_data, f)
    
    if args.save_pred:
        with open(f"{output_bname}_pred.pkl", 'wb') as f:
            pickle.dump(prediction_data, f)
    
    if args.save_image:
        save_Image(f"{output_bname}_img.png", predictions, Loader)
    
    
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
        
def get_output_basename(args: argparse.Namespace, current_fpath) -> None:
    """Resolves and validates a directory path relative to pha_dir if necessary."""
    current_fpath = Path(str(current_fpath))
    path = Path(args.output_dir)
    if not path.is_absolute():
        path = current_fpath / path
    path = path / f"{args.arch}_{args.patch_size}_{args.form}_{args.chunk_size}"
    output_dir = path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / current_fpath.stem

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
            args.pretrained_weights = args.pretrained_weights / utils.pretrained_weights_dict[args.arch][str(args.patch_size)]
        
        if not args.pretrained_weights.is_file():
            print(f'No valid file found for pretrained weights {args.pretrained_weights}.', file=sys.stderr)
            sys.exit(1)
    
def apply_loader_mask(args: argparse.Namespace, Loader: chunk_loader.ImageChunkLoader, which_mask: str) -> None:
    "Apply manual segmentation masks for lakes and grounding lines to Loader."
    ind = np.array(args.track_names) == Loader.IffDD.track_dir.stem
    if which_mask == 'train':
        segmentation_dir = np.array(args.train_segmentation_dir)
        ambiguous_chunk_action = -1
        offcenter_chunk_action = -1
    elif which_mask == 'test0':
        segmentation_dir = np.array(args.test_segmentation_dir)
        ambiguous_chunk_action = 0
        offcenter_chunk_action = 0
    elif which_mask == 'test1':
        segmentation_dir = np.array(args.test_segmentation_dir)
        ambiguous_chunk_action = 0
        offcenter_chunk_action = 1
    else:
        print(f"Invalid argument {which_mask} for which_mask")
        sys.exit(1)
        
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
    
    mask_file = list(segmentation_dir[0].glob(Loader.IffDD.file_name + '*.png'))[0]
    gl_file = (args.grounding_line_dir[0] / Loader.IffDD.track_dir.stem).with_suffix('.png')      
    Loader.apply_masks(mask_file, gl_file, ambiguous_chunk_action, offcenter_chunk_action)
    
def save_Image(fname, predictions: torch.Tensor, Loader: chunk_loader.ImageChunkLoader) -> None:
    """Save image with predicted lake outline drawn on."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    predictions = np.array(predictions, dtype=bool).reshape((-1,))
    Img = Loader()
    detected_bboxes = Loader.all_bboxes[predictions]
    outlines, mask = utils.get_outlines(detected_bboxes, Img.size)    
    plot_utils.draw_outlines(Img, outlines, color='red', linewidth=10)
    Img.save(fname)
    
def embedding_sample(args: argparse.Namespace, id_, bboxes, cls_embed, patch_embed, qkv, attn) -> npt.NDArray[np.void]:
    assert len(id_) == len(bboxes) == len(cls_embed) == len(patch_embed) == len(qkv) == len(attn), "Unequal number of elements in batch"
    bsize = len(id_)
    batch = np.zeros(bsize, dtype=args.embed_dtype)
    #-- Assign
    batch['id'] = id_.astype(np.int32)
    batch['bbox'] = bboxes.astype(np.int32)
    batch['cls_embed'] = cls_embed.astype(np.float32)
    batch['patch_embed'] = patch_embed.astype(np.float32)
    batch['qkv'] = qkv.astype(np.float32)
    batch['attn'] = attn.astype(np.float32)
    return batch    

def pred_data(args: argparse.Namespace, id_, bboxes, pred, train_gt, test_gt_halfwindow, test_gt_fullwindow) -> npt.NDArray[np.void]:
    assert len(id_) == len(bboxes) == len(pred) == len(train_gt) == len(test_gt_halfwindow) == len(test_gt_fullwindow), "Unequal number of elements in set"
    size = len(id_)
    data = np.zeros(size, dtype=args.pred_dtype)
    #-- Assign
    data['id'] = id_.astype(np.int32)
    data['bbox'] = bboxes.astype(np.int32)
    data['prediction'] = pred.astype(np.int32)
    data['train_gt'] = train_gt.astype(np.int32)
    data['test_gt_half'] = test_gt_halfwindow.astype(np.int32)
    data['test_gt_full'] = test_gt_fullwindow.astype(np.int32)
    return data
    
def embedding_dtype(args: argparse.Namespace):
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
    dtype = [
        ('id', np.int32),
        ('bbox', (np.int32, 4)),
        ('cls_embed', (np.float32, args.embed_dim)),
        ('patch_embed', (np.float32, (args.num_patches, args.embed_dim))),
        ('output_embed', (np.float32, (args.num_patches, args.output_dim))),
        ('qkv', (np.float32, (3, args.num_heads, args.num_tokens, args.head_dim))),
        ('attn', (np.float32, (args.num_heads, args.num_tokens, args.num_tokens)))
    ]
    return dtype

def prediction_dtype():
    '''
    output dtype:
    name            type        dimension
    ---------------------------------------
    'id'            int32       scalar
    'bbox'          int32       (4,)
    'prediction'    int32       scalar
    'train_gt'      int32       scalar
    'test_gt_half'  int32       scalar
    'test_gt_full'  int32       scalar
    '''
    dtype = [
        ('id', np.int32),
        ('bbox', (np.int32, 4)),
        ('prediction', np.int32),
        ('train_gt', np.int32),
        ('test_gt_half', np.int32),
        ('test_gt_full', np.int32)
    ]
    return dtype

# def save_features_old(args: argparse.Namespace, features: torch.Tensor, predictions: torch.Tensor, Loader: chunk_loader.ImageChunkLoader) -> None:
#     """Store feature vectors and corresponding bboxes as .pt file"""
#     data_package = {
#         "features": features,
#         "predictions": predictions,
#         "bboxes": Loader.all_bboxes,
#         "lake_mask": str(Loader.path_to_lake_mask),
#         "gl_mask": str(Loader.path_to_gl_mask),
#         "iff_dd": str(Loader.IffDD.file_path),
#         "test_tracks": args.track_names, 
#         "params": {
#             "chunk_size": Loader.chunk_size,
#             "overlap": Loader.overlap,
#             "downscale_factor": Loader.downscale_factor,
#             "form": Loader.form,
#             "stride": Loader.stride,
#             "shape": Loader.shape
#         }
#     }
#     output_dir = args.output_dir / Loader.IffDD.track_dir.stem / 'data'
#     output_dir.mkdir(parents=True, exist_ok=True)
#     output_path = (output_dir / Loader.IffDD.file_name).with_suffix('.pkl')
    
#     with open(output_path, 'wb') as f:
#         pickle.dump(data_package, f)
    

# =============================================================================
# Executable
# =============================================================================


if __name__ == '__main__':
    TESTING = True
    #-- Get input
    parser = argparse.ArgumentParser('infer_test_data', description="Compute features and store as npy array.", parents=[get_args_parser()])
    if TESTING:
        print("TESTING!", file=sys.stderr)
        args = parser.parse_args((
                                  "D:\\dtu\\speciale\\Data\\testData\\PineIsland\\track_065_descending\\iff_dd_rollingRef\\*.pha "
                                  + "D:\\dtu\\speciale\\Data\\testData\\Cook\\track_010_ascending\\iff_dd_rollingRef\\*.pha "
                                  + "--classifier_subdir D:\\dtu\speciale\\Data\\feature_dataset\\base_16_phase_448\\classifier\\002asc_099asc_038des_169des "
                                  + "--pretrained_weights D:\\dtu\\speciale\\Foundational_Models\\dino\\dino\\pretrained_weights "
                                  + "--train_segmentation_dir ../segmentation_v3 "
                                  + "--test_segmentation_dir ../ground_truth "
                                  + "--epoch best "
                                  + "--grounding_line_dir D:\\dtu\\speciale\\Grounding_line_masks\\for_post_processing "
                                  + "--save_image False "
                                  + "--batch_size 32"
                                  ).split())
    else: 
        args = parser.parse_args()
        
    main(args)