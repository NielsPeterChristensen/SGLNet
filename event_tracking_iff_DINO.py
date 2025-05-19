#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================


"""
event_tracking_iff_DINO.py is an executable script that runs the SGLNet
detection routine on iff_dd phase images and return event coordinates in
the iff_eventCoords directory.

:Authors
    NPKC / 02-04-2025 / creation / s203980@dtu.dk

:Todo
    ---

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
import pickle
import json
import os
import time
import numpy as np
import numpy.typing as npt
import pkg_resources
from glob import glob
from pathlib import Path
from PIL import Image
#-- PyTorch
import torch
import torchvision.transforms as pth_transforms
#-- SGLNet modules by Niels
import SGLNet.Corefunc.chunk_loader as chunk_loader
import SGLNet.Corefunc.chunk_masking as chunk_masking
import SGLNet.Corefunc.utils as utils
import SGLNet.NN.classifier as classifier
import SGLNet.Plotting.plot_utils as plot_utils
from SGLNet.PyIPP.iff_dd import Vrt
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
    parser = argparse.ArgumentParser('event_tracking_iff_DINO', add_help=False)
    
    parser.add_argument(
        "iff_dd_paths",
        nargs = '+',
        metavar = "/path/to/iff_dd/<iff_dd_interval>.pha",
        help = "/path/to/iff_dd_phases. Wildcards are possible.\n" \
               + "Note: .pha files are assumed to be in a single directory."
               )
    parser.add_argument(
        "-o",
        "--output_dir",
        type = str,
        metavar = "<str>",
        default = "../",
        help = "Path to output directory. Event coordinates and images are " \
               + "saved in subdirectories. Default is '../' relative to " \
               + "the directory with the iff_dd_paths. Directory is created if" \
               + " it does not exist."
               )
    parser.add_argument(
        '--DINO_weights',
        type = str,
        default = '',
        metavar = "<str>",
        help = "Path to pretrained DINO weights file or directory to use for inference. " \
               + "Default is '', which means the script looks for a suitable " \
               + "weights-file in the SGLNet.Weights folder. If this fails, " \
               + "relevant weights will automatically be downloded."
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
        "--batch_size",
        type = int,
        metavar = "<int>",
        default = 64,
        help = "Set the batch size to use for inference. Default is 64."
        )
    parser.add_argument(
        "--num_workers",
        type = int,
        metavar = "<int>",
        default = 4,
        help = "Number of data loading workers. Default is 4."
        )
    parser.add_argument(
        '--arch',
        type = str,
        default = 'vit_base',
        metavar = '<str>',
        choices = ['vit_small', 'vit_base'],
        help = "Name of architecture for inference. Default is vit_base."
        )
    parser.add_argument(
        '--patch_size',
        type = int,
        default = 16,
        metavar = '<int>',
        choices = [8, 16],
        help = "Patch size to use for ViT. Default is 16."
        )
    parser.add_argument(
        '--form',
        type = str,
        default = 'phase',
        metavar = "<str>",
        choices = ['image', 'phase', 'recta', 'polar'],
        help = "Which form the phase should be in. Default is phase."
        )
    parser.add_argument(
        '--chunk_size',
        type = int,
        default = 448,
        metavar = "<int>",
        help = "Chunk size that each image is divided into. Default is 224."
        )
    parser.add_argument(
        '--n_last_blocks',
        type = int,
        default = 1,
        metavar = "<int>",
        help = "Concatenate [CLS] tokens for the 'n' last blocks in the ViT." \
               + " DINO recommends n=4 for ViT-Small and n=1 for ViT-Base." \
               + " Default is 1."
               )
    parser.add_argument(
        '--avgpool_patchtokens',
        type = utils.bool_flag,
        default = True,
        metavar = "<bool>",
        help = "Concatenate the global average pooled features to the [CLS] token." \
               + " DINO recommends False for ViT-Small and True for ViT-Base." \
               + " Default is True."
               )
    parser.add_argument(
        '--overlap',
        type = utils.bool_flag,
        default = True,
        metavar = "<bool>",
        help = "Whether to use 50%% overlapping chunks. Default is True"
        )
    parser.add_argument(
        '--device',
        type = str,
        metavar = "<str>",
        default = 'cuda',
        choices = ['cuda', 'cpu'],
        help = "Specify whether to use cuda or cpu for training. Default is cuda."
        )
    parser.add_argument(
        '--save_images',
        type = utils.bool_flag,
        default = False,
        metavar = "<bool>",
        help = "Specify whether to save each test image with corresponding " \
               + "lake detections. Default is False."
               )
    parser.add_argument(
        '--save_predictions',
        type = utils.bool_flag,
        default = True,
        metavar = "<bool>",
        help = "Specify whether to save prediction vector for each chunk in the " \
               + "test images. Default is True."
               )
    parser.add_argument(
        '--save_bboxes',
        type = utils.bool_flag,
        default = True,
        metavar = "<bool>",
        help = "Specify whether to save boundary boxes (bboxes) of each chunk in the " \
               + "test data. Default is True."
               )
    parser.add_argument(
        '--save_embeddings',
        type = utils.bool_flag,
        default = True,
        metavar = "<bool>",
        help = ("Specify whether to save embeddings and self attention for each"
               " chunk in the data that is predicted 'True' (i.e. containing a lake)."
               " This is necessary if segmentation is performed afterwards."
               " Default is True.\n"
               " WARNING: Attention matrices are fairly large in terms of data"
               " size, hence why the function only stores the embeddings for"
               " lake chunks, and not all chunks in the image. However, if"
               " this option is enabled on data that contains e.g. grounding lines"
               " and a gl_mask is not given (so all grounding lines are detected"
               " as lakes), then the output embedding files can become very large."
               " Therefor, if not gl_mask is given on data with grounding lines,"
               " it is recommended to disable this option.")
               )
    parser.add_argument(
        '--save_eventcoords',
        type = utils.bool_flag,
        default = True,
        metavar = "<bool>",
        help = "Specify whether to save eventCoords for detected lakes." \
               + " Default is True."
               )
    parser.add_argument(
        '--save_metadata',
        type = utils.bool_flag,
        default = True,
        metavar = "<bool>",
        help = "Specify whether to save metadata with model parameters." \
               + " Default is True."
               )
    parser.add_argument(
        '--overwrite',
        type = utils.bool_flag,
        default = False,
        metavar = "<bool>",
        help = "Specify whether to overwrite already exists eventcoords txt files. " \
               + "Default is False."
               )
    parser.add_argument(
        '--gl_mask',
        type = str,
        metavar = "<str>",
        default = None,
        help = "Optinal input to exclude bboxes overlaping grounding lines. " \
               + "Default is None."
               )
    parser.add_argument(
        '--filter_bboxes',
        type = utils.bool_flag,
        metavar = "<bool>",
        default = True,
        help = "Filter bboxes to make sure bboxes are contained within the " \
               + "valid region of the iff_dd image. Bboxes with nan values at " \
               + "the corner are shrunk until the corners all become valid." \
               + "Default is True. Note: If bboxes are passed to gim_staging.py " \
               + "then --filter_bboxes True is likely required."
               )
    
    return parser

def main(args: argparse.Namespace) -> None:
        
    resolve_iff_paths(args)
    resolve_output(args)
    validate_pretrained_weights(args)
    check_grounding_line_mask(args)
    
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
    dino_utils.load_pretrained_weights(vit_model, args.DINO_weights, 'teacher', args.arch, args.patch_size)
    
    #-- Get relevant parameters
    args.embed_dim = vit_model.embed_dim
    args.output_dim = vit_model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    args.num_heads = vit_model.blocks[-1].attn.num_heads
    args.num_patches = (args.chunk_size // args.patch_size)**2
    args.num_tokens = args.num_patches + 1
    args.head_dim = args.embed_dim // args.num_heads
    
    #-- Setup custom dtype for structured numpy array
    args.embed_dtype = embedding_dtype(args)

    #-- Setup classifier model
    state_dict = torch.load(args.CLS_weights, map_location="cpu", weights_only=True)
    model_state_dict = state_dict["state_dict"]
    classifier_model = classifier.BinaryClassifier(args.output_dim)
    classifier_model.load_state_dict(model_state_dict)
    classifier_model.to(args.device)
    classifier_model.eval()
    print("Setup completed succesfully!", file=sys.stderr)
        
    for i, file in enumerate(args.iff_dd_paths):
        if args.overwrite or not all_outputs_exist(args, file):
            print(f"Evaluating file ({i+1}/{len(args.iff_dd_paths)}): {file.stem}", file=sys.stderr)
            inference(file, vit_model, classifier_model, data_transform, args)
        else:
            print(f"Skipping file ({i+1}/{len(args.iff_dd_paths)}): {file.stem}", file=sys.stderr)
        
    print("Event tracking complete!", file=sys.stderr)

def inference(file: Path, vit_model: vits.VisionTransformer, classifier_model: classifier.BinaryClassifier, transform: pth_transforms.Compose, args: argparse.Namespace):
    
    start_time = time.perf_counter()
    
    Loader = chunk_loader.TensorChunkLoader(
        image_path = file,
        transform = transform,
        chunk_size = args.chunk_size,
        overlap = args.overlap,
        form = args.form,
        batch_size = args.batch_size,
    )
    
    num_chunks = len(Loader)
    num_batches = int(np.ceil(num_chunks/args.batch_size))
    chunk_id = np.arange(num_chunks)
    bboxes = Loader.bboxes
    embedding_data = [np.zeros(0, dtype=args.embed_dtype)]
    
    #-- get index for grounding line chunks
    if args.gl_mask is not None:
        chunk_on_grounding_line = np.array([np.any(args.gl_mask[bbox[1]:(bbox[3]+1), bbox[0]:(bbox[2]+1)]) for bbox in bboxes])
    else:
        chunk_on_grounding_line = np.zeros((num_chunks,), dtype=bool)
    
    print(" Running inference on batches", file=sys.stderr)
    with torch.no_grad():
        predictions = np.zeros((num_chunks,), dtype=np.int32)
        for i, batch in enumerate(Loader):
            torch.cuda.empty_cache()
            batch = batch.to(args.device)
            idx = np.array(Loader._interbatch_idx)
            
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
                valid_batch_pred = batch_pred & ~chunk_on_grounding_line[idx]
                
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
            
            if (i+1)%10 == 0:
                print(f"    Batch {i+1}/{num_batches} complete", file=sys.stderr)
    print(f"    Batch {i+1}/{num_batches} complete", file=sys.stderr)
    #-- get combined array for embeddings         
    embedding_data = np.concatenate(embedding_data, axis=0)
    predictions[chunk_on_grounding_line] = -1
    final_bboxes, outlines = get_bboxes(args, predictions, Loader)
    
    end_time = time.perf_counter()
    run_time = end_time - start_time
           
    bname = file.stem
    subdir = args.eventtracking_dir / f"{bname}"
    if args.save_images or args.save_embeddings or args.save_predictions or args.save_bboxes or args.save_metadata:
        subdir.mkdir(exist_ok=True)
        
    if args.save_eventcoords:
        fname = args.eventcoords_dir / f"{bname}.txt"
        if not args.overwrite and fname.exists():
            print("  eventCoords exist already. Skipping", file=sys.stderr)
        else:
            print("  Saving eventCoords", file=sys.stderr)
            save_eventcoords(args, final_bboxes, fname)
    if args.save_images:
        fname = subdir / f"{bname}_img.png"
        if not args.overwrite and fname.exists():
            print("  Image exists already. Skipping", file=sys.stderr)
        else:
            print("  Saving image", file=sys.stderr)
            img = Loader()
            plot_utils.draw_outlines(img, outlines, color='red', linewidth=10)
            img.save(fname)
    if args.save_embeddings:
        fname = subdir / f"{bname}_embed.pkl"
        if not args.overwrite and fname.exists():
            print("  Chunk embeddings exist already. Skipping", file=sys.stderr)
        else:
            print("  Saving chunk embeddings", file=sys.stderr)
            with open(fname, 'wb') as f:
                pickle.dump(embedding_data, f)
    if args.save_predictions:
        fname = subdir / f"{bname}_pred.pkl"
        if not args.overwrite and fname.exists():
            print("  Chunk predictions exist already. Skipping", file=sys.stderr)
        else:
            print("  Saving chunk predictions", file=sys.stderr)
            with open(fname, 'wb') as f:
                pickle.dump(predictions, f)
    if args.save_bboxes:
        fname = subdir / f"{bname}_bbox.pkl"
        if not args.overwrite and fname.exists():
            print("  Chunk bboxes exists already. Skipping", file=sys.stderr)
        else:
            print("  Saving chunk bboxes", file=sys.stderr)
            with open(fname, 'wb') as f:
                pickle.dump(bboxes, f)
    if args.save_metadata:
        fname = subdir / f"{bname}_meta.json"
        if not args.overwrite and fname.exists():
            print("  Chunk metadata exists already. Skipping", file=sys.stderr)
        else:
            print("  Saving chunk metadata", file=sys.stderr)
            metadata = {**get_metadata(args), "runtime": run_time}
            with open(fname, 'w') as f:
                json.dump(metadata, f, indent=4)
    
def resolve_iff_paths(args: argparse.Namespace) -> None:
    """Resolves iff_dd_paths and sets pha_dir."""
    if os.name == "nt":
        paths = []
        for path in args.iff_dd_paths:
            paths.extend(glob(path))
        args.iff_dd_paths = paths
    args.iff_dd_paths = [Path(f) for f in args.iff_dd_paths]
    args.pha_dir = args.iff_dd_paths[0].parent
    args.track_dir = args.pha_dir.parent
    for path in args.iff_dd_paths:
        vrt = path.with_suffix('.vrt')
        if vrt.exists():
            args.imsize = Vrt(vrt).shape()[::-1]
            return
    print("Error: Unable to determine image size from .vrt files.", file=sys.stderr)
    sys.exit(1)
        
def resolve_output(args: argparse.Namespace) -> None:
    """Resolves and validates a directory path relative to pha_dir if necessary."""
    path = Path(args.output_dir)
    if not path.is_absolute():
        path = args.pha_dir / path
    args.output_dir = path.resolve()
    if not args.output_dir.is_dir():
        args.output_dir.mkdir(exist_ok=True)
        print(f"Output directory created at {args.output_dir}", file=sys.stderr)
    args.eventcoords_dir = args.output_dir / "iff_eventCoords"
    args.eventtracking_dir = args.output_dir / "iff_eventTracking"
    if args.save_eventcoords:
        args.eventcoords_dir.mkdir(exist_ok=True)
    if args.save_images or args.save_embeddings or args.save_predictions or args.save_bboxes or args.save_metadata:
        args.eventtracking_dir.mkdir(exist_ok=True)

def validate_pretrained_weights(args: argparse.Namespace) -> None:
    """Validates and resolves the pretrained weights path."""
    if args.DINO_weights == '':
        try:
            weight_file = utils.pretrained_weights_dict[args.arch][str(args.patch_size)]
            args.DINO_weights = Path(pkg_resources.resource_filename("SGLNet.Weights.dino", weight_file))
        except:
            print('Unable to load DINO weights from SGLNet.Weights. Weights will be downloaded instead.', file=sys.stderr)
            pass
    elif args.DINO_weights != '':
        args.DINO_weights = Path(str(args.DINO_weights)).resolve()
        if args.DINO_weights.is_dir():
            args.DINO_weights = args.DINO_weights / utils.pretrained_weights_dict[args.arch][str(args.patch_size)]
        if not args.DINO_weights.is_file():
            print(f'No valid file found for pretrained weights {args.DINO_weights}.', file=sys.stderr)
            sys.exit(1)
    args.version = f"{args.arch.split('_')[1]}_{args.patch_size}_{args.form}_{args.chunk_size}"
    if args.CLS_weights == '':
        args.CLS_weights = Path(pkg_resources.resource_filename("SGLNet.Weights.classifier", f"{args.version}.pth.tar"))
    if not args.CLS_weights.exists():
        print(f"Error: Failed to load CLS weights from {args.CLS_weights}.", file=sys.stderr)
        sys.exit(1)
    
def padding_size(size, chunk_size):
    return ((size + chunk_size - 1) // chunk_size) * chunk_size
        
def check_grounding_line_mask(args: argparse.Namespace) -> None:
    if args.gl_mask is not None:
        args.gl_mask = Path(args.gl_mask).resolve()
        if not args.gl_mask.exists():
            print(f"No valid grounding line mask found at {args.gl_mask}", file=sys.stderr)
            sys.exit(1)
        mask = Image.open(args.gl_mask).convert('L')
        if mask.size != args.imsize:
            print(f"Warning: size of grounding line mask ({mask.size}) does not " \
                  + "fit size of iff_dd image ({args.imsize}) so the mask is " \
                  + "being resized. Be cautious as this might cause trouble.",
                  file = sys.stderr)
            mask = mask.resize(args.imsize, Image.LANCZOS)
        padded_size_y = padding_size(args.imsize[0], args.chunk_size // (1+int(args.overlap)))
        padded_size_x = padding_size(args.imsize[1], args.chunk_size // (1+int(args.overlap)))
        pad_y = (0, padded_size_y - mask.size[0])
        pad_x = (0, padded_size_x - mask.size[1])
        mask = np.array(mask, dtype=np.float32)
        mask = np.pad(mask, (pad_x, pad_y), mode='constant', constant_values=0)
        args.gl_mask = mask.astype(bool)   
        
def all_outputs_exist(args: argparse.Namespace, fname: Path) -> bool:
    bname = fname.stem
    subdir = args.eventtracking_dir / bname
    if args.save_eventcoords:
        f = args.eventcoords_dir / f"{bname}.txt"
        if not f.exists():
            return False
    if args.save_images:
        f = subdir / f"{bname}_img.png"
        if not f.exists():
            return False
    if args.save_embeddings:
        f = subdir / f"{bname}_embed.pkl"
        if not f.exists():
            return False
    if args.save_predictions:
        f = subdir / f"{bname}_pred.pkl"
        if not f.exists():
            return False
    if args.save_bboxes:
        f = subdir / f"{bname}_bbox.pkl"
        if not f.exists():
            return False
    if args.save_metadata:
        f = subdir / f"{bname}_meta.json"
        if not f.exists():
            return False
    return True

def get_bboxes(args: argparse.Namespace, predictions: npt.ArrayLike, Loader: chunk_loader.ImageChunkLoader) -> np.ndarray[int]:
    if isinstance(predictions, torch.Tensor): 
        predictions = predictions.numpy()
    predictions = (predictions == 1).reshape((-1,))
    detected_bboxes = Loader.bboxes[predictions]
    simplified_bboxes, outlines = utils.simplify_bboxes(detected_bboxes, Loader.shape[::-1])
    
    if args.filter_bboxes:
        nanval = 9.99999968266e-21
        pha_file = Loader.IffDD.file_path.with_suffix('.pha')   
        phase = np.memmap(pha_file, dtype=">f4", mode='r', shape=Loader.IffDD.vrt.shape())
        isnan = np.ones(Loader.shape, dtype=bool)
        isnan[0:phase.shape[0], 0:phase.shape[1]] = phase == nanval
        
        simplified_bboxes = utils.filter_nan_bboxes(simplified_bboxes, isnan)
        
    return simplified_bboxes, outlines

def get_metadata(args: argparse.Namespace) -> dict:
    metadata = {
        'arch': args.arch,
        'patch_size': args.patch_size,
        'form': args.form,
        'chunk_size': args.chunk_size,
        'overlap': args.overlap,
        'embed_dim': args.embed_dim,
        'head_dim': args.head_dim,
        'imsize': args.imsize,
        'num_heads': args.num_heads,
        'num_patches': args.num_patches,
        'num_tokens': args.num_tokens,
        'output_dim': args.output_dim,
    }
    return metadata

def save_eventcoords(args: argparse.Namespace, simplified_bboxes: np.ndarray[int], fname: os.PathLike) -> None:
    with open(fname, 'w') as f:
        f.write("li0 sa0 li1 sa1\n")
        for (sa0, li0, sa1, li1) in simplified_bboxes:
            li0 = max(li0, 0)
            sa0 = max(sa0, 0)
            li1 = min(li1, args.imsize[1])
            sa1 = min(sa1, args.imsize[0])
            f.write(f"{li0} {sa0} {li1} {sa1}\n")

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


# =============================================================================
# Executable
# =============================================================================


if __name__ == '__main__':
    TESTING = False
    #-- Get input
    parser = argparse.ArgumentParser('event_tracking_iff_DINO', description="Compute features and store as npy array.", parents=[get_args_parser()])
    if TESTING:
        print("TESTING!", file=sys.stderr)
        args = parser.parse_args((
            # "D:\\dtu\\speciale\\ipp\\processed\\track_065_descending\\iff_dd\\*.pha "
            "D:\\dtu\\speciale\\ipp\\processed\\track_010_ascending\\iff_dd\\*.pha "
            + "--gl_mask D:\\dtu\\speciale\\Grounding_line_masks\\for_post_processing\\track_010_ascending.png "
            # + "--overwrite True"
        ).split())
    else: 
        args = parser.parse_args()
        
    main(args)