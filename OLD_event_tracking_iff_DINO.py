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
import os
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
import SGLNet.Corefunc.utils as utils
import SGLNet.NN.classifier as classifier
import SGLNet.Plotting.plot_utils as plot_utils
from SGLNet.PyIPP.iff_dd import Vrt
from SGLNet.image2trainset import pretrained_weights_dict as DINO_weights_dict
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
               + "the directory with the iff_dd_paths."
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
        "--batch_size",
        type = int,
        metavar = "<int>",
        default = 64,
        help = "Set the batch size to use for training. Default is 512."
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
        default = 224,
        metavar = "<int>",
        help = "Chunk size that each image is divided into. Default is 224."
        )
    parser.add_argument(
        '--n_last_blocks',
        type = int,
        default = 1,
        metavar = "<int>",
        help = "Concatenate [CLS] tokens for the 'n' last blocks in the ViT." \
               + "DINO recommends n=4 for ViT-Small and n=1 for ViT-Base." \
               + "Default is 1."
               )
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
        '--downscale_factor',
        type = int,
        default = 1,
        metavar = "<int>",
        choices= [1, 2, 4],
        help = "Downscaling factor. Default is 1. Note: This option has not be " \
               + "tested after first implementation."
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
               + "lake detections. Default is True."
               )
    parser.add_argument(
        '--save_features',
        type = utils.bool_flag,
        default = True,
        metavar = "<bool>",
        help = "Specify whether to save feature vectors for each chunk in the " \
               + "test images. Default is True."
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
        default = False,
        help = "Filter bboxes to make sure bboxes are contained within the " \
               + "valid region of the iff_dd image. Bboxes with nan values at " \
               + "the corner are shrunk until the corners all become valid." \
               + "Default is False. Note: If bboxes are passed to gim_staging.py " \
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
    args.embed_dim = vit_model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    vit_model.to(args.device)
    vit_model.eval()
    dino_utils.load_pretrained_weights(vit_model, args.DINO_weights, 'teacher', args.arch, args.patch_size)

    #-- Setup classifier model
    state_dict = torch.load(args.CLS_weights, map_location="cpu", weights_only=True)
    model_state_dict = state_dict["state_dict"]
    classifier_model = classifier.BinaryClassifier(args.embed_dim)
    classifier_model.load_state_dict(model_state_dict)
    classifier_model.to(args.device)
    classifier_model.eval()
    print("Setup completed succesfully!", file=sys.stderr)
        
    for i, file in enumerate(args.iff_dd_paths):
        print(f"Running inference for file ({i+1}/{len(args.iff_dd_paths)}): {file.stem}", file=sys.stderr)
        if args.overwrite or not (args.eventcoords_dir / file.with_suffix('.txt').name).exists():
            inference(file, vit_model, classifier_model, data_transform, args)
        
    print("Event tracking complete!", file=sys.stderr)

def inference(file: Path, vit_model: vits.VisionTransformer, classifier_model: classifier.BinaryClassifier, transform: pth_transforms.Compose, args: argparse.Namespace):
    
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
           
    final_bboxes = get_bboxes(args, predictions, Loader)
    save_eventcoords(args, final_bboxes, Loader)
    if args.save_images:
        save_Image(args, final_bboxes, Loader)
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
    args.output_dir = path
    if not args.output_dir.exists():
        print(f"Error: Output directory not found at {args.output_dir}", file=sys.stderr)
        sys.exit(1)
    args.eventcoords_dir = args.output_dir / "iff_eventCoords"
    args.eventcoords_dir.mkdir(exist_ok=True)
    if args.save_images or args.save_features:
        args.supplementary_dir = args.output_dir / "event_detection_supplementary"
        args.supplementary_dir.mkdir(exist_ok=True)

def validate_pretrained_weights(args: argparse.Namespace) -> None:
    """Validates and resolves the pretrained weights path."""
    if args.DINO_weights != '':
        args.DINO_weights = Path(str(args.DINO_weights)).resolve()
        if args.DINO_weights.is_dir():
            args.DINO_weights = args.DINO_weights / DINO_weights_dict[args.arch][str(args.patch_size)]
        if not args.DINO_weights.is_file():
            print(f'No valid file found for pretrained weights {args.DINO_weights}.', file=sys.stderr)
            sys.exit(1)
    args.version = f"{args.arch.split('_')[1]}_{args.patch_size}_{args.form}_{args.chunk_size}"
    if args.CLS_weights == '':
        args.CLS_weights = Path(pkg_resources.resource_filename("SGLNet.Weights", f"{args.version}.pth.tar"))
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
        args.gl_mask = mask      

def get_bboxes(args: argparse.Namespace, predictions: npt.ArrayLike, Loader: chunk_loader.ImageChunkLoader) -> np.ndarray[int]:
    if isinstance(predictions, torch.Tensor): 
        predictions = predictions.numpy()
    predictions = np.array(predictions, dtype=bool).reshape((-1,))
    detected_bboxes = Loader.bboxes[predictions]
    simplified_bboxes = utils.simplify_bboxes(detected_bboxes, Loader.shape[::-1], gl_mask=args.gl_mask)
    
    if args.filter_bboxes:
        nanval = 9.99999968266e-21
        pha_file = Loader.IffDD.file_path.with_suffix('.pha')   
        phase = np.memmap(pha_file, dtype=">f4", mode='r', shape=Loader.IffDD.vrt.shape())
        isnan = np.ones(Loader.shape, dtype=bool)
        isnan[0:phase.shape[0], 0:phase.shape[1]] = phase == nanval
        
        simplified_bboxes = utils.filter_nan_bboxes(simplified_bboxes, isnan)
        
    return simplified_bboxes
        
def save_eventcoords(args: argparse.Namespace, simplified_bboxes: np.ndarray[int], Loader: chunk_loader.ImageChunkLoader) -> None:
    fname = (args.eventcoords_dir / Loader.IffDD.file_name).with_suffix('.txt')
    with open(fname, 'w') as f:
        f.write("li0 sa0 li1 sa1\n")
        for (sa0, li0, sa1, li1) in simplified_bboxes:
            li0 = max(li0, 0)
            sa0 = max(sa0, 0)
            li1 = min(li1, args.imsize[1])
            sa1 = min(sa1, args.imsize[0])
            f.write(f"{li0} {sa0} {li1} {sa1}\n")
    
def save_Image(args: argparse.Namespace, simplified_bboxes: np.ndarray[int], Loader: chunk_loader.ImageChunkLoader) -> None:
    """Save image with predicted lake outline drawn on."""
    Img = Loader()
    plot_utils.draw_bbox(Img, simplified_bboxes, color='red', linewidth=10)
    Img.save((args.supplementary_dir / Loader.IffDD.file_name).with_suffix('.png'))
    
def save_features(args: argparse.Namespace, features: torch.Tensor, predictions: torch.Tensor, Loader: chunk_loader.ImageChunkLoader) -> None:
    """Store feature vectors and corresponding bboxes as .pt file"""
    data_package = {
        "features": features,
        "predictions": predictions,
        "bboxes": Loader.bboxes,
        "iff_dd": str(Loader.IffDD.file_path), 
        "params": {
            "chunk_size": Loader.chunk_size,
            "overlap": Loader.overlap,
            "downscale_factor": Loader.downscale_factor,
            "form": Loader.form,
            "shape": Loader.shape
        }
    }
    output_path = (args.supplementary_dir / Loader.IffDD.file_name).with_suffix('.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(data_package, f)


# =============================================================================
# Executable
# =============================================================================


if __name__ == '__main__':
    TESTING = True
    #-- Get input
    parser = argparse.ArgumentParser('event_tracking_iff_DINO', description="Compute features and store as npy array.", parents=[get_args_parser()])
    if TESTING:
        print("TESTING!", file=sys.stderr)
        args = parser.parse_args(("D:\\dtu\\speciale\\ipp\\processed\\track_065_descending\\iff_dd\\*.pha "
                                  + "--DINO_weights D:\\dtu\\speciale\\Foundational_Models\\dino\\dino\\pretrained_weights "
                                  + "--save_images True "
                                  + "--save_features False "
                                  # + "--gl_mask D:\\dtu\\speciale\\Grounding_line_masks\\for_post_processing\\track_099_ascending_10x2ML.png "
                                  + "--filter_bboxes True "
                                  # + "--overwrite True"
                                  ).split())
    else: 
        args = parser.parse_args()
        
    main(args)