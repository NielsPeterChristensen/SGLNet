#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================


"""
image2feat.py takes iff_dd phases and extracts feature vectors
(embeddings) usign the DINO foundational model [1]

:Authors
    NPKC / 18-02-2025 / creation / s203980@dtu.dk

:Todo
    Add more comments

:References
    [1] Mathilde Caron et. al. Emerging Properties in Self-Supervised Vision 
    Transformers https://arxiv.org/abs/2104.14294

:Note:
    Uses DINO [1] utility functions
"""


# =============================================================================
# Modules
# =============================================================================

#-- Utilities
import argparse
import sys
import datetime
import json
from pathlib import Path
#-- PyTorch
import torch
import torch.nn as nn
#-- My own
import SGLNet.NN.classifier as classifier
import SGLNet.Corefunc.utils as utils
#-- from DINO [1]
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
    parser = argparse.ArgumentParser('train_classifier_head', add_help=False)
    
    parser.add_argument(
        "data_dir",
        type = str,
        metavar = "<str>",
        help = "Path to directory containing specific feature_datasets. " \
               + "This directory contains subdirectories named "\
               + "'bboxes', 'features', 'labels', 'metadata'.")
    parser.add_argument(
        "-o",
        "--output_dir",
        type = str,
        metavar = "<str>",
        default = "./classifier",
        help = "Path to output directory to save logs and checkpoints. " \
               + "Default is './classifier' relative to data_dir.")
    parser.add_argument(
        "-t",
        "--tracks",
        type = str,
        nargs='*',
        metavar = "<str>",
        default = ["track_002_ascending", "track_065_descending", 
                   "track_169_descending", "track_099_ascending"],
        help = "Names of tracks to be used when training classifier head. " \
               + 'Default is "track_002_ascending" "track_065_descending", ' \
               + '"track_169_descending" "track_099_ascending"')
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
        '--epochs',
        type = int,
        metavar = "<int>",
        default = 100,
        help = "Number of epochs in training. Default is 100.")
    parser.add_argument(
        '--lr',
        type = float,
        metavar = "<float>",
        default = 0.001,
        help = "Learning rate at the beginning of training. Rate is linearly" \
               + "scaled with the batch size. Default is 0.001.")
    parser.add_argument(
        '--ratio',
        type = float,
        metavar = "<float>",
        default = 0.8,
        help = "Ratio between testing and validation data of the full dataset." \
               + "Default is 0.8.")
    parser.add_argument(
        '--val_freq',
        type = int,
        metavar = "<int>",
        default = 1,
        help = "Validation frequency in epochs. Default is 1.")
    parser.add_argument(
        '--seed',
        type = int,
        metavar = "<int>",
        default = 123,
        help = "Set seed for random processes. Default is 123.")
    parser.add_argument(
        '--save_every_checkpoint',
        type = utils.bool_flag,
        metavar = "<bool>",
        default = True,
        help = "Set flag to save state_dict from every epoch instead of" \
               + "final epoch only. Default is True.")
    parser.add_argument(
        '--from_checkpoint',
        type = int,
        metavar = "<int>",
        default = None,
        help = "String with path of epoch checkpoint to start training from. " \
               + "Default is None.")
    parser.add_argument(
        '--device',
        type = str,
        metavar = "<str>",
        default = 'cuda',
        choices = ['cuda', 'cpu'],
        help = "Specify whether to use cuda or cpu for training. Default is cuda.")
    
    return parser


def main(args: argparse.Namespace) -> None:
    
    #-- Check input data
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"data_dir {str(data_dir)} does not exist.", file=sys.stderr)
        sys.exit(1)    
    if isinstance(args.tracks, str):
        args.tracks = [args.tracks]
    feature_files = [data_dir / 'features' / f'{track}.dat' for track in args.tracks]
    label_files = [data_dir / 'labels' / f'{track}.dat' for track in args.tracks]
    meta_files = [data_dir / 'metadata' / f'{track}.pkl' for track in args.tracks]
    for (f, l, m, t) in zip(feature_files, label_files, meta_files, args.tracks):
        if not f.exists():
            print(f"no features found for {t}.", file=sys.stderr)
            sys.exit(1) 
        if not l.exists():
            print(f"no labels found for {t}.", file=sys.stderr)
            sys.exit(1) 
        if not m.exists():
            print(f"no metadata found for {t}.", file=sys.stderr)
            sys.exit(1)
        
    print("Started setup at {time}".format(time=datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d")), file=sys.stderr)
           
    #-- Setup output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (data_dir / output_dir).resolve()
    
    output_name = '_'.join([string.split('_')[1] + string.split('_')[2][:3] for string in args.tracks])
    output_dir = output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict_dir = output_dir / "checkpoints"
    state_dict_dir.mkdir(parents=True, exist_ok=True)
    
    #-- PyTorch setup
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    dino_utils.fix_random_seeds(args.seed)
    
    #-- Prepare data
    print("Loading dataset...", file=sys.stderr)
    dataset = classifier.LakeDataset_v2(feature_files, label_files, meta_files)
    (train_ind, val_ind) = get_subset_indices(dataset.labels, args.ratio)
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers,
        sampler = torch.utils.data.SubsetRandomSampler(train_ind),
        pin_memory = True)
    val_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers,
        sampler = torch.utils.data.SubsetRandomSampler(val_ind),
        pin_memory = True)
    embed_dim = dataset.embed_dim
    pos_weight = torch.tensor(dataset.n_nolake / dataset.n_lake, dtype=torch.float32).to(device)
    print(f"Data loaded: {len(dataset)} feature vectors with {embed_dim=}.", file=sys.stderr)
    print(f"Data contains {sum(dataset.labels).item():.0f} lake features ({(sum(dataset.labels)/len(dataset)).item()*100:.2f}%).", file=sys.stderr)
    print(f"Training set consists of {len(train_ind)} feature vectors ({(len(train_ind)/len(dataset))*100:.0f}%) with #lakes {sum(dataset.labels[val_ind]).item():.0f} ({(sum(dataset.labels[val_ind])/sum(dataset.labels)).item()*100:.0f}%).", file=sys.stderr)
    print(f"Validation set consists of {len(val_ind)} feature vectors ({(len(val_ind)/len(dataset))*100:.0f}%) with #lakes {sum(dataset.labels[train_ind]).item():.0f} ({(sum(dataset.labels[train_ind])/sum(dataset.labels)).item()*100:.0f}%).", file=sys.stderr)
    
    #-- Build network
    model = classifier.BinaryClassifier(embed_dim)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    #-- Set optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr * (args.batch_size * dino_utils.get_world_size()) / 256.,
        momentum=0.9,
        weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    #-- Load checkpoint
    to_restore = {"epoch": 0, "best_perf": ''}
    dino_utils.restart_from_checkpoint(
        str(state_dict_dir / f"epoch{args.from_checkpoint}.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_perf = to_restore["best_perf"]
    
    #-- Setup validation
    perf_logger = utils.PerformanceLogger(device)
    perf_metrics = utils.PerformanceMetrics(best_perf)
    
    #-- Train network
    start_text = "Started network training at {time}\n".format(time=datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"))
    print(start_text, file=sys.stderr)
    with (output_dir / "log.txt").open("w") as f:
        f.write(start_text + "\n")
        
    for epoch in range(start_epoch, args.epochs):        
        train_stats = train_one_epoch(model, train_loader, device, loss_fn, optimizer, epoch)
        print("{time}: Epoch {epoch}: lr: {lr:.6f} loss: {loss:.6f}".format(time=datetime.datetime.now().strftime("%H:%M:%S"), epoch=epoch, lr=train_stats['lr'], loss=train_stats['loss']), file=sys.stderr)
        scheduler.step()
        log_stats = {**{f'train_{k}': f'{v:.6f}'[:8] for k, v in train_stats.items()}, 'epoch': f'{epoch:03.0f}'}
    
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate(model, val_loader, device, loss_fn, perf_logger)
            perf_metrics.update(test_stats, epoch)
            print(perf_metrics, file=sys.stderr)
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{k: v for k, v in perf_metrics.log.items()}}
        
        with (output_dir / "log.txt").open("a") as f:
            f.write("{time}: ".format(time=datetime.datetime.now().strftime("%H:%M:%S")))
            f.write(json.dumps(log_stats) + "\n")
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_perf": str(perf_metrics),
        }
        if args.save_every_checkpoint:
            torch.save(save_dict, str(state_dict_dir / f"epoch{epoch}.pth.tar"))
        elif epoch == args.epochs - 1:
            torch.save(save_dict, str(state_dict_dir / "checkpoint.pth.tar"))
            
    
    print(f"Training of classifier completed. Best performance: "
      f"best_acc: {perf_metrics.best_acc:.2f} (epoch {perf_metrics.epoch_acc})  "
      f"best_spec: {perf_metrics.best_spec:.2f} (epoch {perf_metrics.epoch_spec})  "
      f"best_prec: {perf_metrics.best_prec:.2f} (epoch {perf_metrics.epoch_prec})  "
      f"best_recall: {perf_metrics.best_recall:.2f} (epoch {perf_metrics.epoch_recall})  "
      f"best_f1:  {perf_metrics.best_f1:.2f} (epoch {perf_metrics.epoch_f1})", file=sys.stderr)
    print("All logs and model parameters are stored in {dir}.".format(dir=str(output_dir)), file=sys.stderr)
    
    
def train_one_epoch(model, loader, device, loss_fn, optimizer, epoch) -> dict:
    #-- Setup
    model.train()
    metric_logger = utils.Logger()
    for (feature, target) in loader:
        #-- Move data to GPU
        feature = feature.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        #-- Forward pass
        output = model(feature)
        
        #-- Compute loss and update weights
        loss = loss_fn(output, target)  # Compute Binary Cross Entropy loss
        optimizer.zero_grad()           # Reset gradient from previous iter
        loss.backward()                 # Calculate new gradients
        optimizer.step()                # Update weights
        
        #-- Log results
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    return {k: meter() for k, meter in metric_logger.meters.items()}
        
        
@torch.no_grad()
def validate(model, loader, device, loss_fn, perf_logger) -> dict:
    #-- Setup
    perf_logger.reset()
    model.eval()
    metric_logger = utils.Logger()
    for (feature, target) in loader:
        #-- Move data to GPU
        feature = feature.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        #-- Forward pass
        output = model(feature)
        
        #-- Compute loss and log
        loss = loss_fn(output, target)
        metric_logger.update(loss=loss.item())
        
        #-- Apply activation
        output = model.activation(output)
        
        #-- Update performance logger
        perf_logger.update(output, target)
        
    #-- Return performance log
    return {
        **{k: meter() for k, meter in metric_logger.meters.items()},
        **perf_logger.compute()}


def get_subset_indices(labels: torch.Tensor, ratio: float) -> tuple[list, list]:
    
    nolake_indices = torch.where(labels == 0)[0]
    lake_indices = torch.where(labels == 1)[0]
    nolake_indices = nolake_indices[torch.randperm(len(nolake_indices))]
    lake_indices = lake_indices[torch.randperm(len(lake_indices))]
    
    nolake_split = int(ratio * len(nolake_indices))
    lake_split = int(ratio * len(lake_indices))
    
    train_indices = torch.cat([nolake_indices[:nolake_split], lake_indices[:lake_split]])
    val_indices = torch.cat([nolake_indices[nolake_split:], lake_indices[lake_split:]])
    
    return train_indices, val_indices
        
    
# =============================================================================
# Executable    
# =============================================================================
    

if __name__ == "__main__":
    TESTING = True
    
    parser = argparse.ArgumentParser('train_classifier_head', description="Train binary classifier.", parents=[get_args_parser()])
    
    if TESTING:
        print("TESTING!", file=sys.stderr)
        args = parser.parse_args('D:\\dtu\\speciale\\Data\\feature_dataset\\small_16_phase_224 -t track_010_ascending track_038_descending --epochs 5'.split())
    else:
        args = parser.parse_args()
    
    print("Executing train_classificer_head.py with argparse input:", file=sys.stderr)
    print(args, file=sys.stderr)
    main(args)
    