# =============================================================================
# Docstring
# =============================================================================


"""
utils.py contain helpful utility functions.

:Authors
    NPKC / 14-01-2025 / creation / s203980@dtu.dk

:Todo
    Add more comments

:References
    [1] Mathilde Caron et. al. Emerging Properties in Self-Supervised Vision 
    Transformers https://arxiv.org/abs/2104.14294

:Note:
    None
"""


# =============================================================================
# Packages
# =============================================================================


import sys
import torch
import numpy as np
import numpy.typing as npt
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from collections import defaultdict
import skimage.measure as measure
import torchmetrics


# =============================================================================
# Constants
# =============================================================================


MIN_PHA = -np.pi
MAX_PHA = np.pi
MIN_MAG = 0 
MAX_MAG = 255

pretrained_weights_dict = {
    'vit_small': {
        '16': 'dino_deitsmall16_pretrain.pth',
        '8': 'dino_deitsmall8_pretrain.pth'
    },
    'vit_base': {
        '16': 'dino_vitbase16_pretrain.pth',
        '8': 'dino_vitbase8_pretrain.pth'
    }
}


# =============================================================================
# Functions
# =============================================================================


def bool_flag(flag):
    """Bool flag for argparse. Inspired by [1]."""
    
    import argparse
    
    TRUE = {"true", "1", "on"}
    FALSE = {"false", "0", "off"}
    if flag.lower() in TRUE:
        return True 
    if flag.lower() in FALSE:
        return False 
    else:
        raise argparse.ArgumentTypeError(f"Unknown input {flag} for bool_flag.")


def normalize(arr: np.ndarray, arr_min: float = None, arr_max: float = None) -> np.ndarray:
    if arr_min is None:
        arr_min = np.quantile(arr, 0.05)
    if arr_max is None:
        arr_max = np.quantile(arr, 0.95)
    arr = arr.clip(arr_min, arr_max)
    return (arr-arr_min) / (arr_max-arr_min)


def remove_extensions(path: Path) -> Path:
    """Removes all extensions from a given Path object."""
    while path.suffix:
        path = path.with_suffix("")
    return path


def unix2nt(unix_path: Path) -> Path:
    unix_path = Path(unix_path).as_posix()
    
    if unix_path.startswith("/mnt/"):
        parts = unix_path.strip("/").split("/")
        drive_letter = parts[1].upper() + ":/"
        return Path(drive_letter, *parts[2:])
    
    return Path(unix_path)


def first_multiple(x, mult):
    return int(np.ceil(x / mult) * mult)


def pad_image(img, unit_size):
    us = unit_size
    left = 0
    top = 0
    x, y = img.size
    right = int(first_multiple(x, us) - x)
    bottom = int(first_multiple(y, us) - y)
    return ImageOps.expand(img, border=(left, top, right, bottom), fill = (0,0,0,0))

def pad_array(array: npt.NDArray, imsize: tuple[int], chunk_size: int, overlap: bool, value = False) -> npt.NDArray:
    
    # def padding_size(size, chunk_size):
    #     return ((size + chunk_size - 1) // chunk_size) * chunk_size
    
    # padded_size_y = padding_size(imsize[0], chunk_size // (1+int(overlap)))
    # padded_size_x = padding_size(imsize[1], chunk_size // (1+int(overlap)))
    
    padded_size_y = first_multiple(imsize[0], chunk_size)# // (1+int(overlap)))
    padded_size_x = first_multiple(imsize[1], chunk_size)# // (1+int(overlap)))
    pad_y = (0, padded_size_y - array.shape[0])
    pad_x = (0, padded_size_x - array.shape[1])
    return np.pad(array, (pad_y, pad_x), mode='constant', constant_values=value)

def direct_phase_image(file_path: str, channels: list = [0,1,2], nan: float = 9.99999968266e-21) -> Image.Image:
    import SGLNet.PyIPP.iff_dd as iff_dd
    iffdd = iff_dd.IffDD(file_path)
    phase = iffdd.pha()
    nan_idx = phase == nan
    phase = (normalize(phase, MIN_PHA, MAX_PHA) * 255).astype(np.uint8)
    phase[nan_idx] = 0
    zero_array = np.zeros(phase.shape, dtype=np.uint8)
    image = np.stack([phase if i in channels else zero_array for i in range(3)])
    alpha = (~nan_idx).astype(np.uint8) * 255
    image = np.concatenate((image, alpha[None, :, :]))
    return Image.fromarray(np.transpose(image, (1, 2, 0)), mode="RGBA")

def phase_image(phase: np.ndarray, channels: list = [0,1,2], nan: float = 9.99999968266e-21) -> Image.Image:
    nan_idx = phase == nan
    phase = (normalize(phase, MIN_PHA, MAX_PHA) * 255).astype(np.uint8)
    phase[nan_idx] = 0
    zero_array = np.zeros(phase.shape, dtype=np.uint8)
    image = np.stack([phase if i in channels else zero_array for i in range(3)])
    return Image.fromarray(np.transpose(image, (1, 2, 0)))


def phase2_image(phase: np.ndarray, nan: float = 9.99999968266e-21) -> Image.Image:
    nan_idx = phase == nan
    phase = (normalize(phase, MIN_PHA, MAX_PHA) * 255).astype(np.uint8)
    phase[nan_idx] = 0
    zero_array = np.zeros(phase.shape, dtype=np.uint8)
    nan_array = np.copy(zero_array)
    nan_array[nan_idx] = 255
    image = np.stack((phase, zero_array, nan_array))
    return Image.fromarray(np.transpose(image, (1, 2, 0)))


def recta_image(phase: np.ndarray, magnitude: np.ndarray = None, nan: float = 9.99999968266e-21) -> Image.Image:
    nan_idx = phase == nan
    if magnitude is not None:
        magnitude = normalize(magnitude, MIN_MAG, MAX_MAG)
    else:
        magnitude = 1
    real = (magnitude*np.cos(phase) * 255).astype(np.uint8)
    imag = (magnitude*np.sin(phase) * 255).astype(np.uint8)
    real[nan_idx] = 0
    imag[nan_idx] = 0
    zero_array = np.zeros(phase.shape, dtype=np.uint8)
    image = np.stack([real, imag, zero_array])
    return Image.fromarray(np.transpose(image, (1, 2, 0)))


def recta2_image(phase: np.ndarray, magnitude: np.ndarray = None, nan: float = 9.99999968266e-21) -> Image.Image:
    nan_idx = phase == nan
    if magnitude is not None:
        magnitude = normalize(magnitude, MIN_MAG, MAX_MAG)
    else:
        magnitude = 1
    real = (magnitude*np.cos(phase) * 255).astype(np.uint8)
    imag = (magnitude*np.sin(phase) * 255).astype(np.uint8)
    real[nan_idx] = 0
    imag[nan_idx] = 0
    zero_array = np.zeros(phase.shape, dtype=np.uint8)
    zero_array[nan_idx] = 255
    image = np.stack([real, imag, zero_array])
    return Image.fromarray(np.transpose(image, (1, 2, 0)))


def polar_image(phase: np.ndarray, magnitude: np.ndarray, nan: float = 9.99999968266e-21) -> Image.Image:
    nan_idx = phase == nan
    magnitude = (normalize(magnitude, MIN_MAG, MAX_MAG) * 255).astype(np.uint8)
    phase = (normalize(phase, MIN_PHA, MAX_PHA) * 255).astype(np.uint8)
    magnitude[nan_idx] = 0
    phase[nan_idx] = 0
    zero_array = np.zeros(phase.shape, dtype=np.uint8)
    image = np.stack([magnitude, phase, zero_array])
    return Image.fromarray(np.transpose(image, (1, 2, 0)))


def polar2_image(phase: np.ndarray, magnitude: np.ndarray, nan: float = 9.99999968266e-21) -> Image.Image:
    nan_idx = phase == nan
    magnitude = (normalize(magnitude, MIN_MAG, MAX_MAG) * 255).astype(np.uint8)
    phase = (normalize(phase, MIN_PHA, MAX_PHA) * 255).astype(np.uint8)
    magnitude[nan_idx] = 0
    phase[nan_idx] = 0
    zero_array = np.zeros(phase.shape, dtype=np.uint8)
    zero_array[nan_idx] = 255
    image = np.stack([magnitude, phase, zero_array])
    return Image.fromarray(np.transpose(image, (1, 2, 0)))


def get_outlines(bboxes: npt.ArrayLike, image_size: tuple) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Takes an Nx4 array of bounding boxes and returns simplified outlines of connected components using skimage.
    image_size must have shape according to PIL Image (Width, Height)
    """
    # Create a blank binary image
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)

    # Draw filled bounding boxes
    for x0, y0, x1, y1 in bboxes:
        draw.rectangle([x0, y0, x1, y1], fill=255)

    # Convert PIL image to numpy array
    mask_np = np.array(mask) / 255  # Normalize to 0-1

    # Find contours using skimage
    contours = measure.find_contours(mask_np, level=0.5)  # Extract boundaries

    # Convert contours to integer coordinates
    outlines = [np.round(c).astype(int) for c in contours]

    return outlines, mask


def bboxes2mask(bboxes: npt.ArrayLike, image_size: tuple[int, int]) -> np.ndarray:
    bboxes = np.array(bboxes)
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)

    # Draw filled bounding boxes
    for x0, y0, x1, y1 in bboxes:
        draw.rectangle([x0, y0, x1, y1], fill=255)

    # Convert PIL image to numpy array
    return np.array(mask).astype(bool)


def simplify_bboxes(bboxes: npt.ArrayLike, image_size: tuple, gl_mask: npt.ArrayLike = None, pad: int = None) -> np.ndarray:
    
    if gl_mask is not None:
        gl_mask = np.array(gl_mask, dtype=bool)
        bboxes = [bbox for bbox in bboxes if not np.any(gl_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]])]
    
    if len(bboxes) == 0:
        return np.zeros((0,4)), []
    pad = pad or (bboxes[0][2]-bboxes[0][0]) // 4
    outlines, _ = get_outlines(bboxes, image_size)
    
    simplified_bboxes = np.zeros((len(outlines), 4), dtype=int)
    for i, outline in enumerate(outlines):
        ymin = outline[:,0].min() - pad
        xmin = outline[:,1].min() - pad
        ymax = outline[:,0].max() + pad
        xmax = outline[:,1].max() + pad
        simplified_bboxes[i, 0] = np.maximum(0, xmin)
        simplified_bboxes[i, 1] = np.maximum(0, ymin)
        simplified_bboxes[i, 2] = np.minimum(image_size[0]-1, xmax)
        simplified_bboxes[i, 3] = np.minimum(image_size[1]-1, ymax)
    
    return simplified_bboxes, outlines
    

def filter_nan_bboxes(bboxes: npt.ArrayLike, isnan: npt.ArrayLike):
    bboxes = np.array(bboxes)
    isnan = np.array(isnan)
    del_idx = []
    for i, bbox in enumerate(bboxes):
        bbox = Bbox(bbox)
        
        j = 0
        while any([isnan[bbox.topl], isnan[bbox.topr], isnan[bbox.botl], isnan[bbox.botr]]) and j < 5:
            #-- check top
            if isnan[bbox.topl] and isnan[bbox.topr]:
                l_nonan = ~isnan[bbox.top:bbox.bot+1, bbox.l]
                r_nonan = ~isnan[bbox.top:bbox.bot+1, bbox.r]
                if not any(l_nonan) and not any(r_nonan):
                    del_idx.append(i)
                    break
                first_true = bbox.top + max(np.argmax(l_nonan), np.argmax(r_nonan))
                bbox.top = first_true
                bbox.update()
            #-- check left
            if isnan[bbox.topl] and isnan[bbox.botl]:
                top_nonan = ~isnan[bbox.top, bbox.l:bbox.r+1]
                bot_nonan = ~isnan[bbox.bot, bbox.l:bbox.r+1]
                if not any(top_nonan) and not any(bot_nonan):
                    del_idx.append(i)
                    break
                first_true = bbox.l + max(np.argmax(top_nonan), np.argmax(bot_nonan))
                bbox.l = first_true
                bbox.update()
            #-- check bottom
            if isnan[bbox.botl] and isnan[bbox.botr]:
                l_nonan = ~isnan[bbox.top:bbox.bot+1, bbox.l]
                r_nonan = ~isnan[bbox.top:bbox.bot+1, bbox.r]
                if not any(l_nonan) and not any(r_nonan):
                    del_idx.append(i)
                    break
                last_true = bbox.top + max(np.argmax(~l_nonan), np.argmax(~r_nonan)) - 1
                bbox.bot = last_true
                bbox.update()
            #-- check right
            if isnan[bbox.topr] and isnan[bbox.botr]:
                top_nonan = ~isnan[bbox.top, bbox.l:bbox.r+1]
                bot_nonan = ~isnan[bbox.bot, bbox.l:bbox.r+1]
                if not any(top_nonan) and not any(bot_nonan):
                    del_idx.append(i)
                    break
                last_true = bbox.l + max(np.argmax(~top_nonan), np.argmax(~bot_nonan)) - 1
                bbox.r = last_true
                bbox.update()
            #-- check top left
            if isnan[bbox.topl]:
                l_nonan = ~isnan[bbox.top:bbox.bot+1, bbox.l]
                top_nonan = ~isnan[bbox.top, bbox.l:bbox.r+1]
                discard_l = np.sum(~l_nonan)
                discard_top = np.sum(~top_nonan)
                if discard_l <= discard_top:
                    bbox.top = bbox.top + np.argmax(l_nonan)
                else:
                    bbox.l = bbox.l + np.argmax(top_nonan)
                bbox.update() 
            #-- check top right
            if isnan[bbox.topr]:
                r_nonan = ~isnan[bbox.top:bbox.bot+1, bbox.r]
                top_nonan = ~isnan[bbox.top, bbox.l:bbox.r+1]
                discard_r = np.sum(~r_nonan)
                discard_top = np.sum(~top_nonan)
                if discard_r <= discard_top:
                    bbox.top = bbox.top + np.argmax(r_nonan)
                else:
                    bbox.r = bbox.l + np.argmax(~top_nonan) - 1
                bbox.update() 
            #-- check bottom left
            if isnan[bbox.botl]:
                l_nonan = ~isnan[bbox.top:bbox.bot+1, bbox.l]
                bot_nonan = ~isnan[bbox.bot, bbox.l:bbox.r+1]
                discard_l = np.sum(~l_nonan)
                discard_bot = np.sum(~bot_nonan)
                if discard_l <= discard_bot:
                    bbox.bot = bbox.top + np.argmax(~l_nonan) - 1
                else:
                    bbox.l = bbox.l + np.argmax(bot_nonan)
                bbox.update()  
            #-- check bottom right
            if isnan[bbox.botr]:
                r_nonan = ~isnan[bbox.top:bbox.bot+1, bbox.r]
                bot_nonan = ~isnan[bbox.bot, bbox.l:bbox.r+1]
                discard_r = np.sum(~r_nonan)
                discard_bot = np.sum(~bot_nonan)
                if discard_r <= discard_bot:
                    bbox.bot = bbox.top + np.argmax(~r_nonan) - 1
                else:
                    bbox.r = bbox.l + np.argmax(~bot_nonan) - 1
                bbox.update()  
            if (bbox.l == bbox.r) or (bbox.top == bbox.bot):
                del_idx.append(i)
                break
            j += 1
            
        bboxes[i] = bbox()
        
    bboxes = np.delete(bboxes, del_idx, 0)
    return bboxes


# =============================================================================
# Classes
# =============================================================================


class Bbox:
    def __init__(self, bbox_coords: tuple):
        self.coords = bbox_coords
        #-- coords = (sa0, li0, sa1, li1)
        (self.l, self.top, self.r, self.bot) = self.coords
        self.update()
        
    def __call__(self):
        return (self.l, self.top, self.r, self.bot) #(sa0, li0, sa1, li1)
        
    def update(self):
        self.topl = (self.top, self.l)
        self.topr = (self.top, self.r)
        self.botl = (self.bot, self.l)
        self.botr = (self.bot, self.r)


class LoggerValue(object):
    """Default LoggerValue object for Logger class. Inspired by [1]."""
    
    def __init__(self, fmt="{global_avg:.6f}"):
        self.fmt = fmt
        self.total = 0.0
        self.count = 0
    
    def update(self, value, n=1):
        self.count += n
        self.total += value
    
    def __call__(self):
        count = max(self.count, 1)
        return self.total / count
    
    def __str__(self):
        return self.fmt.format(global_avg=self())
    

class Logger(object):
    """Default Logger class for network training. Inspired by [1]."""
    
    def __init__(self, delimiter="  "):
        self.meters = defaultdict(LoggerValue)
        self.delimiter = delimiter
        
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if not isinstance(v, (float, int)):
                print(f"Expected value to be of type float or int, but got {type(v)}.", file=sys.stderr)
                sys.exit(1)
            self.meters[k].update(v)
            
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)
    
    
class PerformanceLogger:
    
    def __init__(self, device):
        self.metrics = {
        'prec': torchmetrics.classification.BinaryPrecision().to(device),
        'recall': torchmetrics.classification.BinaryRecall().to(device),
        'spec': torchmetrics.classification.BinarySpecificity().to(device),
        'acc': torchmetrics.classification.BinaryAccuracy().to(device),
        'f1': torchmetrics.classification.BinaryF1Score().to(device),
        'auroc': torchmetrics.classification.BinaryAUROC().to(device)
    }
        
    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
            
    def update(self, output, target):
        for metric in self.metrics.values():
            metric.update(output, target)
            
    def compute(self):
        return {key: metric.compute().item() for key, metric in self.metrics.items()}
        

class PerformanceMetrics:
    
    def __init__(self, from_ckp: str = '', delimiter = '  '):
        #-- Store current performance metrics
        self.current_acc = 0.
        self.current_spec = 0.
        self.current_prec = 0. 
        self.current_recall = 0.
        self.current_f1 = 0.
        self.current_auroc = 0.
        self.current_loss = float('inf')
        self.current_epoch = 0
        #-- Store best performance metrics
        self.best_acc = 0.
        self.best_spec = 0.
        self.best_prec = 0. 
        self.best_recall = 0.
        self.best_f1 = 0.
        self.best_auroc = 0.
        self.best_loss = float('inf')
        #-- Store epoch for best performance metrics
        self.epoch_acc = 0
        self.epoch_spec = 0
        self.epoch_prec = 0
        self.epoch_recall = 0
        self.epoch_f1 = 0
        self.epoch_auroc = 0
        self.epoch_loss = 0
        #-- Store delimiter (for printing)
        self.delimiter = delimiter
        #-- Optinally import checkpoint
        if from_ckp != '':
            self.load_ckp(from_ckp)
        
    def update(self, stats, epoch):
        self.current_acc = stats['acc']
        self.current_spec = stats['spec']
        self.current_prec = stats['prec']
        self.current_recall = stats['recall']
        self.current_f1 = stats['f1']
        self.current_auroc = stats['auroc']
        self.current_loss = stats['loss']
        self.current_epoch = epoch
        
        if self.current_acc > self.best_acc:
            self.best_acc = self.current_acc
            self.epoch_acc = epoch
        if self.current_spec > self.best_spec:
            self.best_spec = self.current_spec
            self.epoch_spec = epoch
        if self.current_prec > self.best_prec:
            self.best_prec = self.current_prec
            self.epoch_prec = epoch
        if self.current_recall > self.best_recall:
            self.best_recall = self.current_recall
            self.epoch_recall = epoch
        if self.current_f1 > self.best_f1:
            self.best_f1 = self.current_f1
            self.epoch_f1 = epoch
        if self.current_auroc > self.best_auroc:
            self.best_auroc = self.current_auroc
            self.epoch_auroc = epoch
        if self.current_loss < self.best_loss:
            self.best_loss = self.current_loss
            self.epoch_loss = epoch
            
    def load_ckp(self, ckp):
        ckp_dict = {}
        epoch_dict = {}
        entries = ckp.split(self.delimiter)
        for entry in entries:
            ckp_dict[entry.split(':')[0]] = float(entry.split(' ')[1][:-1])
            epoch_dict[entry.split(':')[0]] = int(entry.split(' ')[2][1:-1])
        
        self.best_acc = ckp_dict['best_acc']
        self.best_spec = ckp_dict['best_spec']
        self.best_prec = ckp_dict['best_prec']
        self.best_recall = ckp_dict['best_recall']        
        self.epoch_acc = epoch_dict['best_acc']        
        self.epoch_spec = epoch_dict['best_spec']        
        self.epoch_prec = epoch_dict['best_prec']        
        self.epoch_recall = epoch_dict['best_recall']
    
    @property
    def current(self):
        return {
            'recall': self.current_recall,
            'acc': self.current_acc,
            'prec': self.current_prec,
            'spec': self.current_spec,
            'f1': self.current_f1,
            'auroc': self.current_auroc
        }
    
    @property
    def best(self):
        return {
            'recall': self.best_recall,
            'acc': self.best_acc,
            'prec': self.best_prec,
            'spec': self.best_spec,
            'f1': self.best_f1,
            'auroc': self.best_auroc  
        }
    
    @property
    def epoch(self):
        return{    
            'recall': self.epoch_recall,
            'acc': self.epoch_acc,
            'prec': self.epoch_prec,
            'spec': self.epoch_spec,
            'f1': self.epoch_f1,
            'auroc': self.epoch_auroc,
            'loss': self.epoch_loss
        }
    
    @property
    def current_str(self):
        string = []
        string.append(f'epoch: {int(self.current_epoch)}')
        string.extend([f'{k}: {v:.3f}' for (k, v) in self.best.items()])
        string.append(f'loss: {self.current_loss:.6f}')
        return self.delimiter.join(string)
    
    @property
    def best_str(self): 
        string = []
        string.extend([f'{k}: {v:.3f} ({int(self.epoch[k])})' for (k, v) in self.best.items()])
        string.append(f'loss: {self.current_loss:.6f} ({int(self.epoch["loss"])})')
        return self.delimiter.join(string)
    
    @property
    def log(self):
        return{
            'test_loss': f'{self.current_loss:.6f}'[:8],
            **{k: f'{v:.3f}'[:5] for (k, v) in self.current.items()},
        }
        
    def __str__(self):
        str1 = 'Current performance: ' + self.current_str
        str2 = 'Best performance: ' + self.best_str
        return str1 + '\n' + str2 + '\n'
