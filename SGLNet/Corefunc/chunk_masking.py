# =============================================================================
# Packages
# =============================================================================

import numpy as np
import numpy.typing as npt
from pathlib import Path
from PIL import Image

import SGLNet.Corefunc.chunk_loader as chunk_loader
import SGLNet.PyIPP.event_coords as ipp_event_coords


# =============================================================================
# Constants
# =============================================================================


POSITIVE: int = 255
AMBIGUOUS: int = 127
NEGATIVE: int = 0


# =============================================================================
# Functions
# =============================================================================


def does_chunk_overlap_EventCoords(bboxes: npt.ArrayLike, EventCoords: ipp_event_coords.EventCoords, margin: int = 0):
    """Checks if the given chunk (x_start, y_start, x_end, y_end) overlaps any bounding box."""
    bboxes = np.array(bboxes)
    aslist = True
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape((-1,2))
        aslist = False
    if bboxes.ndim != 2:
        raise ValueError(f"Wrong dimensionality of bboxes with dimensions {bboxes.ndim}.")
    overlap = []
    for bbox in bboxes:
        x_start, y_start, x_end, y_end = bbox
        for by_min, by_max, bx_min, bx_max in zip(EventCoords.li0, EventCoords.li1, 
                                                  EventCoords.sa0, EventCoords.sa1):
            overlap_x = not (x_end <= (bx_min+margin) or x_start >= (bx_max-margin))
            overlap_y = not (y_end <= (by_min+margin) or y_start >= (by_max-margin))
        if overlap_x and overlap_y:
            overlap.append(True)
        else:
            overlap.append(False)
    if aslist:
        return overlap
    else:
        return overlap[0]
    
    
def does_chunk_overlap_segmentation(bboxes: npt.ArrayLike, label: np.ndarray) -> np.ndarray:
    """Uses a 25% margin to determine if bbox overlap label mask."""
    
    def check_center_mask(bbox):
        x0, y0, x1, y1 = bbox
        center_x0 = int((x0 + x1) / 2 - (x1 - x0) / 4)
        center_y0 = int((y0 + y1) / 2 - (y1 - y0) / 4)
        center_x1 = int((x0 + x1) / 2 + (x1 - x0) / 4)
        center_y1 = int((y0 + y1) / 2 + (y1 - y0) / 4)
        center_region = label[center_y0:center_y1, center_x0:center_x1]
        return np.any(center_region)
    
    bboxes = np.array(bboxes)
    aslist = True
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape((-1,2))
        aslist = False
    if bboxes.ndim != 2:
        raise ValueError(f"Wrong dimensionality of bboxes with dimensions {bboxes.ndim}.")
    overlap = [check_center_mask(bbox) for bbox in bboxes]
    if aslist:
        return overlap
    else:
        return overlap[0]
    
    
def does_chunk_overlap_segmentation_legacy(bboxes: npt.ArrayLike, label: np.ndarray, pxcount: int = 2500) -> np.ndarray:
    """Uses a fixed pixel count to determine if bbox overlap label mask."""
    bboxes = np.array(bboxes)
    aslist = True
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape((-1,2))
        aslist = False
    if bboxes.ndim != 2:
        raise ValueError(f"Wrong dimensionality of bboxes with dimensions {bboxes.ndim}.")
    overlap = []
    crop = np.stack([label[bbox[1]:bbox[3], bbox[0]:bbox[2]] for bbox in bboxes])
    count = np.count_nonzero(crop, axis=(1,2))
    overlap = count >= pxcount
    if aslist:
        return overlap
    else:
        return overlap[0]
    
    
def does_mask_overlap_chunk_center(bboxes: npt.ArrayLike, mask: np.ndarray) -> np.ndarray:
    
    def check_center(bbox):
        x0, y0, x1, y1 = bbox
        center_x0 = int((x0 + x1) / 2 - (x1 - x0) / 4)
        center_y0 = int((y0 + y1) / 2 - (y1 - y0) / 4)
        center_x1 = int((x0 + x1) / 2 + (x1 - x0) / 4)
        center_y1 = int((y0 + y1) / 2 + (y1 - y0) / 4)
        center_region = mask[center_y0:center_y1, center_x0:center_x1]
        return np.any(center_region)
    
    bboxes = np.array(bboxes)
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape((-1,2))
    if bboxes.ndim != 2:
        raise ValueError(f"Wrong dimensionality of bboxes with dimensions {bboxes.ndim}.")
    overlap = [check_center(bbox) for bbox in bboxes]
    return np.array(overlap)

def does_mask_overlap_chunk_center_frac(
    bboxes: npt.ArrayLike, 
    mask: np.ndarray, 
    center_frac: float = 0.5
) -> np.ndarray:
    """
    Check if a mask overlaps the central region of each bbox.

    Parameters:
    - bboxes: array-like of shape (N, 4) with [x0, y0, x1, y1]
    - mask: 2D boolean numpy array
    - center_frac: float between 0 and 1 indicating how much of the bbox
                   is considered the central region (default is 0.5)

    Returns:
    - np.ndarray of bools indicating overlap for each bbox
    """

    if not (0 < center_frac <= 1):
        raise ValueError("center_frac must be in the range (0, 1].")
    
    def check_center(bbox):
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        half_width = (width * center_frac) / 2
        half_height = (height * center_frac) / 2
        center_x0 = int(cx - half_width)
        center_y0 = int(cy - half_height)
        center_x1 = int(cx + half_width)
        center_y1 = int(cy + half_height)
        # Clip to mask bounds
        center_x0 = max(0, center_x0)
        center_y0 = max(0, center_y0)
        center_x1 = min(mask.shape[1], center_x1)
        center_y1 = min(mask.shape[0], center_y1)
        center_region = mask[center_y0:center_y1, center_x0:center_x1]
        return np.any(center_region)
    
    bboxes = np.array(bboxes)
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape((1, 4))
    if bboxes.shape[1] != 4:
        raise ValueError(f"Expected bboxes shape (N, 4), got {bboxes.shape}")
    
    overlap = [check_center(bbox) for bbox in bboxes]
    return np.array(overlap)


def does_mask_overlap_chunk(bboxes: npt.ArrayLike, mask: np.ndarray) -> np.ndarray:
    
    def check_whole(bbox):
        x0, y0, x1, y1 = bbox
        whole_region = mask[y0:y1, x0:x1]
        return np.any(whole_region)
    
    bboxes = np.array(bboxes)
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape((-1,2))
    if bboxes.ndim != 2:
        raise ValueError(f"Wrong dimensionality of bboxes with dimensions {bboxes.ndim}.")
    overlap = [check_whole(bbox) for bbox in bboxes]
    return np.array(overlap)


def chunk_decision(Loader: chunk_loader.ImageChunkLoader, 
                   lake_mask: np.ndarray, 
                   gl_mask: np.ndarray,
                   ambiguous_chunk_action: int = -1,
                   offcenter_chunk_action: int = -1) -> tuple[np.ndarray[bool], np.ndarray[bool]]:

    positive_mask = lake_mask == POSITIVE
    ambiguous_mask = lake_mask == AMBIGUOUS
    exclude_mask = gl_mask > 0
    
    bboxes = Loader.all_bboxes
    img = np.array(Loader().convert("L"), dtype=np.int8)
    
    lake_in_chunk = does_mask_overlap_chunk(bboxes, positive_mask)
    lake_in_chunk_center = does_mask_overlap_chunk_center(bboxes, positive_mask)
    amb_in_chunk = does_mask_overlap_chunk(bboxes, ambiguous_mask)
    # gl_in_chunk_center = does_mask_overlap_chunk_center(bboxes, exclude_mask)
    gl_in_chunk = does_mask_overlap_chunk(bboxes, exclude_mask)
    
    non_blank_chunk = does_mask_overlap_chunk(bboxes, img)
    
    #-- (-1: exclude   0: nolake   1: lake)
    ind = np.zeros(lake_in_chunk_center.shape, dtype=int)
    ind[amb_in_chunk] = ambiguous_chunk_action
    ind[lake_in_chunk] = offcenter_chunk_action
    ind[lake_in_chunk_center] = 1
    # ind[gl_in_chunk_center] = -1
    ind[gl_in_chunk] = -1
    
    valid_chunks = (ind > -1) & non_blank_chunk
    lake_chunks = ind == 1
    
    return valid_chunks, lake_chunks


# =============================================================================
# Classes
# =============================================================================

def action_flag(flag):
    """Bool flag for argparse."""
    
    import argparse
    
    EXCLUDE = {"exclude", "-1"}
    TRUE = {"true", "1",}
    FALSE = {"false", "0"}
    if flag.lower() in EXCLUDE:
        return -1
    if flag.lower() in TRUE:
        return 1
    if flag.lower() in FALSE:
        return 0
    else:
        raise argparse.ArgumentTypeError(f"Unknown input {flag} for action_flag.")