# -*- coding: utf-8 -*-
"""
Library with generic io functions.

@author: niels, 2025
"""


# =============================================================================
# Packages
# =============================================================================


import numpy as np


# =============================================================================
# Functions
# =============================================================================


def read_big_endian_float32(file_path: str, nrows: int, 
                            ncols: int) -> np.ndarray:
    """Read big-endian float32 file."""
    data = np.float32(np.fromfile(file_path, '>f4').reshape(nrows, ncols))
    return data


def read_contours(file_path: str) -> list:
    """
    Read segmentation contours stored in .txt files.

    Parameters
    ----------
    file_path : str
        Path to the .txt file.

    Returns
    -------
    contours : list
        List of (N,2)-np.ndarrays with segmentation contours.

    """
    contours = []
    #-- Read contours. Blank lines in text file indicate a new segment
    with open(file_path, "r") as file:
        segment = []
        for line in file:
            if line.strip():
                segment.append(list(map(float, line.split())))
            else:
                if segment:
                    contours.append(np.array(segment))
                    segment = []
        if segment:
            contours.append(np.array(segment))
    return contours