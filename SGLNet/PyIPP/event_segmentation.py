# -*- coding: utf-8 -*-
"""
Library for importing event coordinates from iff_eventCoords directory.

@author: niels, 2025
"""


# =============================================================================
# Packages
# =============================================================================


import SGLNet.PyIPP.ipp_io as ipp_io
import SGLNet.PyIPP.grd as ipp_grd


# =============================================================================
# Classes
# =============================================================================


class EventSegmentation:
    
    def __init__(self, basename_path: str):
        self.Grd = ipp_grd.SegGrd(basename_path + '.grd')
        self.mask = ipp_io.read_big_endian_float32(basename_path + '.seg', int(self.Grd.eventInfo.nrows), int(self.Grd.eventInfo.ncols))
        self.contours = ipp_io.read_contours(basename_path + '.txt')
    
    def __repr__(self):
        pass
    