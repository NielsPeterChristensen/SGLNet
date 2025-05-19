# -*- coding: utf-8 -*-
"""
Library for importing products from iff directory.

@author: niels, 2025
"""


# =============================================================================
# Packages
# =============================================================================


import numpy as np
import SGLNet.PyIPP.ipp_io as ipp_io
import SGLNet.PyIPP.grd as ipp_grd
from pathlib import Path
from PIL import Image
from osgeo import gdal


# =============================================================================
# Classes
# =============================================================================


class Vrt:
    
    def __init__(self, file_path: str):
        self.file_path = str(file_path)
        
    def __call__(self):
        return gdal.Open(self.file_path)
        
    def shape(self):
        return (self().RasterYSize, self().RasterXSize)
        
    def mag(self):
        return self().ReadAsArray()    


# MAYBE DELETE LATER IF UNUSED
# class OldIffDD:
#     """OldIffDD class used to extract data from iff_dd file."""

#     def __init__(self, file_path: str):
#         """
#         Parameters
#         ----------
#         file_path : str
#             string with path to specific iff_dd file. Usually something like 
#             "./track_<xxx>_<pass>/iff_dd/<DATE1>_<DATE2>_<DATE2>_<DATE3>.pha".
#         """
#         file_path = Path(str(file_path)).resolve()
#         if file_path.suffix != '.pha':
#             if file_path.suffix == '.png':
#                 file_path = file_path.with_suffix('')
#             file_path.with_suffix('.pha')
#         self.file_path = file_path
#         self.iff_dd_path = self.file_path.parent
#         self.track_path = self.iff_dd_path.parent
#         self.basename = self.file_path.stem
#         self.iff_path = self.track_path / 'iff'
#         self._get_grd()
#         self.pha = self.pha()
#         self.png = self.png()
    
#     def __repr__(self):
#         return ("Iff object of the Interferogram Formation product" \
#                 + f"{self.basename}")
    
#     @property
#     def attributes(self):
#         """Return class attributes"""
#         keys = [key for key in self.__dict__.keys()
#                 if not key.startswith(('_', '__'))]
#         return keys
    
#     @property
#     def nrows(self):
#         return int(self.Grd.stripInfo.azSamp)
    
#     @property
#     def ncols(self):
#         return int(self.Grd.stripInfo.raSamp)
    
#     @property
#     def shape(self):
#         """Returns shape of the whole iff product (not the event)."""
#         return (self.nrows, self.ncols)
        
#     def _get_grd(self):
#         """
#         Get nrows and ncols from the original iff product where the
#         event is taken from.
#         """
#         #-- Get folders and use first (dimensions are identical for all .grd)
#         iff_folders = [d for d in self.iff_path.iterdir() if d.is_dir()]
#         name = iff_folders[0].name
#         #-- Get path to GRD-file
#         iff_grd_path = self.iff_path / name / f"{name}.grd"
#         self.Grd = ipp_grd.IffGrd(iff_grd_path)
        
#     def _get_slc_paths(self):
#         acq_dates = self.basename.split('_')
#         iffdd_dates12 = acq_dates[:2]
#         iffdd_dates23 = acq_dates[2:]
#         for dir_ in self.iff_path.iterdir():
#             if dir_.is_dir():
#                 iff_dates = [dir_.stem.split('_')[i] for i in [0, 5]]
#                 if iff_dates == iffdd_dates12:
#                     self.iff_dir1 = dir_
#                 if iff_dates == iffdd_dates23:
#                     self.iff_dir2 = dir_
#         self.slc1_path = [path for path in self.iff_dir1.iterdir() if path.suffix == '.slc1'][0]
#         self.slc2_path = [path for path in self.iff_dir1.iterdir() if path.suffix == '.slc2'][0]
#         self.slc3_path = [path for path in self.iff_dir2.iterdir() if path.suffix == '.slc2'][0]
        
#     def pha(self):
#         return ipp_io.read_big_endian_float32(self.file_path, self.nrows, self.ncols)
        
#     def png(self):
#         return Image.open(self.file_path.with_suffix(
#             self.file_path.suffix + '.png')).convert('RGB')
 
#     def mag(self, method: str = 'simple'):
#         self._get_slc_paths()
#         if method in ['simple', 'Simple', 's', 'S']:
#             self.mag_path = [path for path in self.iff_dir1.iterdir() if path.suffix == '.mag'][0]
#             return ipp_io.read_big_endian_float32(self.mag_path, self.nrows, self.ncols)
#         if method in ['geometric', 'Geometric', 'geom', 'Geom', 'g', 'G']:
#             slc1 = ipp_io.read_big_endian_float32(self.slc1_path, self.nrows, self.ncols)
#             slc2 = ipp_io.read_big_endian_float32(self.slc2_path, self.nrows, self.ncols)
#             slc3 = ipp_io.read_big_endian_float32(self.slc3_path, self.nrows, self.ncols)
#             return np.cbrt(slc1 * slc2 * slc3)
#         if method in ['harmonic', 'Harmonic', 'harm', 'Harm', 'h', 'H']:
#             slc1 = ipp_io.read_big_endian_float32(self.slc1_path, self.nrows, self.ncols)
#             slc2 = ipp_io.read_big_endian_float32(self.slc2_path, self.nrows, self.ncols)
#             slc3 = ipp_io.read_big_endian_float32(self.slc3_path, self.nrows, self.ncols)
#             return 3 / (1/slc1 + 1/slc2+ 1/slc3)
#         if method in ['interferometric', 'Interferometric', 'inter', 'Inter', 'i', 'I']:
#             slc1 = ipp_io.read_big_endian_float32(self.slc1_path, self.nrows, self.ncols)
#             slc2 = ipp_io.read_big_endian_float32(self.slc2_path, self.nrows, self.ncols)
#             slc3 = ipp_io.read_big_endian_float32(self.slc3_path, self.nrows, self.ncols)
#             return (slc1 * slc3) / slc2
#         raise ValueError(f"{method=} not recognized.")
        

class IffDD:
    """IffDD class used to extract data from iff_dd file."""

    def __init__(self, file_path: str):
        """
        Parameters
        ----------
        file_path : str
            string with path to specific iff_dd file. Usually something like 
            "./track_<xxx>_<pass>/iff_dd/<DATE1>_<DATE2>_<DATE2>_<DATE3>.pha".
        """
        #-- Convert to <class Path> and remove suffix(es)
        from SGLNet.Corefunc.utils import remove_extensions
        file_path = remove_extensions(Path(str(file_path)).resolve())
        self.file_path = file_path
        self.file_name = file_path.stem
        self.iffdd_dir = file_path.parent
        self.track_dir = file_path.parent.parent
        self.vrt = Vrt(file_path.with_suffix('.vrt'))
        self.shape = self.vrt.shape()
    
    def __repr__(self):
        return ("Iff object of the Interferogram Formation product" \
                + f"{self.file_name}")  
            
    def png(self):
        return Image.open(self.file_path.with_suffix('.pha.png')).convert('RGB')
    
    def pha(self):
        return ipp_io.read_big_endian_float32(self.file_path.with_suffix('.pha'), 
                           nrows=self.shape[0], ncols=self.shape[1])
    
    def mag(self):
        return self.vrt.mag()
            

if __name__ == "__main__":
    a = IffDD(r"D:\dtu\speciale\ipp\processed\track_002_ascending\iff_dd\20171112_20171124_20171124_20171206.pha")
    
    