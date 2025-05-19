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


# =============================================================================
# Classes
# =============================================================================


class IffProduct():
    """Default nested class used in Iff."""
    
    def __init__(self, IffObj, extension: str):
        """
        Parameters
        ----------
        IffObj : Iff
            Instance of Iff where the product is stored as a nested class.
        extension : str
            File extension for the product of interest.
        """
        self._IffObj = IffObj
        self.extension = extension
        if extension == '.slc1':
            grd_attr = 'imgNaN_1'
        elif extension == '.slc2':
            grd_attr = 'imgNaN_2'
        else:
            grd_attr = self.extension[1:] + 'NaN'
        self.NaN = getattr(self._IffObj.Grd.fileDesc, grd_attr, None)
        path = self._IffObj.product_path / f"{self._IffObj.basename}{self.extension}"
        data = ipp_io.read_big_endian_float32(path, self._IffObj.nrows, self._IffObj.ncols)
        data = np.float32(data)
        data[data == self.NaN] = np.nan
        self.data = data
        del data, path
        
    @property
    def attributes(self):
        """Returns class attributes"""
        keys = [key for key in self.__dict__.keys()
                if not key.startswith(('_', '__'))]
        return keys
    
    @property
    def shape(self):
        "Returns dimensions of the product"
        return (self.event_nrows, self.event_ncols)


class Pha(IffProduct):
    """Define nested class Pha with inheritance from IffProduct."""
    
    def __init__(self, IffObj):
        """
        Parameters
        ----------
        IffObj : Iff
            Reference back to instance of Iff (self).
        """
        super().__init__(IffObj, '.pha')
        
    def __repr__(self):
        return ("Phase nested object.")
        
    
class Mag(IffProduct):
    """Define nested class Mag with inheritance from IffProduct."""
    
    def __init__(self, IffObj):
        """
        Parameters
        ----------
        IffObj : Iff
            Reference back to instance of Iff (self).
        """
        super().__init__(IffObj, '.mag')
        
    def __repr__(self):
        return ("Magnitude nested object.")
    
    
class Cor(IffProduct):
    """Define nested class Cor with inheritance from IffProduct."""
    
    def __init__(self, IffObj):
        """
        Parameters
        ----------
        IffObj : Iff
            Reference back to instance of Iff (self).
        """
        super().__init__(IffObj, '.cor')
        
    def __repr__(self):
        return ("Coherence nested object.")
    
    
class Slc1(IffProduct):
    """Define nested class Slc1 with inheritance from IffProduct."""
    
    def __init__(self, IffObj):
        """
        Parameters
        ----------
        IffObj : Iff
            Reference back to instance of Iff (self).
        """
        super().__init__(IffObj, '.slc1')
        
    def __repr__(self):
        return ("SLC1 nested object.")
    
    
class Slc2(IffProduct):
    """Define nested class Slc2 with inheritance from IffProduct."""
    
    def __init__(self, IffObj):
        """
        Parameters
        ----------
        IffObj : Iff
            Reference back to instance of Iff (self).
        """
        super().__init__(IffObj, '.slc2')
        
    def __repr__(self):
        return ("SLC2 nested object.")


class Iff:
    """Iff class used to extract data from iff folder."""

    def __init__(self, dir_path: str):
        """
        Parameters
        ----------
        dir_path : str
            string with path to specific iff directory. Usually something
            like "./track_<xxx>_<pass>/iff/<SLC1>_<SLC2>"
        """
        dir_path = Path(dir_path).resolve()
        if dir_path.name == "":
            dir_path = dir_path.parent
        self.product_path = dir_path
        self.iff_path = self.product_path.parent
        self.track_path = self.iff_path.parent
        self.basename = self.product_path.stem
        self._get_grd()
        self.Pha = Pha(self)
        self.Mag = Mag(self)
        self.Cor = Cor(self)
        self.Slc1 = Slc1(self)
        self.Slc2 = Slc2(self)
    
    def __repr__(self):
        return ("Iff object of the Interferogram Formation product" \
                + f"{self.basename}")
    
    @property
    def attributes(self):
        """Return class attributes"""
        keys = [key for key in self.__dict__.keys()
                if not key.startswith(('_', '__'))]
        return keys
    
    @property
    def nrows(self):
        return int(self.Grd.stripInfo.azSamp)
    
    @property
    def ncols(self):
        return int(self.Grd.stripInfo.raSamp)
    
    @property
    def shape(self):
        """Returns shape of the whole iff product (not the event)."""
        return (self.nrows, self.ncols)
    
    # def _get_dd(self, dd_dir_path):
    #     if dd_dir_path is not None:
    #         dd_dir_path = Path(dd_dir_path)
    #         if dd_dir_path.is_dir():
    #             print(f'{dd_dir_path} not a valid directory.\n' \
    #                   + 'Continuing without loading dd_files.\n')
    #     else:
    #         if (self.track_path / 'iff_dd').is_dir():
    #             dd_dir_path = self.track_path / 'iff_dd'
    #     self.dd_dir_path = dd_dir_path
        
    #     if (self.dd_dir_path / ''
        
    def _get_grd(self):
        """
        Get nrows and ncols from the original iff product where the
        event is taken from.
        """
        #-- Get folders and use first (dimensions are identical for all .grd)
        iff_folders = [d for d in self.iff_path.iterdir() if d.is_dir()]
        name = iff_folders[0].name
        #-- Get path to GRD-file
        iff_grd_path = self.iff_path / name / f"{name}.grd"
        self.Grd = ipp_grd.IffGrd(iff_grd_path)