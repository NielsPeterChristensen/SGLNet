# -*- coding: utf-8 -*-
"""
Library for importing products from gim_events directory.
It is assumed that the products are given in radar coordinates (pre-geocoding).

@author: niels, 2025
"""


# =============================================================================
# Packages
# =============================================================================


import numpy as np
import os
import SGLNet.PyIPP.event_coords as ipp_event_coords
import SGLNet.PyIPP.ipp_io as ipp_io
import SGLNet.PyIPP.grd as ipp_grd


# =============================================================================
# Classes
# =============================================================================


class EventProduct():
    """Default nested class used in GimEvent."""
    
    def __init__(self, EventObj, extension: str):
        """
        Parameters
        ----------
        EventObj : GimEvent
            Instance of GimEvent where the product is stored as a nested class.
        extension : str
            File extension for the product of interest.
        """
        self._EventObj = EventObj
        self.extension = extension
        event_coord_path= os.path.join(self._EventObj.EventCoords_path, self._EventObj.date + '.txt')
        event_idx = self._EventObj.nr_event
        self.eventcoords = ipp_event_coords.EventCoords(event_coord_path)#, idx=event_coord_idx)
        (self.li0, self.sa0, self.li1, self.sa1) = self.eventcoords[event_idx]
        self.event_nrows = self.li1 - self.li0
        self.event_ncols = self.sa1 - self.sa0
        self.name = self._EventObj.basename
        path = os.path.join(self._EventObj.event_path, self.name + self.extension)
        data = ipp_io.read_big_endian_float32(path, self.event_nrows, self._EventObj.ncols)
        data = np.float32(data)
        data = data[:,self.sa0:self.sa1]
        data[data == self._EventObj.NaN] = np.nan
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


class GimEvent:
    """GimEvents class used to extract data from gim_events folder."""

    def __init__(self, dir_path: str):
        """
        Parameters
        ----------
        dir_path : str
            string with path to specific event directory. Usually something
            like "./track_<xxx>_<pass>/gim_events/<iff_dd_interval>_<#event>"
        """
        if dir_path.endswith(os.sep):
            dir_path = os.path.dirname(dir_path)
        self.event_path = dir_path
        self.gim_path = os.path.dirname(self.event_path)
        self.track_path = os.path.dirname(self.gim_path)
        self.iff_path = os.path.join(self.track_path, 'iff')
        self.segmentation_path = os.path.join(self.track_path, 
                                              'event_segmentation')
        self.EventCoords_path = os.path.join(self.track_path, 
                                              'iff_EventCoords')
        self.basename = os.path.basename(self.event_path)
        (self.date, self.nr_event) = self.basename.rsplit('_', 1)
        self.nr_event = int(self.nr_event)
        self._get_nrows_ncols()
        self._get_nan()
        self.Phu = self.Phu(self)
        self.Phr = self.Phr(self)
        self.Lat = self.Lat(self)
        self.Lon = self.Lon(self)
    
    def __repr__(self):
        return (f"GimEvent object of event {self.basename}")
    
    @property
    def attributes(self):
        """Return class attributes"""
        keys = [key for key in self.__dict__.keys()
                if not key.startswith(('_', '__'))]
        return keys
    
    @property
    def shape(self):
        """Returns shape of the whole iff product (not the event)."""
        return (self.nrows, self.ncols)
        
    def _get_nrows_ncols(self):
        """
        Get nrows and ncols from the original iff product where the
        event is taken from.
        """
        #-- Get folders and use first (dimensions are identical for all .grd)
        iff_folders = [d for d in os.listdir(self.iff_path)
                       if os.path.isdir(os.path.join(self.iff_path, d))]
        name = iff_folders[0]
        #-- Get path to GRD-file
        iff_grd_path = os.path.join(os.path.join(self.iff_path, name),
                                name + '.grd')
        # grd = read_grd(grd_path, 'iff')
        IffGrdObj = ipp_grd.IffGrd(iff_grd_path)
        self.nrows = int(IffGrdObj.stripInfo.azSamp)
        self.ncols = int(IffGrdObj.stripInfo.raSamp)
        
    def _get_nan(self):
        
        gim_grd_path = os.path.join(os.path.join(self.gim_path, self.basename),
                                self.basename + '.grd')
        GimGrdObj = ipp_grd.GimGrd(gim_grd_path)
        self.NaN = float(GimGrdObj.fileDesc.NaN)
    
    class Phu(EventProduct):
        """Define nested class Phu with inheritance from EventProduct."""
        
        def __init__(self, EventObj):
            """
            Parameters
            ----------
            EventObj : GimEvent
                Reference back to instance of GimEvent (self).
            """
            super().__init__(EventObj, '.1.phu')
            
        def __repr__(self):
            return ("Unwrapped phase nested object.")
            
    class Phr(EventProduct):
        """Define nested class Phr with inheritance from EventProduct."""
        
        def __init__(self, EventObj):
            """
            Parameters
            ----------
            EventObj : GimEvent
                Reference back to instance of GimEvent (self).
            """
            super().__init__(EventObj, '.1.phr')
            
        def __repr__(self):
            return ("Wrapped phase nested object.")
        
    class Lat(EventProduct):
        
        def __init__(self, EventObj):
            super().__init__(EventObj, '.lat')
        
    class Lon(EventProduct):
        
        def __init__(self, EventObj):
            super().__init__(EventObj, '.lon')