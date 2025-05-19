# -*- coding: utf-8 -*-
"""
Library for importing event coordinates from iff_eventCoords directory.

@author: niels, 2025
"""


# =============================================================================
# Packages
# =============================================================================


import numpy as np


# =============================================================================
# Classes
# =============================================================================


class EventCoords:
    """EventCoords class storing coordinates outlining events."""
    
    def __init__(self, file_path: str):#, idx: int = None):
        # #-- Read the file contents
        # with open(file_path, "r") as file:
        #     lines = file.readlines()
        # #-- Parse the numeric data, skipping the header
        # data = [list(map(int, line.split())) for line in lines[1:]]
        # if idx is not None: data = data[idx]
        # #-- For more than one event
        # if all(isinstance(element, list) for element in data) is True:
        #     self.li0 = [row[0] for row in data]
        #     self.sa0 = [row[1] for row in data]
        #     self.li1 = [row[2] for row in data]
        #     self.sa1 = [row[3] for row in data]
        # #-- For a single event
        # elif all(isinstance(element, int) for element in data) is True:
        #     self.li0 = data[0]
        #     self.sa0 = data[1]
        #     self.li1 = data[2]
        #     self.sa1 = data[3]
        
        #-- Read the file contents
        self.li0 = []
        self.sa0 = []
        self.li1 = []
        self.sa1 = []
        with open(file_path, "r") as file:
            lines = file.readlines()
        for i, line in enumerate(lines[1:]):
            (li0, sa0, li1, sa1) = line.strip().split()
            self.li0.append(int(li0))
            self.sa0.append(int(sa0))
            self.li1.append(int(li1))
            self.sa1.append(int(sa1))
        self.set_return_fmt()
    
    def __repr__(self):
        return (f"EventCoords(\n   li0={self.li0},\n   sa0={self.sa0},\n   "
                f"li1={self.li1},\n   sa1={self.sa1}\n)")
    
    def __len__(self):
        return len(self.li0)
        
    def __getitem__(self, key):
        return tuple([getattr(self, attr)[key] for attr in self.fmt])
        # return (self.sa0[key], self.li0[key], self.sa1[key], self.li1[key])
    
    @classmethod
    def set_return_fmt(cls, fmt = ['li0','sa0','li1','sa1']):
        assert len(fmt) == 4, 'Wrong number of inputs for fmt.'
        assert all([s in ['li0', 'sa0', 'li1', 'sa1'] for s in fmt]), "Unknown format."
        cls.fmt = fmt
    
    @property
    def attributes(self):
        """Returns class attributes"""
        keys = [key for key in self.__dict__.keys()
                if not key.startswith(('_', '__'))]
        return keys
    
    def to_numpy(self):
        """Returns coordinates as numpy array"""
        return np.column_stack((self.li0, self.sa0, self.li1, self.sa1))