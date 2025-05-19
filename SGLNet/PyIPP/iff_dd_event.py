# -*- coding: utf-8 -*-
"""
Library for importing products from iff directory.

@author: niels, 2025
"""


# =============================================================================
# Packages
# =============================================================================


from pathlib import Path
import sys
import SGLNet.PyIPP.event_coords as ipp_event_coords
import SGLNet.PyIPP.iff_dd as ipp_iff_dd
import SGLNet.Plotting.plot_utils as plot_utils


# =============================================================================
# Classes
# =============================================================================


def event_exist(path : Path):
    if not path.is_dir():
        print(f"Unable to find eventCoords txt files at location: {str(path)}")
        print("For non-default paths, please provide the path manually.")
        sys.exit()


class IffDDEvent(ipp_iff_dd.IffDD):
    """IffDD class used to extract data from iff_dd file."""

    def __init__(self, file_path: str, event_dir: str = None):
        """
        Parameters
        ----------
        file_path : str
            string with path to specific iff_dd file. Usually something like 
            "./track_<xxx>_<pass>/iff_dd/<DATE1>_<DATE2>_<DATE2>_<DATE3>.pha".
        event_dir: str, optional
            String with path to directory where event coordinate .txt files
            are located. Default is None, meaning the directory is assumed
            to be located in "./track_<xxx>_<pass>/iff_eventCoords"
        """
        super().__init__(file_path)
        file_path = Path(file_path).resolve()
        self.event_path = (self.track_path / "iff_eventCoords" / self.basename).with_suffix(".txt")
        self.EventCoords = ipp_event_coords.EventCoords(self.event_path)
        
    def plot(self):
        plot_utils.plot_dd_pha_with_events(self)
        

if __name__ == "__main__":
    a = IffDDEvent(r"D:\dtu\speciale\ipp\processed\track_002_ascending\iff_dd\20171112_20171124_20171124_20171206.pha")