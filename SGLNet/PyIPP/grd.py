# -*- coding: utf-8 -*-
"""
Library for importing .grd files from the iff and gim_events directories.

@author: niels, 2025
"""


# =============================================================================
# Classes
# =============================================================================


class Section:
    """Section class used within GrdFile class."""
    
    def __init__(self):
        self._data = {}

    def __getattr__(self, key: str):
        if key in self._data:
            return self._data[key]
        raise AttributeError(f"'Section' object has no attribute '{key}'")

    def __setattr__(self, key: str, value):
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __repr__(self):
        return f"Section({self._data})"

    def __dir__(self):
        return list(self._data.keys())  # Return only keys for introspection
    
    def keys(self):
        return list(self._data.keys())  # Return only keys for introspection


class GrdFile:
    """GrdFile class used to store contents of GRD-files."""
    
    def __init__(self, file_path: str):
        self._sections = {}
        #-- Read the file contents
        with open(file_path, "r") as file:
            lines = file.readlines()
        #-- Parse the contents into the grid instance
        current_section = None
        for line in lines:
            line = line.strip()  # Remove leading/trailing whitespace
            if line.startswith("[") and line.endswith("]"):  # New section
                current_section = line.strip("[]")
                self.add_section(current_section)
            elif "=" in line and current_section:  # Key-value pair
                key, value = map(str.strip, line.split("=", 1))
                setattr(self._sections[current_section], key, value)

    def __getattr__(self, section_name: str):
        if section_name in self._sections:
            return self._sections[section_name]
        raise AttributeError(
            f"'BaseGrid' object has no section '{section_name}'")

    def __setattr__(self, section_name: str, value):
        if section_name == "_sections":
            super().__setattr__(section_name, value)
        else:
            self._sections[section_name] = value

    def add_section(self, section_name: str):
        if section_name not in self._sections:
            self._sections[section_name] = Section()
    
    def keys(self):
        return list(self._sections.keys())
    
    def get_dict(self):
        return {
            section: vars(self._sections[section]) for section in dir(self)
        }


class IffGrd(GrdFile):
    """IffGrd class used to store iff-related GRD-files.
    Inheritance from GrdFile class.
    """
    
    def __init__(self, file_path: str):
        super().__init__(file_path)
    
    def __repr__(self):
        return f"IffGrd({list(self._sections.keys())})"
    

class GimGrd(GrdFile):
    """GimGrd class used to store gim-related GRD-files.
    Has inheritance from GrdFile class.
    """
    
    def __init__(self, file_path: str):
        super().__init__(file_path)
    
    def __repr__(self):
        return f"GimGrd({list(self._sections.keys())})"
    
class SegGrd(GrdFile):
    """SegGrd class used to store segmentation-related GRD-files.
    Has inheritance from GrdFile class.
    """
    
    def __init__(self, file_path: str):
        super().__init__(file_path)
    
    def __repr__(self):
        return f"SegGrd({list(self._sections.keys())})"