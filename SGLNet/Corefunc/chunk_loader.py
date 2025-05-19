# =============================================================================
# Docstring
# =============================================================================


"""
chunk_loader.py contains custom made loader classes that takes double
difference interferograms and divides them into chunks. These classes are
used to iterate over subsets of these image chunks.

:Authors
    NPKC / 28-01-2025 / creation / s203980@dtu.dk

:Todo
    Add more comments

:References
    None

:Note:
    None
"""


# =============================================================================
# Packages
# =============================================================================


from PIL import Image, ImageOps
from typing import Union
import numpy as np
from pathlib import Path
import sys

import SGLNet.PyIPP.iff_dd as ipp_iff_dd
import SGLNet.PyIPP.event_coords as ipp_event_coords
import SGLNet.Corefunc.utils as utils

import torch
from torchvision.transforms import Compose



# =============================================================================
# Constants
# =============================================================================


VALID_DOWNSCALE_FACTORS = [1, 2, 4]
VALID_FORM = ['image', 'phase', 'recta', 'polar', 'phase2', 'recta2', 'polar2']


# =============================================================================
# Classes
# =============================================================================
    

class ImageChunkLoader:
    
    def __init__(self, image_path: str, chunk_size: int = 224, overlap: bool = True, 
                 downscale_factor: int = 1, form: str = 'phase'):
        self.chunk_size = int(chunk_size)
        self.overlap = bool(overlap)
        self.downscale_factor = int(downscale_factor)
        self.form = str(form)
        self.stride = int(self.chunk_size / 2**int(self.overlap))
        
        image_path = utils.remove_extensions(Path(image_path).resolve())
        self.image_path = image_path
        self.IffDD = ipp_iff_dd.IffDD(image_path)
        if form not in VALID_FORM:
            raise ValueError(f"{form=} not found amongst valid inputs {VALID_FORM}.")
        if form == 'image':
            self.image = self.IffDD.png()
        if form == 'phase':
            self.image = utils.phase_image(self.IffDD.pha())
        if form == 'recta':
            self.image = utils.recta_image(self.IffDD.pha())
        if form == 'polar':
            self.image = utils.polar_image(self.IffDD.pha(), self.IffDD.mag())
        if form == 'phase2':
            self.image = utils.phase2_image(self.IffDD.pha())
        if form == 'recta2':
            self.image = utils.recta2_image(self.IffDD.pha())
        if form == 'polar2':
            self.image = utils.polar2_image(self.IffDD.pha(), self.IffDD.mag())
        if self.chunk_size % self.stride != 0:
            raise ValueError(
                "chunk_size must be divisible by 2 when {overlap=}, " \
                + f"but {chunk_size=} is not.")
        if downscale_factor not in VALID_DOWNSCALE_FACTORS:
            raise ValueError(
                f"downscale of {downscale_factor} not amongst " \
                + f"valid inputs {str(VALID_DOWNSCALE_FACTORS)}.")
        self.downscale_factor = int(downscale_factor)
        self._calc_bboxes()

    def __len__(self):
        """Behaviour of len()."""
        return len(self.bboxes)
    
    def __getitem__(self, index: Union[int, list]):
        """Enable indexing (obj[index])."""
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for list of length {len(self)}")
        return self(self.bboxes[index])
    
    def __iter__(self):
        """Returns the iterator object itself."""
        self._iter_idx = 0
        return self
    
    def __next__(self):
        """Returns the next image."""
        if self._iter_idx < len(self):
            self._iter_idx += 1
            return self[self._iter_idx-1]
        else:
            raise StopIteration
        
    def __call__(self, bbox: tuple = None):
        """Define behavior when calling function."""
        if bbox is None:
            return self.image.copy()
        if not isinstance(bbox, tuple):
            try:
                bbox = tuple(bbox)
            except:
                raise TypeError(f"Expected input <class 'tuple'> but got {type(bbox)}")
        if not bbox in self.bboxes:
            print(f"Warning: {str(bbox)} is not a default bbox.")
            print("To extract default crops use indexing or examine bboxes attribute.")
            print("If you ment to use a custom crop, just ignore this warning.")
        return self.image.crop(bbox).copy()#.resize(self.bbox_size, resample=Image.LANCZOS)
    
    def index(self, bbox: tuple):
        """Extract index of given bbox."""
        if isinstance(bbox, tuple):
            return self.bboxes.index(bbox)
        raise TypeError(f"{str(bbox)} is not of type tuple")
        
    def event_coords(self, event_dir: str = None) -> ipp_event_coords.EventCoords:        
        path = event_dir or (self.image_path.parent.parent / 'iff_eventCoords' / self.image_path.stem).with_suffix('.txt')
        return ipp_event_coords.EventCoords(path)
    
    def apply_masks(self, path_to_lake_mask: str, path_to_gl_mask: str, ambiguous_chunk_action: int = -1, offcenter_chunk_action: int = -1):
        import SGLNet.Corefunc.chunk_masking as chunk_masking
        self.path_to_lake_mask = path_to_lake_mask
        self.path_to_gl_mask = path_to_gl_mask    
        self.lake_mask = np.array(Image.open(self.path_to_lake_mask).convert('L'))
        self.gl_mask = np.array(Image.open(self.path_to_gl_mask).convert('L'))
        self.ambiguous_chunk_action = ambiguous_chunk_action
        self.offcenter_chunk_action = offcenter_chunk_action
        
        y_pad_lake = self.shape[0]-self.lake_mask.shape[0]
        x_pad_lake = self.shape[1]-self.lake_mask.shape[1]
        y_pad_gl = self.shape[0]-self.gl_mask.shape[0]
        x_pad_gl = self.shape[1]-self.gl_mask.shape[1]
        self.lake_pad = ((0, y_pad_lake), (0, x_pad_lake))
        self.gl_pad = ((0, y_pad_gl), (0, x_pad_gl))
        self.lake_mask = np.pad(self.lake_mask, self.lake_pad, mode='constant', constant_values=0)
        self.gl_mask = np.pad(self.gl_mask, self.gl_pad, mode='constant', constant_values=0)
        
        self.valid_chunks, self.is_lake = chunk_masking.chunk_decision(self, self.lake_mask, self.gl_mask, self.ambiguous_chunk_action, self.offcenter_chunk_action)
        self.all_lake = self.is_lake
        self.is_lake = self.is_lake[self.valid_chunks]
        self.excluded_bboxes = self.all_bboxes[~self.valid_chunks]
        self.bboxes = self.all_bboxes[self.valid_chunks]
        
    def get_mask(self, bbox: tuple = None):
        if not hasattr(self, 'lake_mask') and not hasattr(self, 'gl_mask'):
            print('method "get_mask" cannot be used before "apply_masks" has' \
                  'been called.', file=sys.stderr)
            sys.exit(1)
        if bbox is not None:
            lake = self.lake_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            gl = self.gl_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        else:
            lake = self.lake_mask
            gl = self.gl_mask
        lake[gl > 0] = 64
        return lake
        
    def _calc_bboxes(self):
        """Pad image and compute bboxes"""
        X, Y = self.image.size
        N = self.chunk_size * self.downscale_factor
        M = self.stride * self.downscale_factor
        x_pad = (N - X % N) % N
        y_pad = (N - Y % N) % N
        X_pad = X + x_pad
        Y_pad = Y + y_pad
        self.padded_image = ImageOps.expand(self.image, (0, 0, x_pad, y_pad), 
                                            fill=127)
        self.shape = self.padded_image.size[::-1]
        self.bbox_size = (N, N)
        
        bboxes = []
        for j in range(0, Y_pad-int(N/2), M):
            for i in range(0, X_pad-int(N/2), M):
                bbox = (i, j, i + N, j + N) # (sa0, li0, sa1, li1)
                bboxes.append(bbox)
        self.bboxes = np.array(bboxes)
        self.all_bboxes = self.bboxes
        
        
class BatchImageChunkLoader(ImageChunkLoader):
    
    def __init__(self, image_path: str, chunk_size: int = 224, overlap: bool = True, 
                 downscale_factor: int = 1, form: str = 'phase', batch_size: int = 64):
        super().__init__(image_path, chunk_size, overlap, downscale_factor, form)
        self.batch_size = int(min(batch_size, len(self)))
        
    def __iter__(self):
        """Returns the iterator object itself."""
        # _iter_idx controls the iterator
        # _batch_idx and _interbatch_idx are includes for developer convenience.
        self._iter_idx = 0
        self._batch_idx = 0
        self._interbatch_idx = None
        return self
    
    def __next__(self):
        """Returns the next image."""
        L = len(self) # Total number of images to loop over
        B = self.batch_size
        if self._iter_idx < L:
            start_idx = self._iter_idx
            end_idx = min(L, start_idx + B)
            batch = [self[i] for i in range(start_idx, end_idx)]
            self._iter_idx = end_idx
            self._batch_idx += 1
            self._interbatch_idx = list(range(start_idx, end_idx))
            return batch
        else:
            raise StopIteration
            

class TensorChunkLoader(BatchImageChunkLoader):
    
    def __init__(self, image_path: str, transform: Compose, chunk_size: int = 224, overlap: bool = True, downscale_factor: int = 1, form: str = 'phase', batch_size: int = 64):
        super().__init__(image_path, chunk_size, overlap, downscale_factor, form, batch_size)
        self.transform = transform
    
    def __next__(self):
        """Returns the next image."""
        L = len(self) # Total number of images to loop over
        B = self.batch_size
        if self._iter_idx < L:
            start_idx = self._iter_idx
            end_idx = min(L, start_idx + B)
            batch = [self[i] for i in range(start_idx, end_idx)]
            self._iter_idx = end_idx
            self._batch_idx += 1
            self._interbatch_idx = list(range(start_idx, end_idx))
            #-- Convert to tensor
            tensor_images = [self.transform(img) for img in batch]
            batch_tensor = torch.stack(tensor_images)
            return batch_tensor
        else:
            self._iter_idx = 0
            self._batch_idx = 0
            self._interbatch_idx = None
            raise StopIteration
            
        
if __name__ == "__main__":
    
# =============================================================================
#     Test basic ImageChunkLoader
# =============================================================================
    INDIR = r"D:\dtu\speciale\ipp\processed\track_002_ascending\iff_dd\20171112_20171124_20171124_20171206.pha.png"
    OUTDIR = r"C:\Users\niels\Desktop\testout\1"
    CHUNK_SIZE = 224
    OVERLAP_FLAG = False
    DOWNSCALE_FACTOR = 1
    BATCH_SIZE = 64
    
    Obj = ImageChunkLoader(INDIR, chunk_size=CHUNK_SIZE, overlap=OVERLAP_FLAG, downscale_factor=DOWNSCALE_FACTOR)
    # Obj.downscale(4)
    # for i, (Img, box) in enumerate(zip(Obj, Obj.bboxes)):
    #     print(box)
    #     print(Obj._iter_idx)
    #     Img.save(OUTDIR + r"\img_%d.png" % i)
    
# =============================================================================
#     Test BatchImageChunkLoader and arange as torch tensor
# =============================================================================
    # INDIR = r"D:\dtu\speciale\ipp\processed\track_002_ascending\iff_dd\20171112_20171124_20171124_20171206.pha.png"
    # OUTDIR = r"C:\Users\niels\Desktop\testout\1"
    # CHUNK_SIZE = 224
    # OVERLAP_FLAG = False
    # DOWNSCALE_FACTOR = 1
    # BATCH_SIZE = 64

    # import torch
    # from torchvision import transforms as pth_transforms
    # data_transform = pth_transforms.Compose([
    #     pth_transforms.Resize(224, interpolation=3),
    #     pth_transforms.ToTensor(),
    #     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ])
    # Obj = BatchImageChunkLoader(INDIR, chunk_size=CHUNK_SIZE, overlap=OVERLAP_FLAG, downscale_factor=DOWNSCALE_FACTOR, batch_size=BATCH_SIZE)
    # for Img in Obj:
    #     tensor_images = [data_transform(img) for img in Img]
    #     batch_tensor = torch.stack(tensor_images)

# =============================================================================
#     Test [phase // recta // polar]_image functions
# =============================================================================
    # INDIR = r"D:\dtu\speciale\ipp\processed\track_002_ascending\iff_dd\20171112_20171124_20171124_20171206.pha"
    # OUTDIR = r"D:\dtu\speciale\Vejledning\Vejledning 4\phase_visualization"
    # MAG0 = None
    # MAG1 = 'Simple'
    # MAG2 = 'Geometric'
    # MAG3 = 'Harmonic'
    # MAG4 = 'Interferometric'
    # CHANNELS1 = [0, 1, 2]
    # CHANNELS2 = [0]
    # SAVE_FLAG = True
    # SHOW_FLAG = False
    
    #-- for [phase]
    # Img = phase_image(INDIR, CHANNELS1)
    # if SHOW_FLAG: Img.show()
    # if SAVE_FLAG: Img.save(OUTDIR + r"\phase_channels=%s.png" % str(CHANNELS1))
    # Img = phase_image(INDIR, CHANNELS2)
    # if SHOW_FLAG: Img.show()
    # if SAVE_FLAG: Img.save(OUTDIR + r"\phase_channels=%s.png" % str(CHANNELS2))
    
    #-- for [recta]
    # Img = recta_image(INDIR, MAG0)
    # if SHOW_FLAG: Img.show()
    # if SAVE_FLAG: Img.save(OUTDIR + r"\rect_mag=%s.png" % str(MAG0))
    # Img = recta_image(INDIR, MAG1)
    # if SHOW_FLAG: Img.show()
    # if SAVE_FLAG: Img.save(OUTDIR + r"\rect_mag=%s.png" % str(MAG1))
    # Img = recta_image(INDIR, MAG2)
    # if SHOW_FLAG: Img.show()
    # if SAVE_FLAG: Img.save(OUTDIR + r"\rect_mag=%s.png" % str(MAG2))
    # Img = recta_image(INDIR, MAG3)
    # if SHOW_FLAG: Img.show()
    # if SAVE_FLAG: Img.save(OUTDIR + r"\rect_mag=%s.png" % str(MAG3))
    # Img = recta_image(INDIR, MAG4)
    # if SHOW_FLAG: Img.show()
    # if SAVE_FLAG: Img.save(OUTDIR + r"\rect_mag=%s.png" % str(MAG4))
    
    #-- for [polar]
    # Img = polar_image(INDIR, MAG1)
    # if SHOW_FLAG: Img.show()
    # if SAVE_FLAG: Img.save(OUTDIR + r"\polar_mag=%s.png" % str(MAG1))
    # Img = polar_image(INDIR, MAG2)
    # if SHOW_FLAG: Img.show()
    # if SAVE_FLAG: Img.save(OUTDIR + r"\polar_mag=%s.png" % str(MAG2))
    # Img = polar_image(INDIR, MAG3)
    # if SHOW_FLAG: Img.show()
    # if SAVE_FLAG: Img.save(OUTDIR + r"\polar_mag=%s.png" % str(MAG3))
    # Img = polar_image(INDIR, MAG4)
    # if SHOW_FLAG: Img.show()
    # if SAVE_FLAG: Img.save(OUTDIR + r"\polar_mag=%s.png" % str(MAG4))
    
# =============================================================================
#     Test BatchImageChunkLoader with form=[phase // recta // polar]
# =============================================================================
    # INDIR = r"D:\dtu\speciale\ipp\processed\track_002_ascending\iff_dd\20171112_20171124_20171124_20171206.pha"
    # CHUNK_SIZE = 224
    # OVERLAP_FLAG = False
    # DOWNSCALE_FACTOR = 1
    # BATCH_SIZE = 1
    # MAG = True
    # CHANNELS=[0]
    # FORM1 = 'phase'
    # FORM2 = 'recta'
    # FORM3 = 'polar'
    
    # PHASE1 = r"C:\Users\niels\Desktop\testout\phase1"
    # PHASE2 = r"C:\Users\niels\Desktop\testout\phase2"
    # RECT1 = r"C:\Users\niels\Desktop\testout\rect1"
    # RECT2 = r"C:\Users\niels\Desktop\testout\rect2"
    # POLAR1 = r"C:\Users\niels\Desktop\testout\polar1"
    # from pathlib import Path
    # if not Path(PHASE1).exists(): Path(PHASE1).mkdir()
    # if not Path(PHASE2).exists(): Path(PHASE2).mkdir()
    # if not Path(RECT1).exists(): Path(RECT1).mkdir()
    # if not Path(RECT2).exists(): Path(RECT2).mkdir()    
    # if not Path(POLAR1).exists(): Path(POLAR1).mkdir()

    #-- for [Phase]
    # Obj = BatchImageChunkLoader(INDIR, chunk_size=CHUNK_SIZE, overlap=OVERLAP_FLAG, downscale_factor=DOWNSCALE_FACTOR, batch_size=BATCH_SIZE, form=FORM1)
    # for i, Img in enumerate(Obj):
    #     Img[0].save(PHASE1 + r"\img_%d.png" % i)
    # Obj = BatchImageChunkLoader(INDIR, chunk_size=CHUNK_SIZE, overlap=OVERLAP_FLAG, downscale_factor=DOWNSCALE_FACTOR, batch_size=BATCH_SIZE, form=FORM1, channels=CHANNELS)
    # for i, Img in enumerate(Obj):
    #     Img[0].save(PHASE2 + r"\img_%d.png" % i)
    
    #-- for [Recta]
    # Obj = BatchImageChunkLoader(INDIR, chunk_size=CHUNK_SIZE, overlap=OVERLAP_FLAG, downscale_factor=DOWNSCALE_FACTOR, batch_size=BATCH_SIZE, form=FORM2)
    # for i, Img in enumerate(Obj):
    #     Img[0].save(RECT1 + r"\img_%d.png" % i)
    # Obj = BatchImageChunkLoader(INDIR, chunk_size=CHUNK_SIZE, overlap=OVERLAP_FLAG, downscale_factor=DOWNSCALE_FACTOR, batch_size=BATCH_SIZE, form=FORM2, mag=MAG)
    # for i, Img in enumerate(Obj):
    #     Img[0].save(RECT2 + r"\img_%d.png" % i)
        
    #-- for [Polar]
    # Obj = BatchImageChunkLoader(INDIR, chunk_size=CHUNK_SIZE, overlap=OVERLAP_FLAG, downscale_factor=DOWNSCALE_FACTOR, batch_size=BATCH_SIZE, form=FORM3, mag=MAG)
    # for i, Img in enumerate(Obj):
    #     Img[0].save(POLAR1 + r"\img_%d.png" % i)
    

# =============================================================================
# Extract image with bboxes
# =============================================================================
    # from SGLNet.Plotting.plot_utils import draw_bbox
    # INDIR = r"D:\dtu\speciale\ipp\processed\track_002_ascending\iff_dd\20200101_20200113_20200113_20200125.pha"
    # OUTDIR = r"D:\dtu\speciale\Vejledning\Vejledning 6"
    # Loader = ImageChunkLoader(INDIR, chunk_size=448, overlap=False)
    # Img = Loader().copy()
    # draw_bbox(Img, Loader.bboxes)
    # Img.save(Path(OUTDIR) / f'{Loader.IffDD.file_name}.png')