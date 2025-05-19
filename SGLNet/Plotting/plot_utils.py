from matplotlib import pyplot as plt
import matplotlib as mpl
from PIL import ImageDraw, Image
import numpy as np
import numpy.typing as npt
import torch
import sys

def remove_axis_elements(ax : plt.Axes) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.tick_params(
        axis='both', 
        which='both', 
        length=0, 
        labelbottom=False, 
        labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

        
def plot_maximized() -> None:
    manager = plt.get_current_fig_manager()

    if hasattr(manager, 'window'):
        window = manager.window
        try:
            window.attributes('-fullscreen', 1)
        except AttributeError:
            try:
                window.showMaximized()
            except AttributeError:
                pass
            
def set_fontsize(small_size: int = 16, medium_size: int = 20, large_size: int = 24, set_all: int = None, **kwargs) -> None:
    VALID = ['font', 'axestitle', 'label', 'xtick', 'ytick', 'legend', 'figuretitle']
    
    plt.rc('font', size = set_all or small_size)          # controls default text sizes
    plt.rc('axes', titlesize = set_all or small_size)     # fontsize of the axes title
    plt.rc('axes', labelsize = set_all or medium_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize = set_all or small_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = set_all or small_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize = set_all or small_size)    # legend fontsize
    plt.rc('figure', titlesize = set_all or large_size)   # fontsize of the figure title
    
    for key, val in kwargs:
        if key not in VALID:
            print(f"UserWarning: Unknown kwarg '{key}'", file=sys.stderr)
            continue
        if key == 'font':
            plt.rc('font', size=val)
        if key == 'axestitle':
            plt.rc('axes', titlesize=val)
        if key == 'label':
            plt.rc('axes', labelsize=val)
        if key == 'xtick':
            plt.rc('xtick', labelsize=val)
        if key == 'ytick':
            plt.rc('ytick', labelsize=val)
        if key == 'legend':
            plt.rc('legend', fontsize=val)
        if key == 'figuretitle':
            plt.rc('figure', titlesize=val)
        

def TkAgg_focus() -> None:
    current_backend = mpl.get_backend()
    if 'tkagg' in current_backend:
        root = plt.gcf().canvas.manager.window
        root.focus_force()
        
        
def plot_dd_pha_with_events(IffDDEvent):
    draw = ImageDraw.Draw(IffDDEvent.png)
    for y0, y1, x0, x1 in zip(IffDDEvent.EventCoords.li0, IffDDEvent.EventCoords.li1, IffDDEvent.EventCoords.sa0, IffDDEvent.EventCoords.sa1):
        draw.rectangle([x0, y0, x1, y1], outline="red", width=5)
    IffDDEvent.png.show()
    
    
def draw_bbox(img: Image, bbox: tuple, color='white', linewidth=5):
    draw = ImageDraw.Draw(img)
    if isinstance(bbox, list):
        for b in bbox:
            draw.rectangle(tuple(b), outline=color, width=linewidth)
    elif isinstance(bbox, np.ndarray):
        if bbox.ndim != 2:
            raise ValueError("bbox of type <class np.ndarray> must be 2D.")
        for b in bbox:
            draw.rectangle(tuple(b), outline=color, width=linewidth)
    elif isinstance(bbox, tuple):
        draw.rectangle(bbox, outline=color, width=linewidth)
    else:
        raise ValueError("bbox must be a list/ndarray of tuples or a tuple")
        
def draw_outlines(img: Image, outlines, color='white', linewidth=5):
    draw = ImageDraw.Draw(img)
    for outline in outlines:
        outline_list = [(int(x), int(y)) for y, x in outline]
        draw.line(outline_list, fill=color, width=linewidth)
        
def plot_tensor(image_tensor, show: bool = True):
    """Converts a (C, H, W) tensor to a PIL Image."""
    # Ensure tensor is in CPU and detach from computation graph
    image_tensor = image_tensor.cpu().detach()

    # Normalize tensor (if needed) and scale to [0, 255]
    if image_tensor.dtype != torch.uint8:
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min()) * 255

    # Convert to NumPy and permute to (H, W, C)
    image_np = image_tensor.permute(1, 2, 0).byte().numpy()

    # Convert to PIL Image
    Image.fromarray(image_np).show()
    
def to_rgba_array(arr: npt.NDArray, vmin: np.float32 = None, vmax: np.float32 = None, colormap: str = 'gray') -> npt.NDArray[np.uint8]:
    """
    Converts a 2D NumPy array to an RGBA image using a given matplotlib colormap.

    Args:
        arr (np.ndarray): 2D input array.
        normalize (bool): Whether to normalize to [0, 1] before applying colormap.
        colormap (str or Colormap): Colormap name or object.

    Returns:
        np.ndarray: RGBA image (H, W, 4) as uint8.
    """
    if vmin is not None and vmax is not None:
        vmin = vmin.astype(np.float32)
        vmax = vmax.astype(np.float32)
        arr = arr.astype(np.float32)
        arr = (arr - vmin) / (vmax - vmin + 1e-8)

    if isinstance(colormap, str):
        cmap = mpl.colormaps[colormap]
    else:
        cmap = colormap

    rgba_float = cmap(arr)  # (H, W, 4) float32 in range [0, 1]
    rgba_uint8 = (rgba_float * 255).astype(np.uint8)
    return rgba_uint8

def grid_image(nested_list_of_images: list[list[npt.NDArray[np.uint8]]], padding: int = 10, pad_value: tuple = (0, 0, 0, 0)) -> Image:
    """
    Combines lists of RGBA NumPy arrays into a single PIL RGBA Image.
    Lists must have equal lengths and arrays inside lists must have the same shape
    """
    n_lists = len(nested_list_of_images)
    len_lists = [len(list) for list in nested_list_of_images]
    assert all(l == len_lists[0] for l in len_lists), "Lists must be same length"
    h, w, _ = nested_list_of_images[0][0].shape
    num_stacks = len_lists[0]

    total_height = n_lists * h + 2 * padding
    total_width = num_stacks * w + (num_stacks - 1) * padding

    canvas = np.full((total_height, total_width, 4), pad_value, dtype=np.uint8)

    for i in range(num_stacks):
        x_offset = i * (w + padding)
        for j, arr_list in enumerate(nested_list_of_images):
            y_offset = j * (h + padding)
            canvas[y_offset:y_offset + h, x_offset:x_offset + w] = arr_list[i]

    return Image.fromarray(canvas, mode='RGBA')

def rescale_img(img: Image, scale: float):
    new_size = (int(img.width*scale), int(img.height*scale))
    return img.resize(new_size, Image.LANCZOS)