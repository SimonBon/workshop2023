import numpy as np
import tifffile
from cellpose import models
import torch

def segment_image(tiff_path, model_path, diameter, use_gpu=True, gpu_index=-1):
    # Load the .tif image using tifffile
    img = tifffile.imread(tiff_path)
    
    # Determine the device based on the gpu_index
    if use_gpu:
        if gpu_index == -1:
            device = None  
            print("using CPU")
        else:
            if torch.cuda.is_available():
                try:
                    device = torch.device(f'cuda:{gpu_index}')
                    print("using GPU")
                except: 
                    device = torch.device('cuda')
                    print("using GPU")
            else:
                device = None
                print("using CPU")
            
            
    
    # Load the cellpose model
    model = models.CellposeModel(gpu=(gpu_index != -1), pretrained_model=model_path, device=device)
    
    # Segment the image using cellpose
    masks, _, _ = model.eval(img, diameter=diameter, channels=[0,0])
    
    return img, masks

# Example usage:
# img, segmentation = segment_image('path_to_tif_file.tif', 'path_to_trained_model', diameter=30, gpu_index=0

