from scipy import ndimage
import numpy as np

def get_coordinates(x, y, im_sz, hsz):
    
    # Get the center of the object
    cy = np.round((y.start + y.stop)/2, 2).astype(int)
    cx = np.round((x.start + x.stop)/2, 2).astype(int)
    
    # Calculate the object dimensions
    x_max, x_min = np.min([im_sz[1]-1, (cx + hsz)]), np.max([0, (cx - hsz)])
    y_max, y_min = np.min([im_sz[0]-1, (cy + hsz)]), np.max([0, (cy - hsz)])
    
    pad_l, pad_r = np.abs(np.min([0, (cx - hsz)])), np.abs(np.min([0, im_sz[1]-1-(cx + hsz)]))
    pad_t, pad_b = np.abs(np.min([0, (cy - hsz)])), np.abs(np.min([0, im_sz[0]-1-(cy + hsz)]))
    
    return cx, cy, x_max, x_min, y_max, y_min, pad_l, pad_r, pad_t, pad_b

def extract_and_pad_objects(mask, image, patch_sz, exclude_edges=True, use_surrounding=False):

    assert mask.shape[:2] == image.shape[:2]; "Mask and Image do not match shapes"
    assert patch_sz/2 == patch_sz//2

    # Find the objects in the labeled image
    objects = ndimage.find_objects(mask)

    hsz = patch_sz//2
    im_sz = mask.shape

    cell_patches = []
    image_patches = []
    surrounding_patches = []
    background_patches = []
    coords = []
    for i, obj in enumerate(objects):
        
        if obj is None:
            continue
        
        label = i+1
        
        y, x = obj
        
        cx, cy, x_max, x_min, y_max, y_min, pad_l, pad_r, pad_t, pad_b = get_coordinates(x, y, im_sz, hsz)
        
        if exclude_edges and sum([pad_l, pad_r, pad_t, pad_b]) > 0:
            continue

        mask_patch = np.pad(mask[y_min:y_max, x_min:x_max], ((pad_t, pad_b), (pad_l, pad_r)), mode='constant', constant_values=0)
        image_patch = np.pad(image[y_min:y_max, x_min:x_max], ((pad_t, pad_b), (pad_l, pad_r), (0,0)), mode='constant', constant_values=0)
        
        assert mask_patch.shape == (patch_sz, patch_sz), f"Shape should be {patch_sz} x {patch_sz} but is {mask_patch.shape[0]} x {mask_patch.shape[1]}"
        
        cell_mask = (mask_patch == label).astype(int)
        surrounding_mask = np.logical_and((mask_patch != label), (mask_patch != 0)).astype(int)
        background_mask = (mask_patch == 0).astype(int)
        
        if not use_surrounding:
            image_patch[cell_mask!=1] = 0

        image_patches.append(image_patch)
        cell_patches.append(cell_mask)
        surrounding_patches.append(surrounding_mask)
        background_patches.append(background_mask)
        coords.append((cx, cy))
        
    return image_patches, cell_patches, surrounding_patches, background_patches, coords