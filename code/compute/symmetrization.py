import numpy as np
import mrcfile
import time
from datetime import datetime as dt
import os, math
import numpy as np
import platform
from numba import jit, set_num_threads, prange



def lp_filter(image, factor):
    # Perform the Fourier Transform
    F = np.fft.fftn(image)
    F = np.fft.fftshift(F)

    d, h, w = F.shape
    Z, Y, X = np.ogrid[-d//2:d//2, -h//2:h//2, -w//2:w//2]
    distance = np.sqrt(X**2 + Y**2 + Z**2)
    distance = distance/min(d, h, w)
    b_factor = -math.log(2)*4*(factor**2-1)

    # use gaussian mask
    gaussian_mask = np.exp(b_factor*(distance**2))

    # apply mask
    F_filtered = F * gaussian_mask
    # Perform the Inverse Fourier Transform
    filtered_volume = np.fft.ifftn(np.fft.ifftshift(F_filtered))
    filtered_volume = np.real(filtered_volume)

    return filtered_volume

def mask_around(volume, factor):
    lp_volume_more = lp_filter(volume,factor)
    mean = lp_volume_more.mean()
    mask = np.ones_like(lp_volume_more)
    mask[lp_volume_more < mean]=0
    return mask

@jit(nopython=True, cache=False, nogil=True, parallel=False if platform.system()=="Darwin" else True)
def apply_helical_symmetry(data, apix, twist_degree, rise_angstrom, csym=1, fraction=0.5, new_size=None, new_apix=None, cpu=1):
    if new_apix is None:
        new_apix = apix

    nz0, ny0, nx0 = data.shape
    if new_size != data.shape:
        nz1, ny1, nx1 = new_size
        nz2, ny2, nx2 = max(nz0, nz1), max(ny0, ny1), max(nx0, nx1)
        data_work = np.zeros((nz2, ny2, nx2), dtype=np.float32)
    else:
        data_work = np.zeros((nz0, ny0, nx0), dtype=np.float32)

    nz, ny, nx = data_work.shape
    w = np.zeros((nz, ny, nx), dtype=np.float32)

    hsym_max = max(1, int(nz*new_apix/rise_angstrom))
    hsyms = range(-hsym_max, hsym_max+1)
    csyms = range(csym)

    mask = (data!=0)*1
    z_nonzeros = np.nonzero(mask)[0]
    z0 = np.min(z_nonzeros)
    z1 = np.max(z_nonzeros)
    z0 = max(z0, nz0//2-int(nz0*fraction+0.5)//2)
    z1 = min(nz0-1, min(z1, nz0//2+int(nz0*fraction+0.5)//2))

    set_num_threads(cpu)

    for hi in hsyms:
        for k in prange(nz):
            k2 = ((k-nz//2)*new_apix + hi * rise_angstrom)/apix + nz0//2
            if k2 < z0 or k2 >= z1: continue
            k2_floor, k2_ceil = int(np.floor(k2)), int(np.ceil(k2))
            wk = k2 - k2_floor

            for ci in csyms:
                rot = np.deg2rad(twist_degree * hi + 360*ci/csym)
                m = np.array([
                      [ np.cos(rot),  np.sin(rot)],
                      [-np.sin(rot),  np.cos(rot)]
                    ])
                for j in prange(ny):
                    for i in prange(nx):
                        j2 = (m[0,0]*(j-ny//2) + m[0,1]*(i-nx/2))*new_apix/apix + ny0//2
                        i2 = (m[1,0]*(j-ny//2) + m[1,1]*(i-nx/2))*new_apix/apix + nx0//2

                        j2_floor, j2_ceil = int(np.floor(j2)), int(np.ceil(j2))
                        i2_floor, i2_ceil = int(np.floor(i2)), int(np.ceil(i2))
                        if j2_floor<0 or j2_floor>=ny0-1: continue
                        if i2_floor<0 or i2_floor>=nx0-1: continue

                        wj = j2 - j2_floor
                        wi = i2 - i2_floor

                        data_work[k, j, i] += (
                            (1 - wk) * (1 - wj) * (1 - wi) * data[k2_floor, j2_floor, i2_floor] +
                            (1 - wk) * (1 - wj) * wi * data[k2_floor, j2_floor, i2_ceil] +
                            (1 - wk) * wj * (1 - wi) * data[k2_floor, j2_ceil, i2_floor] +
                            (1 - wk) * wj * wi * data[k2_floor, j2_ceil, i2_ceil] +
                            wk * (1 - wj) * (1 - wi) * data[k2_ceil, j2_floor, i2_floor] +
                            wk * (1 - wj) * wi * data[k2_ceil, j2_floor, i2_ceil] +
                            wk * wj * (1 - wi) * data[k2_ceil, j2_ceil, i2_floor] +
                            wk * wj * wi * data[k2_ceil, j2_ceil, i2_ceil]
                        )
                        w[k, j, i] += 1.0
    mask = w>0
    data_work = np.where(mask, data_work / w, data_work)
    if data_work.shape != new_size:
        nz1, ny1, nx1 = new_size
        data_work = data_work[nz//2-nz1//2:nz//2+nz1//2, ny//2-ny1//2:ny//2+ny1//2, nx//2-nx1//2:nx//2+nx1//2]
    return data_work

def apply_helical_symmetry_ds(map_3d, pixel_size, rise, twist_deg, cyclic_sym, helical_axis=2):

    
    import numpy as np
    from scipy.ndimage import rotate, shift
    """
    Symmetrize a 3D density map using helical and cyclic symmetries.
    
    Parameters:
    - map_3d: Input volume (N, N, N) numpy array
    - pixel_size: Pixel size in Å/pixel
    - rise: Helical rise per subunit in Å
    - twist_deg: Helical twist per subunit in degrees
    - cyclic_sym: Cyclic symmetry order (C)
    - helical_axis: Axis index for helical symmetry (0=z, 1=y, 2=x)
    
    Returns:
    - Symmetrized 3D numpy array
    """
    
    # Validate inputs
    if map_3d.ndim != 3 or len(set(map_3d.shape)) != 1:
        raise ValueError("Input map must be cubic (N, N, N)")
        
    # Convert units
    rise_pix = rise / pixel_size
    n = map_3d.shape[0]
    axes = [(0,1), (0,2), (1,2)][helical_axis]  # Rotation plane

    # Initialize output map
    sym_map = np.zeros_like(map_3d)
    total_ops = 0

    # Apply cyclic symmetry
    for c in range(cyclic_sym):
        # Rotate by cyclic symmetry angle
        cyclic_rot = rotate(map_3d, c*(360/cyclic_sym), 
                           axes=axes, reshape=False, order=3, mode='constant', cval=0.0)
        
        # Apply helical symmetry
        for h in [-1, 0, 1]:  # ±1 helical repeat around center
            # Calculate rotation and translation
            rot_angle = h * twist_deg
            trans = h * rise_pix
            
            # Create transformation matrix
            rotated = rotate(cyclic_rot, rot_angle, 
                            axes=axes, reshape=False, order=3, mode='constant', cval=0.0)
            
            # Apply translation along helical axis
            shift_vec = [0, 0, 0]
            shift_vec[helical_axis] = trans
            shifted = shift(rotated, shift_vec, 
                           order=3, mode='constant', cval=0.0)
            
            # Accumulate symmetrized map
            sym_map += shifted
            total_ops += 1

    # Average and return
    return sym_map / total_ops

def apply_helical_symmetry_cg(volume, pixel_size, rise, twist, cyclic_symmetry):

    from scipy.ndimage import affine_transform

    """
    Symmetrizes a 3D density map according to helical parameters and cyclic symmetry.
    
    Parameters:
      volume (numpy.ndarray): 3D array of shape (N,N,N) representing the density map.
      pixel_size (float): Pixel size in angstroms per pixel.
      rise (float): Helical rise in angstroms.
      twist (float): Helical twist in degrees (per symmetry copy).
      cyclic_symmetry (int): Order of cyclic symmetry (number of copies to average).
      
    Returns:
      numpy.ndarray: The symmetrized 3D density map.
    """
    # Get the shape and compute the center (assume volume center is the symmetry center)
    shape = np.array(volume.shape)
    center = (shape - 1) / 2.0  # center in voxel coordinates

    # Prepare an accumulator for the symmetrized map
    sym_volume = np.zeros_like(volume, dtype=np.float32)

    n_ops = 0

    total_ops = int(shape[0] * pixel_size / rise)

    # For each symmetry copy, apply the rotation and translation and add the result.
    for i in range(-total_ops//2, total_ops//2):
        # Calculate the rotation angle for this copy (in radians)
        angle_rad = np.deg2rad(i * twist)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotation about the z-axis (z is unchanged)
        R = np.array([[cos_a, -sin_a, 0],
                      [sin_a,  cos_a, 0],
                      [0,         0,  1]])
        
        # Translation along z (convert rise from angstrom to voxels)
        dz_vox = i * (rise / pixel_size)
        t = np.array([0, 0, dz_vox])
        
        # For affine_transform we need the inverse transformation.
        # The forward transformation is:
        #    y = center + R @ (x - center) + t
        # So the inverse is:
        #    x = center + R.T @ (y - center - t)
        # Therefore, we use matrix = R.T and offset computed as below.
        R_inv = R.T
        offset = center - R_inv.dot(center + t)
        
        # Apply the affine transformation
        # Using cubic interpolation (order=3) and filling areas outside the original with 0.
        transformed = affine_transform(volume, R_inv, offset=offset, order=3, mode='constant', cval=0.0)
        
        # Accumulate the transformed volume
        sym_volume += transformed

    # Average the accumulated volume
    sym_volume /= total_ops
    return sym_volume



def apply_helical_symmetry_cg_gpu(volume, pixel_size, rise, twist, cyclic_symmetry, device='cuda'):

    import torch
    
    """
    Symmetrizes a 3D density map according to helical parameters and cyclic symmetry, using GPU acceleration with PyTorch.
    
    Parameters:
      volume (torch.Tensor): 3D tensor of shape (N,N,N) representing the density map on GPU.
      pixel_size (float): Pixel size in angstroms per pixel.
      rise (float): Helical rise in angstroms.
      twist (float): Helical twist in degrees (per symmetry copy).
      cyclic_symmetry (int): Order of cyclic symmetry (number of copies to average).
      device (str): The device to run the code on ('cuda' or 'cpu').
      
    Returns:
      torch.Tensor: The symmetrized 3D density map on GPU.
    """
    # Ensure volume is on the correct device (GPU or CPU)
    volume = torch.tensor(volume)
    volume = volume.to(device)
    
    shape = torch.tensor(volume.shape, device=device)

    center = (shape - 1) / 2.0  # Center in voxel coordinates

    sym_volume = torch.zeros_like(volume, dtype=torch.float32, device=device)

    # Calculate total number of operations (based on the shape of the z-axis and rise)
    total_ops = int(torch.floor(shape[2] * pixel_size / rise))  # Shape in z-direction
    range_to_use = list(range(-total_ops // 2, total_ops // 2))
    range_to_use = [0]

    # For each symmetry copy, apply the rotation and translation
    for i in range_to_use:

        # Rotation angle for this copy (in radians)
        angle_rad = torch.deg2rad(torch.tensor(i * twist, dtype=torch.float32, device=device))
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)
        
        # Rotation matrix about the z-axis
        R = torch.tensor([[cos_a, -sin_a, 0],
                          [sin_a,  cos_a, 0],
                          [0,         0,  1]], device=device)
        
        # Translation along z (convert rise from angstrom to voxels)
        dz_vox = i * (rise / pixel_size)
        t = torch.tensor([0, 0, dz_vox], device=device)

        print(angle_rad, t)
        
        # Inverse transformation calculation (rotation and translation)
        R_inv = R.T
        offset = center - torch.matmul(R_inv, center + t)

        # Perform the affine transformation
        transformed = affine_transform_torch(volume, R_inv, offset, device)

        # Accumulate the transformed volume
        sym_volume += transformed

    # Average the accumulated volume
    sym_volume /= total_ops

    sym_volume = sym_volume.cpu().numpy()
    return sym_volume

def affine_transform_torch(volume, rotation_matrix, offset, device):
    """
    Custom GPU-based affine transformation for 3D volume using PyTorch.
    
    Parameters:
      volume (torch.Tensor): 3D tensor representing the density map on GPU.
      rotation_matrix (torch.Tensor): 3x3 tensor representing the rotation matrix.
      offset (torch.Tensor): 3D vector for the translation (offset).
      device (str): The device to run the code on ('cuda' or 'cpu').
    
    Returns:
      torch.Tensor: The transformed volume.
    """

    import torch

    # Get the grid of coordinates for the volume
    grid = torch.meshgrid(torch.arange(volume.shape[0], device=device),
                          torch.arange(volume.shape[1], device=device),
                          torch.arange(volume.shape[2], device=device))
    grid = torch.stack(grid, dim=-1).float()

    # Apply the affine transformation: grid' = rotation_matrix * grid + offset
    grid_reshaped = grid.view(-1, 3)  # Flatten to a list of coordinates
    transformed_grid = torch.matmul(grid_reshaped, rotation_matrix.T) + offset

    # Reshape the transformed grid back to the original shape
    transformed_grid = transformed_grid.view(volume.shape[0], volume.shape[1], volume.shape[2], 3)

    # Perform trilinear interpolation
    transformed_volume = F.grid_sample(volume.unsqueeze(0).unsqueeze(0), transformed_grid.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False)

    # Return the transformed volume
    return transformed_volume.squeeze(0).squeeze(0)





def normalized_cross_correlation(volume1, volume2, mask=None):

    # ZNCC in wiki
    # Normalize the volumes


    if mask is not None:
        volume1_normalized = volume1*mask
        volume2_normalized = volume2*mask
    else:
        volume1_normalized = volume1
        volume2_normalized = volume2

    volume1_normalized = (volume1_normalized - volume1_normalized.mean()) / (volume1_normalized.std())
    volume2_normalized = (volume2_normalized - volume2_normalized.mean()) / (volume2_normalized.std())

    # Compute the cross-correlation
    normalized_cross_corr = np.mean(volume1_normalized*volume2_normalized)

    return normalized_cross_corr

def sym_cross_correlation(volume_data, original_pixel, rise_angstrom, twist_degree, rise_val, twist_val, only_original=False, n_rise = 3):

    mask = mask_around(volume_data, 20)
    D, H, W = volume_data.shape

    new_size = (D, H, W)

    fractions = n_rise*rise_angstrom/(D*original_pixel)
    fractions = min(0.1, fractions)

    sym_volume = apply_helical_symmetry(volume_data, original_pixel, twist_degree,
                                        rise_angstrom, new_size=new_size, new_apix=original_pixel,cpu=1,
                                        fraction=fractions)

    cc = normalized_cross_correlation(volume_data, sym_volume, mask=mask)

    if only_original == True:
        cc_hi3d=cc
    else:
        fractions = n_rise*rise_val/(D*original_pixel)
        fractions = min(0.05, fractions)
        sym_volume = apply_helical_symmetry(volume_data, original_pixel, twist_val,
                                            rise_val, new_size=new_size, new_apix=original_pixel, cpu=1,
                                            fraction=fractions)
        cc_hi3d = normalized_cross_correlation(volume_data, sym_volume, mask=mask)

    return cc, cc_hi3d
