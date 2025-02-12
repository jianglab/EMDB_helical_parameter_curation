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

def apply_sym(volume_data, original_pixel, rise_angstrom, twist_degree, rise_val, twist_val, only_original=False):

    mask = mask_around(volume_data, 20)
    D, H, W = volume_data.shape

    new_size = (D, H, W)

    fractions = 2*rise_angstrom/(D*original_pixel)

    sym_volume = apply_helical_symmetry(volume_data, original_pixel, twist_degree,
                                        rise_angstrom, new_size=new_size, new_apix=original_pixel,cpu=1,
                                        fraction=fractions)

    cc = normalized_cross_correlation(volume_data, sym_volume, mask=mask)

    if only_original == True:
        cc_hi3d=cc
    else:
        fractions = 2*rise_val/(D*original_pixel)
        sym_volume = apply_helical_symmetry(volume_data, original_pixel, twist_val,
                                            rise_val, new_size=new_size, new_apix=original_pixel, cpu=1,
                                            fraction=fractions)
        cc_hi3d = normalized_cross_correlation(volume_data, sym_volume, mask=mask)

    return cc, cc_hi3d
