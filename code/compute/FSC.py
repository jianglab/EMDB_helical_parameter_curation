import numpy as np
from numpy.fft import fftn, fftshift
import matplotlib.pyplot as plt

def calculate_fsc(map1, map2, pixel_size, shells=50):
    """
    Calculate FSC curve between two 3D density maps.
    
    Parameters:
    -----------
    map1, map2 : numpy.ndarray
        3D density maps with shape (N,N,N)
    pixel_size : float
        Pixel size in Angstroms
    shells : int
        Number of shells for FSC calculation
        
    Returns:
    --------
    spatial_freq : numpy.ndarray
        Spatial frequencies in 1/Angstrom
    fsc : numpy.ndarray
        FSC correlation values
    """
    # Check input dimensions
    assert map1.shape == map2.shape, "Maps must have the same dimensions"
    assert len(map1.shape) == 3, "Maps must be 3D"
    N = map1.shape[0]
    
    # Calculate Fourier transforms
    ft1 = fftshift(fftn(map1))
    ft2 = fftshift(fftn(map2))
    
    # Create distance matrix from center
    center = N // 2
    x, y, z = np.ogrid[-center:center, -center:center, -center:center]
    r = np.sqrt(x*x + y*y + z*z)
    
    # Maximum radius and shell thickness
    max_r = center
    shell_thickness = max_r / shells
    
    # Initialize arrays for results
    fsc = np.zeros(shells)
    shell_volumes = np.zeros(shells)
    
    # Calculate FSC for each shell
    for i in range(shells):
        r_min = i * shell_thickness
        r_max = (i + 1) * shell_thickness
        shell_mask = (r >= r_min) & (r < r_max)
        
        # Extract complex values in the shell
        f1_shell = ft1[shell_mask]
        f2_shell = ft2[shell_mask]
        
        # Calculate correlation
        numerator = np.abs(np.sum(f1_shell * f2_shell.conj()))
        denominator = np.sqrt(np.sum(np.abs(f1_shell)**2) * np.sum(np.abs(f2_shell)**2))
        
        if denominator != 0:
            fsc[i] = numerator / denominator
        shell_volumes[i] = np.sum(shell_mask)
    
    # Calculate spatial frequencies (in 1/Angstrom)
    spatial_freq = np.arange(shells) * shell_thickness / (N * pixel_size)
    
    return spatial_freq, fsc

def plot_fsc(spatial_freq, fsc, path,title="FSC Curve"):
    """
    Plot FSC curve with common threshold lines.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(spatial_freq, fsc, 'b-', label='FSC')
    
    # Add threshold lines
    plt.axhline(y=0.5, color='r', linestyle='--', label='0.5 threshold')
    plt.axhline(y=0.143, color='g', linestyle='--', label='0.143 threshold')
    
    plt.xlabel('Spatial Frequency (1/Å)')
    plt.ylabel('Fourier Shell Correlation')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.savefig(path)

    resolution = 0
    gold_standard = np.argwhere(fsc < 0.143).min()
    resolution = (spatial_freq[gold_standard] + spatial_freq[gold_standard-1]) / 2
    resolution = round(1/resolution, 2)
    
    return resolution