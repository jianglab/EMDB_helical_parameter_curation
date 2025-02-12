import numpy as np
from scipy.ndimage import map_coordinates



def compute_radial_profile(data):
    proj = data.mean(axis=0)
    ny, nx = proj.shape
    rmax = min(nx//2, ny//2)
    
    r = np.arange(0, rmax, 1, dtype=np.float32)
    theta = np.arange(0, 360, 1, dtype=np.float32) * np.pi/180.
    n_theta = len(theta)

    theta_grid, r_grid = np.meshgrid(theta, r, indexing='ij', copy=False)
    y_grid = ny//2 + r_grid * np.sin(theta_grid)
    x_grid = nx//2 + r_grid * np.cos(theta_grid)

    coords = np.vstack((y_grid.flatten(), x_grid.flatten()))

    polar = map_coordinates(proj, coords, order=1).reshape(r_grid.shape)

    rad_profile = polar.mean(axis=0)
    return rad_profile

def estimate_radial_range(radprofile, thresh_ratio=0.1):
    background = np.mean(radprofile[-3:])
    thresh = (radprofile.max() - background) * thresh_ratio + background
    indices = np.nonzero(radprofile>thresh)
    rmin_auto = np.min(indices)
    rmax_auto = np.max(indices)
    return float(rmin_auto), float(rmax_auto)

def rmin_max_estimateion(data, thresh_ratio=0.1):
    rad_profile = compute_radial_profile(data)
    rmin_auto, rmax_auto = estimate_radial_range(rad_profile, thresh_ratio = thresh_ratio)
    centroid = find_centroid(rad_profile[int(rmin_auto):int(rmax_auto)])
    return rmin_auto, rmax_auto, centroid

def find_centroid(profile):
    profile_normalize = profile/profile.sum()
    x = np.linspace(0,len(profile_normalize),len(profile_normalize))
    centroid = np.sum(x * profile_normalize)
    return centroid

def vector_diff(rise_original, twist_original, csym_original, rise_predicted, twist_predicted, radius):

    vector_length_ori = np.sqrt(rise_original**2+ radius**2)

    csym_originals = np.arange(0,2*np.pi,2*np.pi/csym_original)
    twist_originals = twist_original*np.pi/180+csym_originals
    rise_originals = np.ones_like(twist_originals)*rise_original


    x_oris = np.cos(twist_originals)*radius
    y_oris = np.sin(twist_originals)*radius
    z_oris = rise_originals

    vectors_oris = np.stack([x_oris,y_oris,z_oris], axis=0).T

    vector_length_ori = np.sqrt(rise_original**2+ radius**2)
    twist_predicted = twist_predicted*np.pi/180

    x_pred = np.cos(twist_predicted)*radius
    y_pred = np.sin(twist_predicted)*radius
    z_pred = rise_predicted

    vector_predict = np.array([x_pred,y_pred,z_pred])

    distances = np.linalg.norm(vectors_oris - vector_predict, axis=1)

    distance = distances.min()

    return distance
    
