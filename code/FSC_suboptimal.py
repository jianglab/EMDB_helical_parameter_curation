# this code requires the modules in the jspr/jianglab

import numpy as np
import pandas as pd
import os
import mrcfile
from compute.download import get_half_maps
from compute.symmetrization import apply_helical_symmetry
from compute.FSC import calculate_fsc, plot_fsc


def cylinder_mask(volume, soft_scale = 0.3, radius_ratio = None):
    shape = volume.shape
    z,y,x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = tuple(map(lambda x: int(x // 2), shape))

    distance = np.sqrt((x - center[2])**2 + (y - center[1])**2)

    if radius_ratio is None:
        radius = int(len(volume)//2)
    else:
        radius = int(len(volume)//2*radius_ratio)
    outer_radius = radius
    inner_radius = int(radius * (1-soft_scale))

    mask = np.zeros((1, shape[1],shape[2]))

    # Inside the inner radius
    mask[distance < inner_radius] = 1

    # Between inner and outer radius
    transition_zone = (distance >= inner_radius) & (distance <= outer_radius)
    mask[transition_zone] = 1 - (distance[transition_zone] - inner_radius) / (outer_radius - inner_radius)


    mask = np.repeat(mask, len(volume),axis=0)
    return mask

def apply_cos_circular_mask(volume, ratio = 1):
    d, h, w = volume.shape
    Z, Y, X = np.ogrid[-d // 2:d // 2, -h // 2:h // 2, -w // 2:w // 2]
    outer_radius = min(d, h, w) // 2
    outer_radius = int(outer_radius*ratio)
    inner_radius = int(outer_radius*0.9)
    distance = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    mask = np.ones((d, h, w), dtype=np.float32)
    mask[distance > outer_radius] = 0
    background = volume[mask == 0].mean()

    mask[distance <= inner_radius] = 1

    # Apply cosine transition between inner_mask and outer_mask
    transition_region = (distance > inner_radius) & (distance < outer_radius)
    mask[transition_region] = 0.5 * (
                1 + np.cos(np.pi * (distance[transition_region] - inner_radius) / (outer_radius - inner_radius)))

    #volume = volume-background
    volume = volume*mask
    return volume


def fsc_calculation(map1, map2, rise, twist, apix, n_rise=3, mask_path = None, trueFSC=True, name='ori',emdid='0000'):

    #dic_path = '/tmp/fsc'
    dic_path = '/net/jiang/scratch/li3221/curation/suboptimal_2'
    save_path = dic_path + f'/{emdid}'
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)

    D, H, W = map1.shape
    new_size = (D, H, W)

    fractions = n_rise*rise/(D*apix)
    fractions = min(0.1, fractions)
    fractions = 1

    map1_copy = map1.copy()
    map2_copy = map2.copy()

    c_mask = cylinder_mask(map1_copy, radius_ratio=0.7)
    #map1_copy = map1_copy*c_mask
    #map2_copy = map2_copy*c_mask

    #map1_copy = apply_cos_circular_mask(map1_copy, ratio=0.7)
    #map2_copy = apply_cos_circular_mask(map2_copy, ratio=0.7)
    

    sym_map1 = apply_helical_symmetry(map1_copy, apix, twist,
                                        rise, new_size=new_size, new_apix=apix,cpu=1,
                                        fraction=fractions)
    sym_map2 = apply_helical_symmetry(map2_copy, apix, twist,
                                        rise, new_size=new_size, new_apix=apix,cpu=1,
                                        fraction=fractions)
    

    sym_map1 = apply_cos_circular_mask(sym_map1, ratio=0.9)
    sym_map2 = apply_cos_circular_mask(sym_map2, ratio=0.9)

    average_map  = (sym_map1+sym_map2)/2

    #sym_map1 = sym_map1*c_mask
    #sym_map2 = sym_map2*c_mask
    
    sym_map1_path = save_path+f'/map1_{name}.mrc'
    sym_map2_path = save_path+f'/map2_{name}.mrc'
    average_path = save_path+f'/average_{name}.mrc'
    
    with mrcfile.new(sym_map1_path, overwrite=True) as mrc:
        mrc.set_data(sym_map1.astype(np.float32))
        mrc.voxel_size = apix
    
    with mrcfile.new(sym_map2_path, overwrite=True) as mrc:
        mrc.set_data(sym_map2.astype(np.float32))
        mrc.voxel_size = apix
    
    with mrcfile.new(average_path, overwrite=True) as mrc:
        mrc.set_data(average_map.astype(np.float32))
        mrc.voxel_size = apix

    if trueFSC is True:
        command = f'trueFSC.py {sym_map1_path} {sym_map2_path} {save_path}/fsc_plot_{name}.pdf'
        os.system(command)
        log_file = save_path+f'/fsc_plot_{name}.log'
        with open(log_file, "r") as f:
            output_lines = f.read().strip().split('\n')
        value_line = output_lines[7].split(' ')
        resolution = value_line[3]
    else:
        fiugre_path = save_path+f'/fsc_plot_{name}.png'
        spatial_freq, fsc = calculate_fsc(sym_map1, sym_map2, pixel_size=apix,shells=len(sym_map1)//2)
        resolution = plot_fsc(spatial_freq, fsc, fiugre_path)
    
    return resolution

data_path = './files/EMDB_validation.csv'
output_pd_path = './files/suboptimal_2.csv'
use_trueFSC = True
data_pd = pd.read_csv(data_path, index_col=False)
suboptimal_pd = pd.DataFrame({'emdb_id':[], 'FSC_original':[], 'FSC_validated':[]})

suboptimal_data = data_pd[(data_pd['reason'] == 'suboptimal') | (data_pd['reason 1'] == 'suboptimal')]
emdb_list = suboptimal_data['emdb_id'].str[4:]
emdb_list = list(emdb_list)
#emdb_list = emdb_list[1:2]
#emdb_list = ['31953']

for i in range(len(emdb_list)):

    emdid = emdb_list[i]
    emdid_full = 'EMD-'+emdid

    value_list = ['rise_deposited (Å)', 'twist_deposited (°)','rise_validated (Å)', 'twist_validated (°)']
    rise_original, twist_original, rise, twist = list(suboptimal_data.loc[suboptimal_data['emdb_id']==emdid_full, value_list].iloc[0])

    print(rise_original, twist_original, rise, twist)

    if os.path.exists(output_pd_path):
        suboptimal_pd = pd.read_csv(output_pd_path, dtype='str')
        if emdid_full in list(suboptimal_pd['emdb_id']):
            print(f'{emdid_full} has been checked')
            continue

    try:
        map1, map2, apix1, apix2 = get_half_maps(emdid)
    except:
        continue

    apix = (float(apix1)+float(apix2))/2

    FSC_original_sym = fsc_calculation(map1, map2, float(rise_original), float(twist_original), apix, trueFSC=use_trueFSC, name='ori', emdid=emdid)
    print('Finish original')
    FSC_hi3d_sym = fsc_calculation(map1, map2, float(rise), float(twist), apix, trueFSC=use_trueFSC, name='hi3d', emdid=emdid)
    print('Finish validated')

    new_row = pd.DataFrame([{
        'emdb_id': emdid_full,
        'FSC_original': FSC_original_sym,
        'FSC_validated': FSC_hi3d_sym
    }])

    suboptimal_pd = pd.concat([suboptimal_pd, new_row], ignore_index=True)
    suboptimal_pd.to_csv(output_pd_path,index=False)

    print(emdid, FSC_original_sym, FSC_hi3d_sym)

