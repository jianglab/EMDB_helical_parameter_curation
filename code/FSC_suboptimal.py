# this code requires the modules in the jspr/jianglab

import numpy as np
import pandas as pd
import os
import mrcfile
from compute.download import get_half_maps
from compute.symmetrization import apply_helical_symmetry


def fsc_calculation(map1, map2, rise, twist, apix, n_rise=3, mask_path = None):

    save_path = '/tmp/fsc'
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)

    D, H, W = map1.shape
    new_size = (D, H, W)

    fractions = n_rise*rise/(D*apix)
    fractions = min(0.1, fractions)
    fractions = 0.2

    

    sym_map1 = apply_helical_symmetry(map1, apix, twist,
                                        rise, new_size=new_size, new_apix=apix,cpu=1,
                                        fraction=fractions)
    sym_map2 = apply_helical_symmetry(map2, apix, twist,
                                        rise, new_size=new_size, new_apix=apix,cpu=1,
                                        fraction=fractions)
    
    sym_map1_path = save_path+'/map1.mrc'
    sym_map2_path = save_path+'/map2.mrc'
    
    with mrcfile.new(sym_map1_path, overwrite=True) as mrc:
        mrc.set_data(sym_map1.astype(np.float32))
        mrc.voxel_size = apix
    
    with mrcfile.new(sym_map2_path, overwrite=True) as mrc:
        mrc.set_data(sym_map2.astype(np.float32))
        mrc.voxel_size = apix

    # Construct your command with output redirection
    command = f'trueFSC.py {sym_map1_path} {sym_map2_path} {save_path}/fsc.pdf'
    #if mask_path is not None:

    os.system(command)

    log_file = save_path+'/fsc.log'
    #os.system(f'trueFSC.py -h')

    with open(log_file, "r") as f:
        output_lines = f.read().strip().split('\n')

    # mask FSC
    #value_line = output_lines[-3].split(' ')

    # unmasked FSC
    value_line = output_lines[7].split(' ')
    
    resolution = value_line[3]

    print(value_line)

    return resolution

data_path = './EMDB_validation.csv'
output_pd_path = './files/suboptimal.csv'

data_pd = pd.read_csv(data_path, index_col=False)
suboptimal_pd = pd.DataFrame({'emdb_id':[], 'FSC_original':[], 'FSC_curated':[]})

suboptimal_data = data_pd[(data_pd['reason'] == 'suboptimal') | (data_pd['reason 1'] == 'suboptimal')]
emdb_list = suboptimal_data['emdb_id'].str[4:]
emdb_list = list(emdb_list)
#emdb_list = emdb_list[1:2]
#emdb_list = ['33934']

for i in range(len(emdb_list)):

    emdid = emdb_list[i]
    emdid_full = 'EMD-'+emdid

    value_list = ['rise_deposited (Å)', 'twist_deposited (°)','curated_rise (Å)', 'curated_twist (°)']
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

    FSC_original_sym = fsc_calculation(map1, map2, float(rise_original), float(twist_original), apix)
    FSC_hi3d_sym = fsc_calculation(map1, map2, float(rise), float(twist), apix)

    new_row = pd.DataFrame([{
        'emdb_id': emdid_full,
        'FSC_original': FSC_original_sym,
        'FSC_curated': FSC_hi3d_sym
    }])

    suboptimal_pd = pd.concat([suboptimal_pd, new_row], ignore_index=True)
    suboptimal_pd.to_csv(output_pd_path,index=False)

    print(emdid, FSC_original_sym, FSC_hi3d_sym)

