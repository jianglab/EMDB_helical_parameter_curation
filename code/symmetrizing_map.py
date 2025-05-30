import numpy as np
import pandas as pd
import os
import mrcfile
from compute.download import get_correct_data_url
from compute.symmetrization import apply_helical_symmetry, apply_helical_symmetry_cg, apply_helical_symmetry_cg_gpu, apply_helical_symmetry_ds

data_path = './files/EMDB_validation.csv'

data_pd = pd.read_csv(data_path, dtype='str')
emdb_list = ['28063']
#emdb_list = ['43868']

save_path = '/net/jiang/scratch/li3221/curation/sym_map'

if os.path.exists(save_path) is False:
    os.mkdir(save_path)


for i in range(len(emdb_list)):

    emdid = emdb_list[i]
    emdid_full = 'EMD-'+emdid

    value_list = ['rise_deposited (Å)', 'twist_deposited (°)','rise_validated (Å)', 'twist_validated (°)']
    rise_original, twist_original, rise, twist = list(data_pd.loc[data_pd['emdb_id']==emdid_full, value_list].iloc[0].astype(np.float32))

    print(emdid_full, rise_original, twist_original, rise, twist)

    map, apix = get_correct_data_url(emdid)


    D, H, W = map.shape
    new_size = (D, H, W)

    fractions = 2*rise/(D*apix)
    fractions = min(0.1, fractions)
    fractions = 1

    sym_map1_path = save_path+f'/map_{emdid}_original.mrc'
    sym_map2_path = save_path+f'/map_{emdid}_validated.mrc'

    sym_map1 = apply_helical_symmetry(map, apix, twist_original,
                                        rise_original, new_size=new_size, new_apix=apix,cpu=1,
                                        fraction=fractions)
    print('sym_map1', sym_map1.shape)
    with mrcfile.new(sym_map1_path, overwrite=True) as mrc:
        mrc.set_data(sym_map1.astype(np.float32))
        mrc.voxel_size = apix
    
    sym_map2 = apply_helical_symmetry(map, apix, twist,
                                        rise, new_size=new_size, new_apix=apix,cpu=1,
                                        fraction=fractions)
    
    print('sym_map2', sym_map2.shape)
    with mrcfile.new(sym_map2_path, overwrite=True) as mrc:
        mrc.set_data(sym_map2.astype(np.float32))
        mrc.voxel_size = apix

    #sym_map1 = apply_helical_symmetry_cg_gpu(map, apix, rise_original, twist_original, 1)
    #sym_map2 = apply_helical_symmetry_cg_gpu(map, apix, rise, twist, 1)
    
    

    
    
    