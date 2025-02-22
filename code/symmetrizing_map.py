import numpy as np
import pandas as pd
import os
import mrcfile
from compute.download import get_correct_data_url
from compute.symmetrization import apply_helical_symmetry

data_path = './EMDB_validation.csv'

data_pd = pd.read_csv(data_path, dtype='str')
emdb_list = ['32033', '0614','41844']
emdb_list = ['33934']

save_path = '/tmp/sym_map'

if os.path.exists(save_path) is False:
    os.mkdir(save_path)


for i in range(len(emdb_list)):

    emdid = emdb_list[i]
    emdid_full = 'EMD-'+emdid

    value_list = ['rise_deposited (Å)', 'twist_deposited (°)','curated_rise (Å)', 'curated_twist (°)']
    rise_original, twist_original, rise, twist = list(data_pd.loc[data_pd['emdb_id']==emdid_full, value_list].iloc[0].astype(np.float32))

    print(emdid_full, rise_original, twist_original, rise, twist)

    map, apix = get_correct_data_url(emdid)


    D, H, W = map.shape
    new_size = (D, H, W)

    fractions = 3*rise/(D*apix)
    fractions = min(0.1, fractions)
    fractions = 0.5

    sym_map1 = apply_helical_symmetry(map, apix, twist_original,
                                        rise_original, new_size=new_size, new_apix=apix,cpu=1,
                                        fraction=fractions)
    
    sym_map2 = apply_helical_symmetry(map, apix, twist,
                                        rise, new_size=new_size, new_apix=apix,cpu=1,
                                        fraction=fractions)
    
    sym_map1_path = save_path+f'/map_{emdid}_original.mrc'
    sym_map2_path = save_path+f'/map_{emdid}_curated.mrc'

    with mrcfile.new(sym_map1_path, overwrite=True) as mrc:
        mrc.set_data(sym_map1.astype(np.float32))
        mrc.voxel_size = apix
    
    with mrcfile.new(sym_map2_path, overwrite=True) as mrc:
        mrc.set_data(sym_map2.astype(np.float32))
        mrc.voxel_size = apix