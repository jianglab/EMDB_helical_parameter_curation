import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)



import numpy as np
import pandas as pd
import sys, os

from compute.download import get_correct_data_url, get_emdb_parameters, is_amyloid
from compute.calculate_radius import rmin_max_estimateion, vector_diff
from compute.auto_correlation import cylindrical_projection, compute_helical_parameters
from compute.symmetrization import apply_sym

data_pd = pd.read_csv('./files/need_curation.csv', dtype='str')
data_pd = data_pd.iloc[0:2]

emdb_list = data_pd['emdb_id'].str[4:]
#emdb_list = ['43640']

for i in range(len(emdb_list)):
    emdid = emdb_list[i]
    
    data, apix = get_correct_data_url(emdid)
    meta_data = get_emdb_parameters(emdid)
    amyloid_id = is_amyloid(meta_data)

    resolution = meta_data['resolution']
    rise_original = meta_data['rise']
    twist_original = meta_data['twist']
    csym_original = meta_data['csym']

    if amyloid_id is True:
        da = 0.5
        dz = 0.2
        group = 'amyloid'
    else:
        da = 0.5
        dz = 0.5
        group = 'non-amyloid'

    rmin_auto, rmax_auto, centroid = rmin_max_estimateion(data)
    twist, rise, csym = compute_helical_parameters(data, apix = apix, da=da, dz=dz, rmin=rmin_auto, rmax=rmax_auto)

    difference = vector_diff(rise_original=rise_original, twist_original=twist_original, csym_original=csym_original, 
                             rise_predicted=rise, twist_predicted=twist, radius=centroid)

    if difference < resolution:
        cc, cc_hi3d = apply_sym(data, apix, rise_original, twist_original, rise, twist, only_original=True)
        validated = 'Yes'
        update = 'equal'
    elif difference >= resolution:
        cc, cc_hi3d = apply_sym(data, apix, rise_original, twist_original, rise, twist, only_original=False)
        validated = 'No'
        if cc < cc_hi3d:
            update = 'improve'
        else:
            update = 'equal'

    print(emdid, group, resolution, difference, cc, cc_hi3d)

    if difference >= resolution:
        print(rise_original, twist_original, rise, twist)

    
    list_header = ['group','curated_rise (Å)','curated_twist (°)',
                   'curated_csym', 'vector difference','axes order',
                   'cc_emdb','cc_curated','validated','update']
    list_value = [group, rise, twist, csym, difference, '', cc, cc_hi3d, validated, update]

    data_pd.loc[data_pd['emdb_id'] == 'EMD-' + str(emdid),list_header] = list_value

data_pd_validated = data_pd[data_pd['validated']=='Yes']
validated_path = './files/validated.csv'
data_pd_validated.to_csv(validated_path, index=False)

data_pd_non_validated = data_pd[data_pd['validated']=='No']
non_validated_path = './files/non_validated.csv'
data_pd_non_validated.to_csv(non_validated_path, index=False)


#import matplotlib.pyplot as plt
#
#plt.imshow(cylinder_projection_img, cmap='gray')
#plt.show()