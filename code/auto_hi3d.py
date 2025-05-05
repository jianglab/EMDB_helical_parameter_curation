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
from compute.symmetrization import sym_cross_correlation


data_path = './files/need_curation.csv'
validated_path = './files/validated.csv'
non_validated_path = './files/non_validated.csv'

data_pd = pd.read_csv(data_path, dtype='str')
#data_pd = data_pd.iloc[0:2]

emdb_list = data_pd['emdb_id'].str[4:]
#emdb_list = ['43585']

if os.path.exists(validated_path) is False:
    data_pd_validated = data_pd[data_pd['validated']=='Yes']
    data_pd_validated.to_csv(validated_path, index=False)
    
if os.path.exists(non_validated_path) is False:
    data_pd_non_validated = data_pd[data_pd['validated']=='No']
    data_pd_non_validated.to_csv(non_validated_path, index=False)



for i in range(len(emdb_list)):

    emdid = emdb_list[i]

    # check this emdb path has been checked or not

    list_header = ['group','rise_validated (Å)','twist_validated (°)',
                   'csym_validated', 'vector difference','axes order',
                   'cc_emdb','cc_validated','validated','update']

    emdid_full = 'EMD-'+str(emdid)
    if os.path.exists(validated_path):
        data_pd_validated = pd.read_csv(validated_path, dtype='str')
        if emdid_full in list(data_pd_validated['emdb_id']):
            print(f'{emdid_full} has been checked')
            continue
    
    if os.path.exists(non_validated_path):
        data_pd_non_validated = pd.read_csv(non_validated_path, dtype='str')
        if emdid_full in list(data_pd_non_validated['emdb_id']):
            print(f'{emdid_full} has been checked')
            continue
    
    downloaded_data = get_correct_data_url(emdid)
    if downloaded_data is None:
        continue
    data, apix = downloaded_data
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
    
    rise_val = rise
    twist_val = twist
    
    if difference >= resolution:
        print(rise_original, twist_original, rise, twist)

    if rise == 0:
        rise_val = 1
        twist_val = 1

    if difference < resolution:
        cc, cc_hi3d = sym_cross_correlation(data, apix, rise_original, twist_original, rise_val, twist_val, only_original=False)
        validated = 'Yes'
        update = 'equal'
    elif difference >= resolution:
        cc, cc_hi3d = sym_cross_correlation(data, apix, rise_original, twist_original, rise_val, twist_val, only_original=False)
        validated = 'No'
    if cc_hi3d/cc > 1.1:
        update = 'improve'
    elif cc_hi3d/cc <0.9:
        update = 'worse'
        validated = 'No'
    else:
        update = 'equal'

    print(emdid, group, resolution, difference, cc, cc_hi3d)

    
    list_value = [group, rise, twist, csym, difference, '', cc, cc_hi3d, validated, update]

    data_pd.loc[data_pd['emdb_id'] == 'EMD-' + str(emdid),list_header] = list_value

    if validated == 'Yes':
        data_pd_validated = pd.concat([data_pd_validated, data_pd[data_pd['emdb_id'] == 'EMD-' + str(emdid)]],ignore_index=True)
        data_pd_validated.to_csv(validated_path, index=False)
    elif validated == 'No':
        data_pd_non_validated = pd.concat([data_pd_non_validated, data_pd[data_pd['emdb_id'] == 'EMD-' + str(emdid)]],ignore_index=True)
        data_pd_non_validated.to_csv(non_validated_path, index=False)


#import matplotlib.pyplot as plt
#
#plt.imshow(cylinder_projection_img, cmap='gray')
#plt.show()