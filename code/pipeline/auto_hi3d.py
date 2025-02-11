import numpy as np
import pandas as pd
import sys, os

from compute.download import get_correct_data_url, get_emdb_parameters, is_amyloid
from compute.calculate_radius import rmin_max_estimateion, vector_diff
from compute.auto_correlation import cylindrical_projection, compute_helical_parameters

data_pd = pd.read_csv('./files/need_curation.csv')

data_pd = data_pd.iloc[0:10]


emdb_list = data_pd['emdb_id'].str[4:]

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
    else:
        da = 0.5
        dz = 0.5

    rmin_auto, rmax_auto, centroid = rmin_max_estimateion(data)
    twist, rise, csym = compute_helical_parameters(data, apix = apix, da=da, dz=dz, rmin=rmin_auto, rmax=rmax_auto)

    difference = vector_diff(rise_original=rise_original, twist_original=twist_original, csym_original=csym_original, 
                             rise_predicted=rise, twist_predicted=twist, radius=centroid)

    print(rise_original, twist_original, rise, twist)
    print(emdid, amyloid_id, resolution, difference)


#import matplotlib.pyplot as plt
#
#plt.imshow(cylinder_projection_img, cmap='gray')
#plt.show()