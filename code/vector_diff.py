import numpy as np
import pandas as pd
import sys, os
import math

from compute.download import get_correct_data_url, get_emdb_parameters, is_amyloid
from compute.calculate_radius import rmin_max_estimateion, vector_diff


data_path = './EMDB_validation.csv'
data_path_save = './EMDB_validation_vector_diff.csv'

data_pd = pd.read_csv(data_path, dtype='str')

emdb_list = list(data_pd['emdb_id'].str[4:])
#emdb_list = ['17680']


if os.path.exists(data_path_save) is False:
    data_pd_validated = pd.DataFrame(columns=data_pd.columns)
    data_pd_validated.to_csv(data_path_save, index=False)



for i in range(len(emdb_list)):

    emdid = emdb_list[i]
    emdid_full = 'EMD-'+str(emdid)

    if os.path.exists(data_path_save):
        data_pd_validated = pd.read_csv(data_path_save, dtype='str')
        if emdid_full in list(data_pd_validated['emdb_id']):
            print(f'{emdid_full} has been checked')
            continue
    

    value_list = ['rise_deposited (Å)', 'twist_deposited (°)','csym_deposited','curated_rise (Å)', 'twist_curated (°)']

    rise_original, twist_original,csym_original, rise, twist = list(data_pd.loc[data_pd['emdb_id']==emdid_full, value_list].iloc[0])
    values = rise_original, twist_original, rise, twist
    
    

    if any(math.isnan(float(v)) for v in values):
        data_pd_validated = pd.concat([data_pd_validated, data_pd[data_pd['emdb_id'] == 'EMD-' + str(emdid)]],ignore_index=True)
        data_pd_validated.to_csv(data_path_save, index=False)
        print(f'{emdid_full} has no value, just continue')
        continue

    print(rise_original, twist_original,csym_original, rise, twist)

    if str(csym_original)[0] in ('C', 'D'):
        csym_original = csym_original[1:]
    
    if math.isnan(float(csym_original)):
        csym_original = 1
    
    data, apix = get_correct_data_url(emdid)
    rmin_auto, rmax_auto, centroid = rmin_max_estimateion(data)
    radius = centroid*apix
    difference = vector_diff(rise_original=float(rise_original), twist_original=float(twist_original), csym_original=int(csym_original), 
                             rise_predicted=float(rise), twist_predicted=float(twist), radius=centroid)
    

    data_pd.loc[data_pd['emdb_id'] == 'EMD-' + str(emdid),['vector difference']] = difference

    data_pd_validated = pd.concat([data_pd_validated, data_pd[data_pd['emdb_id'] == 'EMD-' + str(emdid)]],ignore_index=True)
    data_pd_validated.to_csv(data_path_save, index=False)

    print(i, emdid_full, rise_original, twist_original, csym_original, rise, twist)
    print(emdid_full, difference)


#data_pd.to_csv(data_path_save, index=False)