import numpy as np
import pandas as pd
import sys, os

from compute.download import get_correct_data_url, get_emdb_parameters, is_amyloid
from compute.symmetrization import apply_sym


#data_path = './files/need_curation.csv'
data_path = './EMDB_validation.csv'

data_pd = pd.read_csv(data_path, dtype='str')

emdb_list = list(data_pd[data_pd['reason']=='check']['emdb_id'].str[4:])
emdb_list = ['17678']
# 18432

print(len(emdb_list))

for i in range(len(emdb_list)):

    emdid = emdb_list[i]

    emdid_full = 'EMD-'+str(emdid)

    value_list = ['rise_deposited (Å)', 'twist_deposited (°)','curated_rise (Å)', 'curated_twist (°)']

    rise_original, twist_original, rise, twist = list(data_pd.loc[data_pd['emdb_id']==emdid_full, value_list].iloc[0])
    print(rise_original, twist_original, rise, twist)


    data, apix = get_correct_data_url(emdid)
    cc, cc_hi3d = apply_sym(data, apix, float(rise_original), float(twist_original), float(rise), float(twist), only_original=False, n_rise=10)
    print(emdid, cc, cc_hi3d)

    data_pd.loc[data_pd['emdb_id'] == 'EMD-' + str(emdid),['cc_emdb','cc_curated']] = (cc, cc_hi3d)

data_pd.to_csv(data_path, index=False)