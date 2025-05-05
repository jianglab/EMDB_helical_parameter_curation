import numpy as np
import pandas as pd
import os

output_pd_path = './EMDB_validation.csv'
output_pd_path_1 = './EMDB_validation_hi3d.csv'

df = pd.read_csv(output_pd_path)
print(df.columns)

#https://helical-indexing-hi3d.streamlit.app/?emd_id=emd-14057&rise=2.368&twist=179.597&csym=1&rise2=2.367&twist2=-179.6&csym2=1

df['HI3D_link'] = 'https://helical-indexing-hi3d.streamlit.app/?emd_id=emd-' + df['emdb_id'].str.extract(r'EMD-(\d+)', expand=False) \
    + '&rise=' + df['rise_deposited (Å)'].astype(str) + '&twist=' + df['twist_deposited (°)'].astype(str) + '&csym=' + df['csym_deposited'].str[1:]\
        + '&rise2=' + df['rise_curated (Å)'].astype(str) + '&twist2=' + df['twist_curated (°)'].astype(str) + '&csym2=' + df['csym_deposited'].str[1:]

df.to_csv(output_pd_path_1,index=False)