import numpy as np
import pandas as pd
import os

output_pd_path = './files/EMDB_validation.csv'
output_pd_path_excel = './EMDB_validation.xlsx'
output_pd_path_csv = './EMDB_validation.csv'

df = pd.read_csv(output_pd_path)
print(df.columns)

#https://helical-indexing-hi3d.streamlit.app/?emd_id=emd-14057&rise=2.368&twist=179.597&csym=1&rise2=2.367&twist2=-179.6&csym2=1

df['HI3D_link'] = 'https://helical-indexing-hi3d.streamlit.app/?emd_id=emd-' + df['emdb_id'].str.extract(r'EMD-(\d+)', expand=False) \
    + '&rise=' + df['rise_deposited (Å)'].astype(str) + '&twist=' + df['twist_deposited (°)'].astype(str) + '&csym=' + df['csym_deposited'].str[1:]\
        + '&rise2=' + df['rise_validated (Å)'].astype(str) + '&twist2=' + df['twist_validated (°)'].astype(str) + '&csym2=' + df['csym_validated'].str[1:]

df2 = df.copy()

df['HI3D_link'] = df['HI3D_link'].apply(lambda x: f'=HYPERLINK("{x}", "Link")' if pd.notnull(x) else '')



# Write to Excel with openpyxl engine
with pd.ExcelWriter(output_pd_path_excel, engine='openpyxl') as writer:
    df.to_excel(writer, index=False)

df2.to_csv(output_pd_path_csv, index=False)