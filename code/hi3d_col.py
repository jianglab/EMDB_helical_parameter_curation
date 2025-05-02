import numpy as np
import pandas as pd
import os

output_pd_path = './EMDB_validation.csv'
output_pd_path_1 = './EMDB_validation_hi3d.csv'

df = pd.read_csv(output_pd_path)
df['HI3D_link'] = 'https://helical-indexing-hi3d.streamlit.app/?emd_id=emd-' + df['emdb_id'].str.extract(r'EMD-(\d+)', expand=False)

df.to_csv(output_pd_path_1,index=False)