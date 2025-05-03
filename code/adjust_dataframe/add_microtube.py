import numpy as np
import pandas as pd
import os


output_pd_path = './EMDB_validation.csv'
suboptimal_pd_path = './files/suboptimal.csv'

df_data = pd.read_csv(output_pd_path)
df_suboptimal = pd.read_csv(suboptimal_pd_path)

# Step 2: Merge df2 with df1 on 'id'
df_suboptimal = df_suboptimal.merge(df_data[['emdb_id', 'group']], on='emdb_id', how='left')

df_suboptimal.to_csv(suboptimal_pd_path,index=False)

print(df_suboptimal['group'])
