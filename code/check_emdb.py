import pandas as pd
import numpy as np

def get_emdb_helical_structures():

    df = pd.read_csv('https://www.ebi.ac.uk/emdb/api/search/structure_determination_method:"helical"?rows=1000000&wt=csv&download=true&fl=emdb_id,structure_determination_method,resolution,image_reconstruction_helical_delta_z_value,image_reconstruction_helical_delta_phi_value,image_reconstruction_helical_axial_symmetry_details',index_col=False)
    df = df.rename(columns={"structure_determination_method": "method", "image_reconstruction_helical_delta_z_value": "rise", "image_reconstruction_helical_delta_phi_value": "twist", "image_reconstruction_helical_axial_symmetry_details": "csym"}).reset_index()
    #df["Pitch"] = np.abs(360/df["twist"].astype(float).abs()*df["rise"].astype(float).abs())
    return df

helical_pd = get_emdb_helical_structures()

helical_pd = helical_pd.drop(columns=['index','method'])
print(helical_pd.columns)

helical_pd = helical_pd.rename(columns={'resolution':'resolution (Å)','rise':'rise_deposited (Å)', 'twist':'twist_deposited (°)', 'csym': 'csym_deposited'})

helical_pd = helical_pd.reindex(columns=['emdb_id','group','resolution (Å)','rise_deposited (Å)','twist_deposited (°)','csym_deposited','rise_curated (Å)','twist_curated (°)','csym_curated', 'vector difference','axes order','cc_emdb','cc_curated','validated','update','reason','reason 1','reason 2'])

# rise_curated (Å),twist_curated (°),csym_curated,axes order,cc_emdb,cc_curated,validated,update,reason

## add hi3d link
#link = 'https://helical-indexing-hi3d.streamlit.app/?emd_id=emd'
#helical_pd['HI3D_link'] = link+helical_pd['emdb_id'].str[3:]

data_path = './EMDB_validation.csv'
curated_pd = pd.read_csv(data_path)
non_curated_pd = helical_pd[~helical_pd['emdb_id'].isin(curated_pd['emdb_id'])]

# save the csv file
save_path = './files/need_curation.csv'
non_curated_pd.to_csv(save_path, index=False)