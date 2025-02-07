import pandas as pd
import numpy as np

def get_emdb_helical_structures():

    df = pd.read_csv('https://www.ebi.ac.uk/emdb/api/search/structure_determination_method:"helical"?rows=1000000&wt=csv&download=true&fl=emdb_id,structure_determination_method,resolution,image_reconstruction_helical_delta_z_value,image_reconstruction_helical_delta_phi_value,image_reconstruction_helical_axial_symmetry_details')
    df = df.rename(columns={"structure_determination_method": "method", "image_reconstruction_helical_delta_z_value": "rise", "image_reconstruction_helical_delta_phi_value": "twist", "image_reconstruction_helical_axial_symmetry_details": "csym"}).reset_index()
    #df["Pitch"] = np.abs(360/df["twist"].astype(float).abs()*df["rise"].astype(float).abs())
    return df

helical_pd = get_emdb_helical_structures()
data_path = './EMDB_validation.csv'
curated_pd = pd.read_csv(data_path)
non_curated_pd = helical_pd[~helical_pd['emdb_id'].isin(curated_pd['emdb_id'])]

link = 'https://helical-indexing-hi3d.streamlit.app/?emd_id=emd'
non_curated_pd['HI3D_link'] = link+helical_pd['emdb_id'].str[3:]

save_path = './files/need_curation.csv'

non_curated_pd.to_csv(save_path)