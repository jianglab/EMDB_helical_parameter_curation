import numpy as np
import pandas as pd
import os

def dict_recursive_search(d, key, default=None):
    stack = [iter(d.items())]
    while stack:
        for k, v in stack[-1]:
            if k == key:          
                return v
            elif isinstance(v, dict):
                stack.append(iter(v.items()))
                break
        else:
            stack.pop()
    return default

def get_emdb_parameters(emd_id):
    try:
        emd_id2 = ''.join([s for s in str(emd_id) if s.isdigit()])
        url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emd_id2}/header/emd-{emd_id2}.xml"
        from urllib.request import urlopen
        with urlopen(url) as response:
            xml_data = response.read()
        import xmltodict
        data = xmltodict.parse(xml_data)
    except:
        return None
    ret = {}
    try:
      ret['sample'] = data['emd']['sample']['name']
      ret["method"] = data['emd']['structure_determination_list']['structure_determination']['method']
      dimensions = data['emd']['map']['dimensions']
      ret["nz"] = int(dimensions["sec"])
      ret["ny"] = int(dimensions["row"])
      ret["nx"] = int(dimensions["col"])
      res_dict = dict_recursive_search(data, 'resolution')
      if res_dict:
        ret["resolution"] = float(res_dict['#text'])
      if ret["method"] == 'helical':
          #ret["resolution"] = float(data['emd']['structure_determination_list']['structure_determination']['helical_processing']['final_reconstruction']['resolution']['#text'])
          helical_parameters = data['emd']['structure_determination_list']['structure_determination']['helical_processing']['final_reconstruction']['applied_symmetry']['helical_parameters']
          #assert(helical_parameters['delta_phi']['@units'] == 'deg')
          #assert(helical_parameters['delta_z']['@units'] == 'Å')
          ret["twist"] = float(helical_parameters['delta_phi']['#text'])
          ret["rise"] = float(helical_parameters['delta_z']['#text'])
          ret["csym"] = int(helical_parameters['axial_symmetry'][1:])
    finally:
      return ret
    
def is_microtubule(params, cutoff=6):
    if "sample" in params:
        sample = params["sample"].lower()
        for target in "microtubule microtubules MT tubulin".split():
            if sample.find(target)!=-1: 
                return True
    return False

output_pd_path = './EMDB_validation.csv'
df_data = pd.read_csv(output_pd_path)

#df_data = df_data[(df_data['emdb_id'] == 'EMD-43868') | (df_data['emdb_id'] == 'EMD-14238')]

print(len(df_data))

for i in range(len(df_data)):
    emdid_full = df_data['emdb_id'].iloc[i]
    group = df_data['group'].iloc[i]
    emdid = emdid_full[4:]

    if group == 'amyloid':
        #print(f'{emdid_full} is amyloid')
        continue

    params = get_emdb_parameters(emdid)

    #print(params["sample"])

    if params is None:
        print(f'{emdid_full} has no parameters')
        continue
    if is_microtubule(params):
        print(f'{emdid_full} is microtubule')
        print(params["sample"])
        df_data.loc[i,'group'] = 'microtubule'
    else:
        #print(f'{emdid_full} is not microtubule')
        df_data.loc[i,'group'] = 'non-amyloid'

df_data.to_csv(output_pd_path, index=False)