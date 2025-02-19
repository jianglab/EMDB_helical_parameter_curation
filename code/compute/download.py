import numpy as np
import pandas as pd
import sys, os


def get_direct_url(url):
    import re
    if url.startswith("https://drive.google.com/file/d/"):
        hash = url.split("/")[5]
        return f"https://drive.google.com/uc?export=download&id={hash}"
    elif url.startswith("https://app.box.com/s/"):
        hash = url.split("/")[-1]
        return f"https://app.box.com/shared/static/{hash}"
    elif url.startswith("https://www.dropbox.com"):
        if url.find("dl=1")!=-1: return url
        elif url.find("dl=0")!=-1: return url.replace("dl=0", "dl=1")
        else: return url+"?dl=1"
    elif url.find("sharepoint.com")!=-1 and url.find("guestaccess.aspx")!=-1:
        return url.replace("guestaccess.aspx", "download.aspx")
    elif url.startswith("https://1drv.ms"):
        import base64
        data_bytes64 = base64.b64encode(bytes(url, 'utf-8'))
        data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
        return f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    else:
        return url

def get_file_size(url):
    import requests
    response = requests.head(url)
    if 'Content-Length' in response.headers:
        file_size = int(response.headers['Content-Length'])
        return file_size
    else:
        return None

def download_file_from_url(url):
    import tempfile
    import requests
    #print(url)
    try:
        filesize = get_file_size(url)
        local_filename = url.split('/')[-1]
        suffix = '.' + local_filename
        fileobj = tempfile.NamedTemporaryFile(suffix=suffix)
        msg = f'Downloading {url}'
        if filesize is not None:
            msg += f" ({filesize/2**20:.1f} MB)"
        with requests.get(url) as r:
            r.raise_for_status()  # Check for request success
            fileobj.write(r.content)
        return fileobj
    except Exception as e:
        return None

def get_3d_map_from_file(filename):
    if filename.endswith(".gz"):
        filename_final = filename[:-3]
        import gzip, shutil
        with gzip.open(filename, 'r') as f_in, open(filename_final, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        filename_final = filename
    import mrcfile
    with mrcfile.open(filename_final,mode="r") as mrc:
        data = mrc.data
        map_crs = [int(mrc.header.mapc), int(mrc.header.mapr), int(mrc.header.maps)]
        apix = mrc.voxel_size.x.item()
        is3d = mrc.is_volume() or mrc.is_volume_stack()
    return data, map_crs, apix

def get_emdb_map_url(emdid):
    emdid_number = emdid.lower().split("emd-")[-1]
    server = "https://files.wwpdb.org/pub"    # Rutgers University, USA
    #server = "https://ftp.ebi.ac.uk/pub/databases" # European Bioinformatics Institute, England
    #server = "http://ftp.pdbj.org/pub" # Osaka University, Japan
    url = f"{server}/emdb/structures/EMD-{emdid_number}/map/emd_{emdid_number}.map.gz"
    return url

def get_half_maps_url(emdid):
    emdid_number = emdid.lower().split("emd-")[-1]
    server = "https://files.wwpdb.org/pub"    # Rutgers University, USA
    #server = "https://ftp.ebi.ac.uk/pub/databases" # European Bioinformatics Institute, England
    #server = "http://ftp.pdbj.org/pub" # Osaka University, Japan
    url_1 = f"{server}/emdb/structures/EMD-{emdid_number}/other/emd_{emdid_number}_half_map_1.map.gz"
    url_2 = f"{server}/emdb/structures/EMD-{emdid_number}/other/emd_{emdid_number}_half_map_2.map.gz"
    return url_1, url_2

def get_half_maps(emdid):
    url_1, url_2 = get_half_maps_url(emdid)
    try:
        data1, map_crs1, apix1 = get_3d_map_from_url(url_1)
        data1 = correct_data(map_crs1, data1, emdid)

        data2, map_crs2, apix2 = get_3d_map_from_url(url_2)
        data2 = correct_data(map_crs2, data2, emdid)
        return data1, data2, apix1, apix2
    except:
        print(f'{emdid} has no half maps')
        return None
    
    return map_1, map_2

def get_3d_map_from_url(url):
    url_final = get_direct_url(url)    # convert cloud drive indirect url to direct url
    with download_file_from_url(url_final) as fileobj:
        if fileobj is None:
            return None
        data = get_3d_map_from_file(fileobj.name)
    file_to_remove = fileobj.name[:-3]
    if os.path.exists(file_to_remove):
        os.unlink(file_to_remove)
    return data


def change_mrc_map_crs_order(data, current_order, target_order=[1, 2, 3]):
    if current_order == target_order: return data
    map_crs_to_np_axes = {1:2, 2:1, 3:0}
    current_np_axes_order = [map_crs_to_np_axes[int(i)] for i in current_order]
    target_np_axes_order = [map_crs_to_np_axes[int(i)] for i in target_order]
    import numpy as np
    ret = np.moveaxis(data, current_np_axes_order, target_np_axes_order)
    return ret

def correct_data(map_crs, data, emdb_id):
    if map_crs != [1, 2, 3]:
        print(f'{emdb_id} order is wrong, need corrected')
        map_crs_to_xyz = {1:'x', 2:'y', 3:'z'}
        xyz = ','.join([map_crs_to_xyz[int(i)] for i in map_crs])
        try:
            target_map_axes_order = ['x','y','z']
            assert len(target_map_axes_order) == 3
            xyz_to_map_crs = {'x':1, 'y':2, 'z':3}
            target_map_crs = [xyz_to_map_crs[a] for a in target_map_axes_order]
        except:
            target_map_crs = [1, 2, 3]
        data = change_mrc_map_crs_order(data=data, current_order=map_crs, target_order=target_map_crs)

    return data

def get_correct_data_url(emdid):
    emdb_url = get_emdb_map_url(emdid)
    try:
        data, map_crs, apix = get_3d_map_from_url(emdb_url)
        data = correct_data(map_crs, data, emdid)
        return data, apix
    except:
        print(f'{emdid} has no map')
        return None

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

def is_amyloid(params, cutoff=6):
    if "twist" in params and "rise" in params:
        twist = params["twist"]
        rise = params["rise"]
        r = np.hypot(twist, rise)
        if r<cutoff: return True
        twist2 = abs(twist)-180
        r = np.hypot(twist2, rise)
        if r<cutoff: return True
    if "sample" in params:
        sample = params["sample"].lower()
        for target in "tau synuclein amyloid tdp-43".split():
            if sample.find(target)!=-1: return True
    return False