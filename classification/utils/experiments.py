import yaml,os
import traceback

KEYS_RELATED_PATH = ["hyparam_file","optional_file","label2id_file","id2name_file"]

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..") 

def _expand_path(path):
    if path[0] == '.':
        return os.path.join(PROJECT_ROOT,path)
    return path

def _try_expand_paths(config_data):
    for key in KEYS_RELATED_PATH:
        if key not in config_data.keys():
            continue
        config_data[key] = _expand_path(config_data[key])
    return config_data
    

def _apply_custom_hyparam_file(config_data):
    if "hyparam_file_custom" in config_data.keys() and os.path.exists(config_data['hyparam_file_custom']):
        config_data['hyparam_file'] = _expand_path(config_data['hyparam_file_custom'])
        config_data['hyparam_file_custom'] = None   
        print("!!use custom hyparam file")  
    return config_data

def load_config(exp_conf_file,module_name):
    try:
        with open(exp_conf_file,"r") as f:
            cfg_data = yaml.load(f,yaml.FullLoader)
        cfg_data = cfg_data[module_name]
        cfg_data = _try_expand_paths(cfg_data) 
        cfg_data = _apply_custom_hyparam_file(cfg_data)          
    except:
        cfg_data = None
        traceback.print_exc()
    return cfg_data
        
        
        
    
    