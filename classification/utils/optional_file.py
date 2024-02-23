import yaml
def load_optional_file(cfg, optional_file):
    with open(optional_file,"r") as f:
        optional_data = yaml.load(f,yaml.FullLoader)
    try:
        cfg.DATA.AUGMENT.RESIZE = optional_data['DATA']["AUGMENT"]["RESIZE"]
    except:
        print("!!use default RESIZE")
    try:
        cfg.SOLVER.LR_BASE = optional_data['SOLVER']["LR_BASE"]
    except:
        print("!!use default LR_BASE optional")
    try:
        cfg.SOLVER.EPOCHS = optional_data['SOLVER']['EPOCHS']
    except:
        print("!!use default EPOCHS")
    try:
        cfg.SOLVER.BATCH_SIZE = optional_data['SOLVER']["BATCH_SIZE"]
    except:
        print("!!use default BATCH_SIZE")
    return cfg