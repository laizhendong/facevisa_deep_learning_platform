import os
import sys
import json
import io
import glob
# make sure that the cwd() in the beginning is the location of the python script (so that every path makes sense)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# change directory to the one with the files to be changed
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
print(parent_path)
GT_PATH = os.path.join(parent_path, 'datasets', 'voc_data', 'stain', 'imgxml')
print(GT_PATH)
os.chdir(GT_PATH)
# old files (json format) will be moved to a "backup" folder
## create the backup dir if it doesn't exist already
if not os.path.exists("backup"):
  os.makedirs("backup")  
# 2. move old file (json format) to backup
json_list = glob.glob('*.json')
if len(json_list) == 0:   
    print("Error: no .json files found in train")
    sys.exit()
for tmp_file in json_list:
    os.rename(tmp_file, os.path.join("backup", tmp_file))
print("Move completed!")