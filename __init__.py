'''
import json
json_file = "./test.json"
path = "./blind/2019/val/*.jpg"
import glob
file_names = glob.glob(path)
import os
file_names = [filename.split("\\")[-1] for filename in file_names]
print(file_names)
with open(json_file) as f:
    json_data = json.load(f)
images = json_data["images"]
for image in images:
    if image["file_name"] not in file_names:
        print(image["file_name"])
'''