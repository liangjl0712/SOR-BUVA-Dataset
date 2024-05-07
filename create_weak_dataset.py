import os
import shutil
import json
import numpy as np
from pycocotools.coco import COCO

CATNMS = ["person", "stop sign", "backpack", "umbrella", "handbag",
          "tennis racket", "cup", "fork", "spoon", "banana",
          "apple", "orange", "chair", "laptop", "mouse",
          "microwave", "refrigerator", "clock", "vase", "scissors",
          "hair drier", "toothbrush", "cat", "dog", "horse",
          "suitcase", "wine glass", "knife", "toilet", "cell phone"]

COCO_CATIDS_TO_OURS = {1: 1,
                       13: 2,
                       27: 3,
                       28: 4,
                       31: 5,
                       43: 6,
                       47: 7,
                       48: 8,
                       50: 9,
                       52: 10,
                       53: 11,
                       55: 12,
                       62: 13,
                       73: 14,
                       74: 15,
                       78: 16,
                       82: 17,
                       85: 18,
                       86: 19,
                       87: 20,
                       89: 21,
                       90: 22,
                       17: 23,
                       18: 24,
                       19: 25,
                       33: 26,
                       46: 27,
                       49: 28,
                       70: 29,
                       77: 30}

# coco = COCO("E:\\Datasets\\COCO\\2017\\annotations\\instances_val2017.json")
coco = COCO("E:\\MS_RCNN\\datasets\\blind\\2019\\annotations\\instances_val2019.json")
catIds = coco.getCatIds(catNms=CATNMS)
# import pprint
# pprint.pprint(coco.cats)

from glob import glob
HAS_IMAGE_PATH = os.path.join(os.getcwd(), "blind", "2019", "val", "*.jpg")
def image_path_to_image_id(image_path):
    return int(os.path.basename(image_path).split(".")[0])
HAS_LABEL_IMAGE_ID = set(map(image_path_to_image_id, glob(HAS_IMAGE_PATH)))  # 要排除的图像ID
TOTAL_IMAGE_PATH = "E:\\Datasets\\COCO\\2017\\val\\*.jpg"
TOTAL_IMAGE_ID = set(map(image_path_to_image_id, glob(TOTAL_IMAGE_PATH)))
CHOICEABLE_IMAGE_ID = list(TOTAL_IMAGE_ID.difference(HAS_LABEL_IMAGE_ID))
CHOICED_IDS = []
while len(CHOICED_IDS) < 800:
    json_data = {}
    annIds = []
    while len(annIds) == 0:
        current_image_id = np.random.choice(CHOICEABLE_IMAGE_ID, 1)[0]
        CHOICEABLE_IMAGE_ID.remove(current_image_id)
        annIds = coco.getAnnIds(imgIds=[current_image_id], catIds=catIds, iscrowd=None)
    image_data = coco.loadImgs(ids=[current_image_id])
    current_image_path = TOTAL_IMAGE_PATH.replace("*.jpg", image_data[0]["file_name"])
    object_image_path = "E:/MS_RCNN/datasets/blind/2019/weak-val/{}".format(image_data[0]["file_name"])
    shutil.copyfile(current_image_path, object_image_path)
    image_data[0]["file_path"] = object_image_path
    annotations_data = coco.loadAnns(annIds)
    for annIdx in range(len(annotations_data)):
        annotation_data = annotations_data[annIdx]
        annotation_data["category_id"] = COCO_CATIDS_TO_OURS[annotation_data["category_id"]]
    json_data["images"] = image_data
    json_data["annotations"] = annotations_data
    json.dump(json_data, open(object_image_path.replace(".jpg", ".json"), "w"), indent=4)
    CHOICED_IDS.append(current_image_id)
