import glob
import json


def create_weak_train():
    weak_train_json_path = "E:/MS_RCNN/datasets/blind/2019/annotations/weak-train/*.json"
    json_files = glob.glob(weak_train_json_path)
    total_images = []
    total_annotations = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            json_data = json.load(f)
            image = json_data["images"]
            annotations = json_data["annotations"]
            total_images.extend(image)
            total_annotations.extend(annotations)
    weak_json_data = {}
    weak_json_data["images"] = total_images
    weak_json_data["annotations"] = total_annotations
    json.dump(weak_json_data, open("E:/MS_RCNN/datasets/blind/2019/annotations/instances_weak_train2019.json", "w"), indent=4)


def combine_dataset():
    weak_train_json_path = "E:/MS_RCNN/datasets/blind/2019/annotations/instances_weak_val2019.json"
    train_json_path = "E:/MS_RCNN/datasets/blind/2019/annotations/instances_val2019.json"
    with open(weak_train_json_path, "r") as f:
        weak_train_json_data = json.load(f)
    with open(train_json_path, "r") as f:
        train_json_data = json.load(f)
    total_images = []
    total_annotations = []
    total_categories = []
    total_saliencies = []
    total_categories.extend(weak_train_json_data["categories"])
    total_saliencies.extend(train_json_data["saliencies"])
    total_images.extend(train_json_data["images"])
    total_annotations.extend(train_json_data["annotations"])
    total_images.extend(weak_train_json_data["images"])
    total_annotations.extend(weak_train_json_data["annotations"])
    total_data = {}
    total_data["images"] = total_images
    total_data["annotations"] = total_annotations
    total_data["categories"] = total_categories
    total_data["saliencies"] = total_saliencies
    json.dump(total_data, open("E:/MS_RCNN/datasets/blind/2019/annotations/instances_total_train2019.json", "w"), indent=4)
combine_dataset()