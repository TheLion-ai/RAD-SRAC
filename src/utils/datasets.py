import os
import jsonlines


def create_id(umie_path):
    return int(os.path.basename(umie_path)[0:-4].replace("_", "0").lstrip("0"))


def create_labels(dataset_name: str) -> dict:
    labels = {}
    img_paths = []
    json_path = f"data/{dataset_name}/train.jsonl"
    with jsonlines.open(json_path, mode="r") as reader:
        for obj in reader:
            id = create_id(obj["umie_path"])
            labels[id] = obj["source_labels"][0]
            img_paths.append(os.path.join("data", obj["umie_path"]))
