import jsonlines
import os
import cv2
from src.utils import convert_to_base64
from src.vector_db.retriever import Retreiver
from src.llm.image_analyzer import ImageAnalyzer
from time import sleep as wait
from alive_progress import alive_bar
import time
import math
from src.utils.metrics import get_scores_for_df


class DatasetAnalyzer:
    def __init__(
        self,
        dataset,
        model,
        few_shot,
        retriever,
        verbose=False,
        add_classes=True,
        n_images=5,
        max_retries=5,
    ):
        self.dataset = dataset
        self.model = model
        self.add_classes = add_classes
        self.n_images = n_images
        self.retriever = retriever

        self.input_path, self.output_path = self._get_jsonl_paths(few_shot)
        self.val_data = self._get_val_data(self.input_path)
        self.image_analyzer = ImageAnalyzer(dataset, model, add_classes)
        self.preds = []
        self.few_shot = few_shot
        self.verbose = verbose
        self.max_retries = max_retries

    def _get_jsonl_paths(self, few_shot):
        input_path = self.dataset.val_file
        if self.retriever.collection_name == "ROCO":
            task = "roco"
        elif few_shot:
            task == "rag"
        else:
            task == "raw"
        # task = "rag" if few_shot else "raw"
        model = self.model.name.replace("/", "_")
        output_path = os.path.join(
            "results",
            f"{self.dataset.name}_{task}_{model}_{self.add_classes}_{self.n_images}.jsonl",
        )
        return input_path, output_path

    def _get_val_data(self, json_path):
        val_data = []
        with jsonlines.open(json_path, mode="r") as reader:
            for obj in reader:
                sample = {
                    "umie_path": obj["umie_path"],
                    "y_true": obj["source_labels"][0],
                }
                val_data.append(sample)
        return val_data

    def _get_done_preds(self, path):
        preds = []
        if os.path.exists(path):
            with jsonlines.open(path, mode="r") as reader:
                for obj in reader:
                    preds.append(obj)
        return preds

    def _get_img_path(self, img_path):
        return os.path.join("data", img_path)

    def analyze_imgs(self, batch_size=10):
        from concurrent.futures import ThreadPoolExecutor

        self.preds = self._get_done_preds(self.output_path)
        done_paths = [pred["umie_path"] for pred in self.preds]
        self.val_data = [d for d in self.val_data if d["umie_path"] not in done_paths]

        def process_single_image(sample):
            if sample["umie_path"] in done_paths:
                return None

            img = cv2.imread(self._get_img_path(sample["umie_path"]))
            img_str = convert_to_base64(img)

            if self.few_shot:
                similar_imgs = self.retriever.get_similar_imgs([img_str])
            else:
                similar_imgs = []

            for attempt in range(self.max_retries):
                # if attempt> 0 :
                # time.sleep(2)
                try:
                    pred = self.image_analyzer.analyze_img(
                        img_str, similar_imgs, self.verbose
                    )
                    if pred is not None:
                        return {**sample, **pred}
                    else:
                        raise Exception("Error in image processing")
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        print(
                            f"Failed to process {sample['umie_path']} after {self.max_retries} attempts: {e}"
                        )
                    else:
                        print(
                            f"Attempt {attempt + 1} failed for {sample['umie_path']}: {e}. Retrying..."
                        )
            return None

        # Process images in batches
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            with alive_bar(total=len(self.val_data)) as bar:
                for i in range(0, len(self.val_data), batch_size):
                    batch = self.val_data[i : i + batch_size]
                    results = list(executor.map(process_single_image, batch))
                    self.preds.extend([r for r in results if r is not None])
                    # wait(5)  # Wait after each batch
                    bar(batch_size)

        # Save results
        with jsonlines.open(self.output_path, mode="w") as writer:
            for obj in self.preds:
                writer.write(obj)

        print(get_scores_for_df(self.output_path))
