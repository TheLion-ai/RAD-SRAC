import requests
import cv2
from qdrant_client.models import PointStruct, VectorParams, Distance
import os

from src.utils import convert_to_base64
from src.config.qdrant import client
from src.config.encoder import encoders
from datasets import load_dataset


class Retreiver:
    def __init__(self, encoder, collection_name: str, n_images: int = 5):
        self.encoder = encoder
        self.client = client
        self.collection_name = collection_name
        self.n_images = n_images

    def get_similar_imgs(self, img_str: list[str]):
        embedings = self.encoder.encode(img_str)[0]

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=embedings,
            with_payload=True,
            limit=self.n_images,
        ).points

        results = []
        for p in search_result:
            results.append(
                {
                    "id": p.id,
                    "data": p.payload,
                }
            )

        return results

    def _create_id(self, umie_path):
        return int(os.path.basename(umie_path)[0:-4].replace("_", "0").lstrip("0"))

    def _upload_batch(self, batch):
        img_paths = batch["umie_path"]
        labels = batch["source_labels"]

        images = []

        for img_path in img_paths:
            # print(img_path)
            img = cv2.imread(os.path.join("data", img_path))
            if img is None:
                print(img_path)
                del img_paths[img_paths.index(img_path)]
                continue
            base_img = convert_to_base64(img)
            images.append(base_img)
        img_embeddings = self.encoder.encode(images)

        points = []
        ids = []

        for img_path, img_embedding, img, label in zip(
            img_paths, img_embeddings, images, labels
        ):
            try:
                id = self._create_id(img_path)
                if id in ids:
                    print(f"error: {id}")
                    print(f"error: {img_path}")
                    break
                ids.append(id)
                label = label[0]
                points.append(
                    PointStruct(
                        id=id,
                        vector=img_embedding,
                        payload={"label": label, "id": id, "img": img},
                    )
                )
            except Exception as e:
                print(f"error: {img_path} {e}")

        operation_info = self.client.upsert(
            collection_name=self.collection_name, wait=True, points=points
        )

    def upload_dataset(self, dataset, batch_size=8, replace_collection=False):
        if client.collection_exists(self.collection_name):
            if replace_collection:
                client.delete_collection(self.collection_name)

                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.encoder.vector_size, distance=Distance.COSINE
                    ),
                )
        else:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.encoder.vector_size, distance=Distance.COSINE
                ),
            )

        dataset = load_dataset("json", data_files=dataset.train_file)["train"]
        dataset = dataset.select(range(20))
        dataset.map(self._upload_batch, batch_size=batch_size, batched=True)
