import requests


class MedImageInsightEncoder:
    def __init__(self, api_link):
        self.api_link = api_link
        self.vector_size = 1024

    def encode(self, img_str: list[str]):
        prediction_payload = {
            "images": img_str,
        }
        response = requests.post(self.api_link + "/encode", json=prediction_payload)
        return response.json()["image_embeddings"]
