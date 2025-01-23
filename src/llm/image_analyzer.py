import numpy as np
import requests
from litellm import completion
import json_repair
import re

from src.utils import convert_to_base64
from src.llm.prompts import system_message, final_rag_message
from src.utils.convert_to_base64 import resize_base64


class ImageAnalyzer:
    def __init__(self, dataset, model, add_classes=True):
        self.dataset = dataset
        self.model = model
        self.add_classes = add_classes
        # self.task_prompt = "Return the result in json format: {'y_pred': y_pred, 'explanation': explanation}. The explanation should be brief"

    def _create_base_prompt(self, modality: str, labels: list) -> str:
        prompt = system_message.format(modality=modality, labels=labels)

        return prompt

    def _create_few_shot_message(self, similar_images):
        messages = []

        for example in similar_images:
            image = (
                example["data"]["img"]
                if "img" in example["data"]
                else example["data"]["image"]
            )
            img = resize_base64(image)

            if self.add_classes:
                label = example["data"]["label"]

                prompt = f"""Example:
                                class: {label}
                                    """
            else:
                prompt = "Example simillar images:"

            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"},
                    },
                ],
            }
            messages.append(message)

        return messages

    def _description_message(self, modality, classes, few_shot=False):
        prompt = self._create_base_prompt(modality, classes)

        if few_shot:
            prompt += "\n Here are some examples to learn from:"

        message = [
            {
                "role": "system",
                "content": prompt,
            }
        ]
        return message

    def _create_rag_final_message(self, img_str):
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": final_rag_message},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_str}"},
                },
            ],
        }
        return message

    def _get_message(self, img_str, similar_images=[]):
        modality = self.dataset.modality
        classes = self.dataset.classes
        few_shot = True if len(similar_images) > 0 else False
        messages = self._description_message(modality, classes, few_shot)
        if len(similar_images) > 0:
            messages.extend(self._create_few_shot_message(similar_images))
        messages.append(self._create_rag_final_message(img_str))

        return messages

    def _get_aux_imgs_info(self, similar_imgs):
        aux_images = {}
        for idx, example in enumerate(similar_imgs):
            aux_images[f"aux_{idx}_id"] = (example["data"]["id"],)
            if self.add_classes:
                aux_images[f"aux_{idx}_label"] = example["data"]["label"]
        return aux_images

    def analyze_img(self, img_str, similar_imgs, verbose=False):
        messages = self._get_message(img_str, similar_imgs)

        response = completion(
            model=self.model.name, api_key=self.model.api_key, messages=messages
        )
        result = response.choices[0].message.content
        if verbose:
            print(result)
        try:
            pred = json_repair.loads(self._extract_json_string(result))
            assert pred["y_pred"] in self.dataset.classes
        except Exception as e:
            print(e)
            return None
            # pred = {"y_pred": "error", "explanation": result}
        aux_images_dict = self._get_aux_imgs_info(similar_imgs)
        json_line = {**pred, **aux_images_dict}
        return json_line

    def _extract_json_string(self, text: str) -> str:
        # Try to find JSON enclosed in triple backticks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)

        if json_match:
            json_str = json_match.group(1)
        else:
            # If not found, try to find JSON without backticks
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                json_str = json_match.group(0)
            else:
                return ""

        return json_str
