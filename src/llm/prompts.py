system_message = """You are a medical expert.
Analize the {modality} image and classify if there is a tumor present.
Select the appropriate class from {labels}."""

final_rag_message = """
Now, please analyze the new image and provide your classification.
You always need to classify.
Return only the result in the json format: {'y_pred': y_pred, 'explanation': explanation}.
The explanation should be brief."""
