import os
from src.encoders.medimageinsight import MedImageInsightEncoder
from dotenv import load_dotenv

load_dotenv()


encoders = {
    "MedImageInsight": MedImageInsightEncoder(
        api_link=os.environ["MEDIMAGEINSIGHT_URL"]
    )
}
